"""
torchrun --nproc_per_node=#GPUs --standalone <>_train_simclr.py --config config/simclr_cifar10.yaml
#
# The `torchrun` command is a wrapper around `python -m torch.distributed.run` that simplifies the process of launching distributed training jobs.
# The `--standalone` flag is used to run the script as a standalone script, rather than as a module.
# The `--config` flag is used to specify the path to the configuration file.
# The `nproc_per_node` flag is used to specify the number of GPUs per node.

Arguments can be found here:
https://github.com/pytorch/pytorch/blob/bbe803cb35948df77b46a2d38372910c96693dcd/torch/distributed/run.py#L401
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
# from torchlars import LARS
from torch.amp import autocast
import wandb

# distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# utils
from data_utils.dataloaders import get_dataset
from eval_utils.feature_extractor import FeatureExtractor
from eval_utils.nccc_utils import NCCCEvaluator
from eval_utils.geometry import GeometricEvaluator
from eval_utils.similarity_metrics import CenteredKernelAlignment, RepresentationSimilarityAnalysis

from utils.optimizer import LARS

# model
from models.simclr import SimCLR
from models.model_config import ModelConfig
from models.model_factory import generate_model_configs

import argparse
import yaml

from tqdm import tqdm
from collections import namedtuple, defaultdict
from typing import Literal

# # set seed
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)
torch.backends.cudnn.benchmark = True

# initialize distributed training
def ddp_setup():
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", 
                            world_size=world_size, 
                            rank=local_rank)
    

def cleanup():
  dist.destroy_process_group()

class ParallelTrainer:
    def __init__(
            self,
            models_config: dict,  # Pass the pre-configured models
            train_loader: torch.utils.data.DataLoader,
            save_every: int,
            log_every: int,
            test_loader: torch.utils.data.DataLoader = None,
            snapshot_dir: str = "checkpoints",
            **kwargs,
    ) -> None:
        
        # set seed
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_every = save_every
        self.log_every = log_every
        self.epochs_run = 0
        self.snapshot_dir = snapshot_dir

        self.models_config = models_config

        self.track_performance = kwargs.get("track_performance", False)
        self.settings = kwargs.get("settings", None)
        self.perform_cdnv = kwargs.get("perform_cdnv", False)
        self.perform_nccc = kwargs.get("perform_nccc", False)
        self.perform_rsa = kwargs.get("perform_rsa", False)
        self.perform_cka = kwargs.get("perform_cka", False) 
        self.wandb_defined = False
    
    def _load_snapshot(self, snapshot_dir: str) -> None:
        loc = f"cuda:{self.gpu_id}"
        # load the latest snapshot
        dir_list = os.listdir(snapshot_dir)
        if len(dir_list) == 0:
            print("No snapshots found!")
            return
        latest_snapshot = sorted(dir_list, reverse=True)[0]
        snapshot_path = os.path.join(snapshot_dir, latest_snapshot)

        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resume training from snapshot at epoch {self.epochs_run}")

    def _load_optimizer_scheduler(self, snapshot_dir: str) -> None:
        loc = f"cuda:{self.gpu_id}"
        # load the latest snapshot
        dir_list = os.listdir(snapshot_dir)
        if len(dir_list) == 0:
            print("No snapshots found!")
            return
        latest_snapshot = sorted(dir_list, reverse=True)[0]
        snapshot_path = os.path.join(snapshot_dir, latest_snapshot)

        snapshot = torch.load(snapshot_path, map_location=loc)
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER"])
    
    def _run_epoch(self, epoch: int) -> dict:
        print(f"[GPU {self.gpu_id}] Training epoch {epoch}...")
        
        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)
            print('Distributed Stratified Samplers set epoch method called.')

        # Initialize loss tracking
        losses_per_epoch = {name: 0.0 for name in self.models_config.keys()}
        
        for i, batch in enumerate(tqdm(self.train_loader)):
            # Train all models in a single loop
            for model_config in self.models_config.values():
                loss = model_config.train_step(batch, self.gpu_id)
                losses_per_epoch[model_config.name] += loss

            # Debug output for first epoch
            if epoch == 0:
                for name, total_loss in losses_per_epoch.items():
                    print(f"🧮 Accumulative batch loss at batch idx {i} for {name.upper()} model: {total_loss}")

        for model_config in self.models_config.values():
            model_config.scheduler.step()

        # Return average losses
        num_batches = len(self.train_loader)
        return {name: loss / num_batches for name, loss in losses_per_epoch.items()}

    def train(self, max_epochs: int) -> None:
        # Set all models to training mode
        for model_config in self.models_config.values():
            model_config.model.train()

        for epoch in range(self.epochs_run, max_epochs):
            # Run one epoch
            losses_per_epoch = self._run_epoch(epoch)
            
            # On GPU 0 do extra logging, snapshot saving, and evaluation
            if self.gpu_id == 0:
                # Save snapshots for all models
                if epoch % self.save_every == 0 or (epoch < 100 and epoch % 10 == 0):
                    for model_config in self.models_config.values():
                        model_config.save_snapshot(epoch, self.snapshot_dir)
                    print(f"Saved all models at epoch {epoch}")

                # Evaluate and log performance
                if epoch % self.log_every == 0:
                    for name, loss in losses_per_epoch.items():
                        print(f"{name.upper()} Loss per epoch: {loss}")
                    
                    if self.track_performance:
                        with torch.no_grad():
                            eval_outputs = self._run_evaluation()
                        
                        for name, loss in losses_per_epoch.items():
                            eval_outputs[name]['Loss'] = loss
                        
                        self.log_metrics(eval_outputs, epoch)

            # Barrier for distributed training
            if dist.get_world_size() > 1:
                dist.barrier()

        print("Training complete! 🎉")

    def _compute_cka(self, model_features, eval_outputs):
        print("--- Starting CKA Computation ---")

        embed_layer = 0 # 0 for h, 1 for g(h)
        cka_sample_size = 10000
        cka = CenteredKernelAlignment()
        print(f"Subsampling {cka_sample_size} images for CKA calculation due to memory constraints.")

        dcl_features = model_features['dcl'][embed_layer]
        num_samples = dcl_features.shape[0]

        indices = torch.randperm(num_samples)[:cka_sample_size]
        sub_dcl_features = dcl_features[indices]

        for model_name in self.models_config.keys():
            if model_name != "dcl":
                print(f"Computing CKA for {model_name}...")
                other_features = model_features[model_name][embed_layer]
                sub_other_features = other_features[indices]
                
                try:
                    cka_score = cka.cka_linear_kernel(sub_dcl_features, sub_other_features, device=self.settings.device)
                    print(f"\nCKA (Linear Kernel) between DCL and {model_name.upper()} features: {cka_score:.4f}")
                    eval_outputs[model_name]['CKA'] = cka_score
                except Exception as e:
                    print(f"Error computing CKA for {model_name}: {e}")
                    eval_outputs[model_name]['CKA'] = None
        
        print("\n--- CKA Computation Complete ---")

    def _compute_rsa(self, model_features, eval_outputs):
        print(f"\n=== Starting RSA Computation ===")

        embed_layer = 0 # 0 for h, 1 for g(h)
        rsa = RepresentationSimilarityAnalysis("cosine")

        dcl_features = model_features['dcl'][embed_layer]
        dcl_rdm = rsa.compute_rdm(dcl_features, chunk_size=1024)

        for model_name in self.models_config.keys():
            if model_name != "dcl":
                other_model_features = model_features[model_name][embed_layer]
                other_rdm = rsa.compute_rdm(other_model_features, chunk_size=1024)

                # Compute the RSA between the two RDMs
                rsa_pearson_score, p_value = rsa.compute_rsa(dcl_rdm, other_rdm, correlation_type='pearson')
                print(f"\nRSA (Pearson) Correlation between DCL and {model_name} features: {rsa_pearson_score:.4f} with p-value: {p_value:.4e}")

                eval_outputs[model_name]['RSA'] = rsa_pearson_score
                eval_outputs[model_name]['p-value'] = p_value

        print("\n--- RSA Computation Complete ---")

    def _run_evaluation(self):
        eval_outputs = {}
        model_features = {}

        print(f"\n=== Starting Evaluation ===")
        for model_name, model_config in self.models_config.items():
            model_config.model.eval()

            extractor = FeatureExtractor(model_config.model)
            features, labels = extractor.extract_features(self.train_loader)
            model_features[model_name] = features

            eval_outputs[model_name] = self._evaluate_single_model(model_config.model)
            model_config.model.train()

        if self.perform_rsa and len(self.models_config) >= 2:
            self._compute_rsa(model_features, eval_outputs)

        if self.perform_cka and len(self.models_config) >= 2:
            self._compute_cka(model_features, eval_outputs)

        return eval_outputs

    @torch.no_grad
    def _evaluate_single_model(self, model: torch.nn.Module):
        """
        Extracts features and computes all specified metrics for a single model.
        Return eval_outputs dictionary
        """
        # 1. Extract features
        extractor = FeatureExtractor(model)
        test_features, test_labels = extractor.extract_features(self.test_loader)
        # 2. Compute specified metrics
        eval_outputs = defaultdict()
        embedding_layer = 0 # 0 for h, 1 for g(h)
        if self.perform_nccc:
            evaluator = NCCCEvaluator(device=device)
            centers, selected_classes = evaluator.compute_class_centers(
                test_features[embedding_layer], test_labels,
                n_shot=100,
                repeat=1,
                selected_classes=None
            )

            nccc_accs = evaluator.evaluate(
                test_features[embedding_layer], test_labels, centers, selected_classes
            )
            eval_outputs['NCCC'] = nccc_accs[0]
            print(f"Evaluation accuracies: {nccc_accs}")
        if self.perform_cdnv:
            evaluator = GeometricEvaluator(self.settings.num_output_classes)
            cdnv = evaluator.compute_cdnv(test_features[embedding_layer], test_labels)
            dir_cdnv = evaluator.compute_directional_cdnv(test_features[embedding_layer], test_labels)
            eval_outputs['CDNV'] = cdnv
            eval_outputs['d-CDNV'] = dir_cdnv
            print(f'CDNV: {cdnv}, Dir-CDNV: {dir_cdnv}')
        # if self.perform_rsa:
        #     pass # TODO
        # if self.perform_cka:
        #     pass # TODO
        return eval_outputs
    
    def log_metrics(self, eval_outputs, cur_epoch):
        # Define metrics once
        if not self.wandb_defined:
            wandb.define_metric("epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")
            
            for model_name in self.models_config.keys():
                for metric in ["loss", "nccc", "cdnv", "d_cdnv", "rsa", "cka"]:
                    wandb.define_metric(f"{model_name}_{metric}", step_metric="epoch")
            
            self.wandb_defined = True

        # Collect all logs
        log_data = {
            "epoch": cur_epoch,
            "learning_rate": list(self.models_config.values())[0].optimizer.param_groups[0]["lr"]
        }

        # Log metrics for all models
        for model_name, outputs in eval_outputs.items():
            log_data[f"{model_name}_loss"] = outputs['Loss']
            
            if self.perform_nccc:
                log_data[f"{model_name}_nccc"] = outputs["NCCC"]
            
            if self.perform_cdnv:
                log_data[f"{model_name}_cdnv"] = torch.log10(torch.tensor(outputs["CDNV"]))
                log_data[f"{model_name}_d_cdnv"] = torch.log10(torch.tensor(outputs["d-CDNV"]))
            
            if self.perform_rsa and "RSA" in outputs:
                log_data[f"{model_name}_rsa"] = float(outputs["RSA"]) # convert from np.float to float
                log_data[f"{model_name}_p_value"] = float(outputs["p-value"])

            if self.perform_cka and "CKA" in outputs and outputs["CKA"] is not None:
                log_data[f"{model_name}_cka"] = float(outputs["CKA"])
        
        wandb.log(log_data)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='SimCLR Training')
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters
    experiment_name = config['experiment_name']
    method_type = config['method_type']
    supervision = config['supervision']

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']
    num_output_classes = config['dataset']['num_output_classes']
    
    batch_size = config['training']['batch_size']
    epochs = config['training']['num_epochs']
    lr = config['training']['lr']
    augmentations_type = config['training']['augmentations_type'] # imagenet or cifar or other dataset name
    augment_both = config['training']['augment_both']
    save_every = config['training']['save_every']
    log_every = config['training']['log_every']
    # save_model = config['training']['save_model']
    track_performance = config['training']['track_performance']
    multi_gpu = config['training']['multi_gpu']
    world_size = config['training']['world_size']

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    hidden_dim = config['model']['hidden_dim']
    projection_dim = config['model']['projection_dim']

    temperature = config['loss']['temperature']

    perform_rsa = config['evaluation']['perform_rsa']
    perform_cka = config['evaluation']['perform_cka']
    perform_cdnv = config['evaluation']['perform_cdnv']
    perform_nccc = config['evaluation']['perform_nccc']
    checkpoints_dir = config['evaluation']['checkpoints_dir']

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Settings = namedtuple("Settings", ["batch_size", "device", "num_output_classes"])
    settings = Settings(batch_size=batch_size, 
                        device=device,
                        num_output_classes=num_output_classes)

    # initialize distributed training
    ddp_setup()
    print(f"Local rank: {os.environ.get('LOCAL_RANK')}, World size: {os.environ.get('WORLD_SIZE')}")

    if dist.get_rank() == 0 and track_performance:
        wandb.init(
            project = "understanding_ssl_v2",
            config = {
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "lr": lr,
                "augment_both": augment_both,
                "world_size": world_size,
                "encoder_type": encoder_type,
                "width_multiplier": width_multiplier,
                "hidden_dim": hidden_dim,
                "projection_dim": projection_dim,
                "temperature": temperature,
            }
        )
    
    # load dataset
    world_size = int(os.environ.get('WORLD_SIZE'))
    print(f"Dataset: {dataset_name}")
    _, train_loader, _, test_loader, _, _ = get_dataset(dataset_name=dataset_name, 
                                    dataset_path=dataset_path,
                                    augment_both_views=augment_both,
                                    batch_size=batch_size, multi_gpu=multi_gpu,
                                    world_size=world_size, supervision='SCL', # sample with NSCL strategies
                                    test=True)
    # define model
    if encoder_type == 'resnet50':
        encoder = models.resnet50(weights=None)
    elif encoder_type == 'vit_b':
        encoder = models.VisionTransformer(
            patch_size=16 if 'imagenet' in dataset_name else 4,
            image_size=224 if 'imagenet' in dataset_name else 32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768 if 'imagenet' in dataset_name else 384,
            mlp_dim=3072 if 'imagenet' in dataset_name else 1536,
        )
    else:
        raise NotImplementedError(f"{encoder_type} not implemented")
    
    if method_type == 'simclr':
        # Calculate effective learning rate
        effective_lr = lr * world_size * (batch_size // 256)
        
        # Create ALL model configurations
        model_configs = generate_model_configs(
            encoder=encoder,
            supervision=supervision,
            temperature=temperature,
            device=device,
            effective_lr=effective_lr,
            total_epochs=epochs,
            gpu_id=int(os.environ.get('LOCAL_RANK')),
            # SimCLR specific parameters
            dataset=dataset_name,
            width_multiplier=width_multiplier,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            track_performance=track_performance,
            image_size=224 if 'imagenet' in dataset_name else 32,
            patch_size=16 if 'imagenet' in dataset_name else 4,
            stride=16 if 'imagenet' in dataset_name else 2,
            token_hidden_dim=768 if 'imagenet' in dataset_name else 384,
            mlp_dim=3072 if 'imagenet' in dataset_name else 1536,
        )
        
        # Create trainer with the model configurations
        trainer = ParallelTrainer(
            models_config=model_configs,
            train_loader=train_loader,
            test_loader=test_loader,
            save_every=save_every,
            log_every=log_every,
            snapshot_dir=checkpoints_dir,
            track_performance=track_performance,
            settings=settings,
            perform_cdnv=perform_cdnv,
            perform_nccc=perform_nccc,
            perform_rsa=perform_rsa,
            perform_cka=perform_cka,
            total_epochs=epochs
        )
    else:
        raise NotImplementedError(f"{method_type} not implemented")
    # breakpoint()
    trainer.train(epochs)
    dist.destroy_process_group()