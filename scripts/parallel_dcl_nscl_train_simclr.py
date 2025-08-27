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
from torch.amp import GradScaler, autocast
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
from utils.losses import NTXentLoss, DecoupledNTXentLoss, NegSupConLoss, SupConLoss
from utils.optimizer import LARS

# model
from models.simclr import SimCLR

import argparse
import yaml
from copy import deepcopy
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
            dcl_model: torch.nn.Module,
            nscl_model: torch.nn.Module,
            scl_model: torch.nn.Module,
            # ce_model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            criterion1: torch.nn.Module,
            criterion2: torch.nn.Module,
            criterion3: torch.nn.Module,
            # criterion4: torch.nn.Module,
            save_every: int,
            log_every: int,
            snapshot_dir: str,
            **kwargs,
    ) -> None:
        
        # set seed
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.dcl_model = dcl_model.to(f'cuda:{self.gpu_id}')
        self.nscl_model = nscl_model.to(f'cuda:{self.gpu_id}')
        self.scl_model = scl_model.to(f'cuda:{self.gpu_id}')
        # self.ce_model = ce_model.to(f'cuda:{self.gpu_id}')
        self.train_loader = train_loader
        self.test_loader = kwargs.get("test_loader", None)
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.criterion3 = criterion3
        # self.criterion4 = criterion4
        self.save_every = save_every
        self.log_every = log_every
        self.epochs_run = 0
        self.snapshot_dir = snapshot_dir
        # if os.path.exists(self.snapshot_dir):
            # TODO

        self.dcl_model = DDP(self.dcl_model, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.nscl_model = DDP(self.nscl_model, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.scl_model = DDP(self.scl_model, device_ids=[self.gpu_id], find_unused_parameters=True)
        # self.ce_model = DDP(self.ce_model, device_ids=[self.gpu_id], find_unused_parameters=True)

        # optimizer and scheduler
        effective_lr = kwargs.get("effective_lr", 0.1)
        total_epochs = kwargs.get("total_epochs", 100)
        self.optimizer1, self.scheduler1 = self._configure_optimizers(self.dcl_model, effective_lr, total_epochs)
        self.optimizer2, self.scheduler2 = self._configure_optimizers(self.nscl_model, effective_lr, total_epochs)
        self.optimizer3, self.scheduler3 = self._configure_optimizers(self.scl_model, effective_lr, total_epochs)
        # self.optimizer4, self.scheduler4 = self._configure_optimizers(self.ce_model, effective_lr, total_epochs)
        # if os.path.exists(self.snapshot_dir):
            # TODO
        #     print(f"Loaded optimizer and scheduler from {self.snapshot_dir}")

        self.track_performance = kwargs.get("track_performance", False)
        self.settings = kwargs.get("settings", None)
        self.perform_cdnv = kwargs.get("perform_cdnv", False)
        self.perform_nccc = kwargs.get("perform_nccc", False)
        self.perform_rsa = kwargs.get("perform_rsa", False)
        self.perform_cka = kwargs.get("perform_cka", False) 
        self.wandb_defined = False

        # mixed precision training
        self.scaler1 = GradScaler()
        self.scaler2 = GradScaler()
        self.scaler3 = GradScaler()
        # self.scaler4 = GradScaler()

    def _configure_optimizers(self, model, effective_lr,
                             total_epochs, warmup_epochs = 10):
        # LARS optimizer
        optimizer = LARS(
            model.parameters(),
            lr=effective_lr,
            weight_decay=1e-6,
            exclude_from_weight_decay=["batch_normalization", "bias"]
        )
        # Learning rate warmup + cosine decay
        scheduler = lr_scheduler.LambdaLR(
            optimizer, 
            lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) * 0.5 * (1 + torch.cos(torch.tensor(epoch / total_epochs * 3.1416)))
        )
        return optimizer, scheduler
    
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
    
    def _save_snapshot(self, model: nn.Module, epoch: int, 
                       optimizer, scheduler,
                       supervision: Literal["dcl", "nscl", "scl", "ce"]) -> None:
        snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER": optimizer.state_dict(),
            "SCHEDULER": scheduler.state_dict()
        }
        snapshot_dir = f'{self.snapshot_dir}/{supervision}'
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"snapshot_{epoch}.pth")
        torch.save(snapshot, snapshot_path)
        print(f"Saved model to {snapshot_path} at epoch {epoch}")

    def _run_epoch(self, epoch: int) -> None:
        print(f"[GPU {self.gpu_id}] Training epoch {epoch}...")
        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        # for custom distributed sampler
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)
            print('Distributed Startified Samplers set epoch method called.')

        loss1_per_epoch = 0.0
        loss2_per_epoch = 0.0
        loss3_per_epoch = 0.0
        loss4_per_epoch = 0.0
        for i, batch in enumerate(tqdm(self.train_loader)):
            self.optimizer1.zero_grad()
            # enable mixed precision training
            with autocast(device_type='cuda'):
              loss1 = self.dcl_model.module.run_one_batch(batch,
                                                self.criterion1,
                                                self.gpu_id)
            loss1_per_epoch += loss1.item()
            # backward + update using gradscaler
            self.scaler1.scale(loss1).backward()
            # torch.cuda.synchronize()
            self.scaler1.unscale_(self.optimizer1)  # Unscale gradients before clipping
            #clip model gradients
            clip_grad_norm_(self.dcl_model.parameters(), max_norm=1.0)
            self.scaler1.step(self.optimizer1)         
            self.scaler1.update()
            torch.cuda.synchronize()
            del loss1
            
            # repeat the steps for nscl model
            self.optimizer2.zero_grad()
            with autocast(device_type='cuda'):
                loss2 = self.nscl_model.module.run_one_batch(batch,
                                                             self.criterion2, 
                                                             # change this to self.criterion1 for debugging
                                                             # both models should have exact same losses
                                                             self.gpu_id)
            loss2_per_epoch += loss2.item()
            self.scaler2.scale(loss2).backward()
            self.scaler2.unscale_(self.optimizer2)
            clip_grad_norm_(self.nscl_model.parameters(), max_norm=1.0)
            self.scaler2.step(self.optimizer2)
            self.scaler2.update()
            torch.cuda.synchronize()
            del loss2

            # repeat the steps for scl model
            self.optimizer3.zero_grad()
            with autocast(device_type='cuda'):
                loss3 = self.scl_model.module.run_one_batch(batch,
                                                            self.criterion3, 
                                                            self.gpu_id)
            loss3_per_epoch += loss3.item()
            self.scaler3.scale(loss3).backward()
            self.scaler3.unscale_(self.optimizer3)
            clip_grad_norm_(self.scl_model.parameters(), max_norm=1.0)
            self.scaler3.step(self.optimizer3)
            self.scaler3.update()
            torch.cuda.synchronize()
            del loss3

            # # repeat the steps for ce model
            # self.optimizer4.zero_grad()
            # with autocast(device_type='cuda'):
            #     loss4 = self.ce_model.module.run_one_batch(batch,
            #                                                self.criterion4, 
            #                                                self.gpu_id)
            # loss4_per_epoch += loss4.item()
            # self.scaler4.scale(loss4).backward()
            # self.scaler4.unscale_(self.optimizer4)
            # clip_grad_norm_(self.scl_model.parameters(), max_norm=1.0)
            # self.scaler4.step(self.optimizer4)
            # self.scaler4.update()
            # torch.cuda.synchronize()
            # del loss4

            if epoch == 0:
                print(f"🧮 Accumulative batch loss at batch idx {i} for DCL model: {loss1_per_epoch}")
                print(f"🧮 Accumulative batch loss at batch idx {i} for NSCL model: {loss2_per_epoch}")
                print(f"🧮 Accumulative batch loss at batch idx {i} for SCL model: {loss3_per_epoch}")
                print(f"🧮 Accumulative batch loss at batch idx {i} for CE model: {loss4_per_epoch}")

        
        # update learning rate
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler3.step()
        # self.scheduler4.step()

        return (loss1_per_epoch / len(self.train_loader), loss2_per_epoch / len(self.train_loader),
                loss3_per_epoch / len(self.train_loader), loss4_per_epoch / len(self.train_loader))

    def train(self, max_epochs: int) -> None:
        self.dcl_model.train()
        self.nscl_model.train()
        self.scl_model.train()
        # self.ce_model.train()
        dcl_loss_per_epoch = 0.0
        nscl_loss_per_epoch = 0.0
        scl_loss_per_epoch = 0.0
        # ce_loss_per_epoch = 0.0
 
        for epoch in range(self.epochs_run, max_epochs):
            # run one epoch
            dcl_loss_per_epoch, nscl_loss_per_epoch, scl_loss_per_epoch, ce_loss_per_epoch = self._run_epoch(epoch)
            # On GPU 0 do extra logging, snapshot saving, and evaluation
            if self.gpu_id == 0:
                # Save a snapshot
                if epoch % self.save_every == 0 or (epoch < 100 and epoch % 10 == 0):
                    self._save_snapshot(self.dcl_model, epoch, self.optimizer1, self.scheduler1, supervision='dcl')
                    self._save_snapshot(self.nscl_model, epoch, self.optimizer2, self.scheduler2, supervision='nscl')
                    self._save_snapshot(self.scl_model, epoch, self.optimizer3, self.scheduler3, supervision='scl')
                    # self._save_snapshot(self.ce_model, epoch, self.optimizer4, self.scheduler4, supervision='ce')
                    print(f"Saved model at epoch {epoch}")

                # Evaluate and log performance every self.save_every epochs
                if epoch % self.log_every == 0:
                    print(f"SSL Loss per epoch: {dcl_loss_per_epoch}")
                    print(f"NSCL Loss per epoch: {nscl_loss_per_epoch}")
                    print(f"SCL Loss per epoch: {scl_loss_per_epoch}")
                    # print(f"CE Loss per epoch: {ce_loss_per_epoch}")
                    if self.track_performance:
                        with torch.no_grad():
                            dcl_eval_outputs, nscl_eval_outputs, scl_eval_outputs, ce_eval_outputs = self._run_evaluation()
                        dcl_eval_outputs['Loss'] = dcl_loss_per_epoch
                        nscl_eval_outputs['Loss'] = nscl_loss_per_epoch
                        scl_eval_outputs['Loss'] = scl_loss_per_epoch
                        # ce_eval_outputs['Loss'] = ce_loss_per_epoch
                        self.log_metrics(dcl_eval_outputs, nscl_eval_outputs, scl_eval_outputs, ce_eval_outputs, epoch)

            # Optionally, if using distributed training, you might call a barrier here:
            if dist.get_world_size() > 1:
                dist.barrier()

        print("Training complete! 🎉")

    def _run_evaluation(self):
        # Evaluate SSL Model
        self.dcl_model.eval()
        dcl_eval_outputs = self._evaluate_single_model(self.dcl_model)
        # Evaluate NSCL Model
        self.nscl_model.eval()
        nscl_eval_outputs = self._evaluate_single_model(self.nscl_model)
        # Evaluate SCL Model
        self.scl_model.eval()
        scl_eval_outputs = self._evaluate_single_model(self.scl_model)
        # # Evaluate CE Model
        # self.ce_model.eval()
        # ce_eval_outputs = self._evaluate_single_model(self.ce_model)
        ce_eval_outputs = None # TODO

        # set models back to training mode
        self.dcl_model.train()
        self.nscl_model.train()
        self.scl_model.train()
        # self.ce_model.train()

        return dcl_eval_outputs, nscl_eval_outputs, scl_eval_outputs, ce_eval_outputs

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
            # make sure to use above selected classes while evaluating
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
        if self.perform_rsa:
            pass # TODO
        if self.perform_cka:
            pass # TODO
        return eval_outputs
    
    def log_metrics(self, dcl_eval_outputs, nscl_eval_outputs, scl_eval_outputs, ce_eval_outputs, cur_epoch):
        # define epoch as x-axis
        if not self.wandb_defined:
            wandb.define_metric("epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")
            for model_prefix in ["dcl", "nscl", "scl", "ce"]:
                wandb.define_metric(f"{model_prefix}_loss", step_metric="epoch")
                wandb.define_metric(f"{model_prefix}_nccc", step_metric="epoch")
                wandb.define_metric(f"{model_prefix}_cdnv", step_metric="epoch")
                wandb.define_metric(f"{model_prefix}_d_cdnv", step_metric="epoch")
                wandb.define_metric(f"{model_prefix}_rsa", step_metric="epoch")
                wandb.define_metric(f"{model_prefix}_cka", step_metric="epoch")
            self.wandb_defined = True

        # collect all logs in one dictionary
        log_data = {
                "epoch": cur_epoch,
                "learning_rate": self.optimizer1.param_groups[0]["lr"]
                }
        
        # create eval_outputs_map
        eval_outputs_map = {
            "dcl": dcl_eval_outputs,
            "nscl": nscl_eval_outputs,
            "scl": scl_eval_outputs,
            # "ce": ce_eval_outputs
        }
        for model_prefix, eval_outputs in eval_outputs_map.items():
            log_data[f"{model_prefix}_loss"] = eval_outputs['Loss']
            if self.perform_nccc:
                log_data[f"{model_prefix}_nccc"] = eval_outputs["NCCC"]
            if self.perform_cdnv:
                log_data[f"{model_prefix}_cdnv"] = torch.log10(torch.tensor(eval_outputs["CDNV"]))
                log_data[f"{model_prefix}_d_cdnv"] = torch.log10(torch.tensor(eval_outputs["d-CDNV"]))
            if self.perform_rsa:
                pass # TODO
            if self.perform_cka:
                pass # TODO
        # log all metrics
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
        encoder = torchvision.models.resnet50(weights=None)
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
        dcl_model = SimCLR(model=encoder,
                           dataset=dataset_name,
                           width_multiplier=width_multiplier,
                           hidden_dim=hidden_dim,
                           projection_dim=projection_dim,
                           track_performance=track_performance,
                           # hyperparams for ViT
                           image_size = 224 if 'imagenet' in dataset_name else 32,
                           patch_size = 16 if 'imagenet' in dataset_name else 4,
                           stride = 16 if 'imagenet' in dataset_name else 2,
                           token_hidden_dim = 768 if 'imagenet' in dataset_name else 384,
                           mlp_dim = 3072 if 'imagenet' in dataset_name else 1536,
                           )
        nscl_model = deepcopy(dcl_model)
        nscl_model.encoder.remove_hook()
        nscl_model.encoder._register_hook()
        scl_model = deepcopy(dcl_model)
        scl_model.encoder.remove_hook()
        scl_model.encoder._register_hook()
    else:
        raise NotImplementedError(f"{method_type} not implemented")

    # convert all BatchNorm layers to SyncBatchNorm
    dcl_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dcl_model)
    nscl_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(nscl_model)
    scl_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(scl_model)
    # dist.barrier() # wait for all processes to catch up

    # define loss & optimizer
    if supervision == 'DCL':
        print("Using Decoupled Contrastive Learning")
        criterion1 = DecoupledNTXentLoss(temperature, device) 
    elif supervision == 'CL':
        print("Using Contrastive Learning")
        criterion1 = NTXentLoss(temperature, device)
    else:
        raise NotImplementedError(f"{supervision} not implemented")
    
    print("Using Negatives-only Supervised Contrastive Learning")
    criterion2 = NegSupConLoss(temperature, device)
    print("Using Supervised Contrastive Learning")
    criterion3 = SupConLoss(temperature, device)
    # criterion4 = nn.CrossEntropyLoss(reduction='mean')
    effective_lr = lr*world_size*(batch_size//256)
    # train model
    trainer = ParallelTrainer(
        dcl_model=dcl_model,
        nscl_model=nscl_model,
        scl_model=scl_model,
        # ce_model=ce_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion1=criterion1,
        criterion2=criterion2,
        criterion3=criterion3,
        # criterion4=criterion4,
        save_every=save_every,
        log_every=log_every,
        snapshot_dir=checkpoints_dir,
        track_performance=track_performance,
        effective_lr = effective_lr,
        settings = settings,
        perform_cdnv = perform_cdnv,
        perform_nccc = perform_nccc,
        total_epochs = epochs
    )
    # breakpoint()
    trainer.train(epochs)
    dist.destroy_process_group()