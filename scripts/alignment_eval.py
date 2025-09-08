import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
torch.set_default_dtype(torch.float32)

import torchvision.models as models
import sys, os, argparse, yaml, pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and models.
from data_utils.dataloaders import get_dataset
from eval_utils.feature_extractor import FeatureExtractor
from eval_utils.similarity_metrics import RepresentationSimilarityAnalysis, CenteredKernelAlignment
from models.simclr import SimCLR, SimCLRWithClassificationHead

def ddp_setup():
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", 
                            world_size=world_size, 
                            rank=local_rank)
    

def cleanup():
  dist.destroy_process_group()

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

set_seed(42)

def load_snapshot(snapshot_path, model, device):
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    state_dict = snapshot['MODEL_STATE']
    epochs_trained = snapshot['EPOCHS_RUN']
    print(f"Loaded model from epoch {epochs_trained}")
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
    return model

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

RSA=True
CKA=True

if __name__ == '__main__':
    ddp_setup()
    parser = argparse.ArgumentParser(description="General Evluation Script")
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    parser.add_argument('--ckpt_path', '-ckpt', required=True, default=None,
                        help='path to model checkpoints')
    parser.add_argument('--output_path', '-out', required=True, default=None,
                        help='path to save logs')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters required for evaluation
    experiment_name = config['experiment_name']
    method_type = config['method_type']
    supervision = config['supervision']

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']
    num_output_classes = config['dataset']['num_output_classes']

    batch_size = config['training']['batch_size']
    augment_both = config['training']['augment_both']
    

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    projection_dim = config['model']['projection_dim']
    hidden_dim = config['model']['hidden_dim']

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get dataset
    augment_both = False # override for evaluation
    train_dataset, _, test_dataset, _, train_labels, test_labels = get_dataset(dataset_name=dataset_name, 
                                                                    dataset_path=dataset_path,
                                                                    augment_both_views=augment_both,
                                                                    batch_size=batch_size, test=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    num_train_images = len(train_dataset)
    num_test_images = len(test_dataset)


    # load model
    if encoder_type == 'resnet50':
        encoder = models.resnet50(pretrained=False)
    elif encoder_type == 'vit_b':
        image_size = 224 if 'imagenet' in dataset_name else 32
        # encoder = models.vit_b_16(weights=None, image_size=image_size)
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
        # use_old = True if dataset_name == 'imagenet' and supervision == 'SSL' else False,
        ssl_model = SimCLR(model=encoder,
                           dataset=dataset_name,
                           width_multiplier=width_multiplier,
                           hidden_dim=hidden_dim,
                           projection_dim=projection_dim,
                           # hyperparams for ViT
                           image_size = 224 if 'imagenet' in dataset_name else 32,
                           patch_size = 16 if 'imagenet' in dataset_name else 4,
                           stride = 16 if 'imagenet' in dataset_name else 2,
                           token_hidden_dim = 768 if 'imagenet' in dataset_name else 384,
                           mlp_dim = 3072 if 'imagenet' in dataset_name else 1536,
                           )
    else:
        raise NotImplementedError(f"{method_type} not implemented")
    
    # deepcopy NSCL model
    nscl_model = deepcopy(ssl_model)
    nscl_model.encoder.remove_hook()
    nscl_model.encoder._register_hook()

    scl_model = deepcopy(ssl_model)
    scl_model.encoder.remove_hook()
    scl_model.encoder._register_hook()

    ce_model_arch = deepcopy(ssl_model)
    ce_model_arch.encoder.remove_hook()
    ce_model_arch.encoder._register_hook()
    ce_model = SimCLRWithClassificationHead(
        simclr_model=ce_model_arch,
        num_classes=num_output_classes
    )
    
    # load model checkpoint
    checkpoints_dir = f'{args.ckpt_path}/dcl'
    print(f"Loading checkpoints from {checkpoints_dir}")
    # load SSL model
    checkpoint_files = os.listdir(checkpoints_dir)
    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Output logging
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    log_file = os.path.join(args.output_path, dataset_name + '_train_alignment.csv')
    # log_file = os.path.join(args.output_path, dataset_name + '_test_alignment.csv')
    log_columns = ['Epoch', 'NSCL_RSA', 'SCL_RSA', 'NSCL_CKA', 'SCL_CKA', 'CE_RSA', 'CE_CKA']
    if not os.path.exists(log_file):
        df = pd.DataFrame(columns=log_columns)
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
    
    ssl_model = ssl_model.to(device)
    nscl_model = nscl_model.to(device)
    scl_model = scl_model.to(device)
    ce_model = ce_model.to(device)
    ssl_model = DDP(ssl_model, device_ids=[0], find_unused_parameters=True)
    nscl_model = DDP(nscl_model, device_ids=[0], find_unused_parameters=True)
    scl_model = DDP(scl_model, device_ids=[0], find_unused_parameters=True)
    ce_model = DDP(ce_model, device_ids=[0], find_unused_parameters=True)

    for ssl_ckpt in sorted_checkpoints:
        epoch = int(ssl_ckpt.split('_')[-1].split('.')[0])
        if epoch in df['Epoch'].values:
            print(f"Epoch {epoch} already evaluated. Skipping.")
            continue
        print(f'\nEvaluating Epoch {epoch}')
        ssl_snapshot_path = f'{args.ckpt_path}/dcl/snapshot_{epoch}.pth'
        ssl_model = load_snapshot(ssl_snapshot_path, ssl_model, device)
        ssl_model.eval()
        nscl_snapshot_path = f'{args.ckpt_path}/nscl/snapshot_{epoch}.pth'
        nscl_model = load_snapshot(nscl_snapshot_path, nscl_model, device)
        nscl_model.eval()
        scl_snapshot_path = f'{args.ckpt_path}/scl/snapshot_{epoch}.pth'
        scl_model = load_snapshot(scl_snapshot_path, scl_model, device)
        scl_model.eval()
        ce_snapshot_path = f'{args.ckpt_path}/ce/snapshot_{epoch}.pth'
        ce_model = load_snapshot(ce_snapshot_path, ce_model, device)
        ce_model.eval()
        # freeze models
        ssl_model = freeze_model(ssl_model)
        nscl_model = freeze_model(nscl_model)
        scl_model = freeze_model(scl_model)
        ce_model = freeze_model(ce_model)
        print("Models frozen for feature extraction")
        # --- Feature Extraction ---
        emb_layer = 0 # 0 for h and 1 for g(h)
        dcl_extractor = FeatureExtractor(ssl_model)
        dcl_train_features, _ = dcl_extractor.extract_features(test_loader)

        nscl_extractor = FeatureExtractor(nscl_model)
        nscl_train_features, _ = nscl_extractor.extract_features(test_loader)

        scl_extractor = FeatureExtractor(scl_model)
        scl_train_features, _ = scl_extractor.extract_features(test_loader)

        ce_extractor = FeatureExtractor(ce_model)
        ce_train_features, _ = ce_extractor.extract_features(test_loader)

        if RSA:
            rsa_eval = RepresentationSimilarityAnalysis(metric='cosine')
            # 1. Compute RDMs for both feature sets
            dcl_rdm = rsa_eval.compute_rdm(dcl_train_features[emb_layer])
            nscl_rdm = rsa_eval.compute_rdm(nscl_train_features[emb_layer])
            scl_rdm = rsa_eval.compute_rdm(scl_train_features[emb_layer])
            ce_rdm = rsa_eval.compute_rdm(ce_train_features[emb_layer])

            # 2. Compute RSA score (correlation between RDMs)
            nscl_rsa_score, p_value = rsa_eval.compute_rsa(dcl_rdm, nscl_rdm, correlation_type='pearson')
            print(f"\nRSA (Pearson) Correlation between DCL and NSCL features: {nscl_rsa_score:.4f} with p-value: {p_value:.4e}")
            scl_rsa_score, p_value = rsa_eval.compute_rsa(dcl_rdm, scl_rdm, correlation_type='pearson')
            print(f"RSA (Pearson) Correlation between DCL and SCL features: {scl_rsa_score:.4f} with p-value: {p_value:.4e}")
            ce_rsa_score, p_value = rsa_eval.compute_rsa(dcl_rdm, ce_rdm, correlation_type='pearson')
            print(f"RSA (Pearson) Correlation between DCL and CE features: {ce_rsa_score:.4f} with p-value: {p_value:.4e}")

            print("\n--- RSA Computation Complete ---")
        
        if CKA:
            # --- CKA Execution ---
            print("--- Starting CKA Computation ---")

            cka_sample_size = 10000

            print(f"Subsampling {cka_sample_size} images for CKA calculation due to memory constraints.")
            # Ensure consistent subsampling for both feature sets
            indices = torch.randperm(num_test_images)[:cka_sample_size]
            sub_dcl_features = dcl_train_features[emb_layer][indices]
            sub_nscl_features = nscl_train_features[emb_layer][indices]
            sub_scl_features = scl_train_features[emb_layer][indices]
            sub_ce_features = ce_train_features[emb_layer][indices]

            # Perform CKA calculation
            cka_eval = CenteredKernelAlignment(kernel='linear')
            try:
                nscl_cka_score = cka_eval.cka_linear_kernel(sub_dcl_features, sub_nscl_features, device=device)
                print(f"\nCKA (Linear Kernel) between DCL and NSCL features: {nscl_cka_score:.4f}")
                scl_cka_score = cka_eval.cka_linear_kernel(sub_dcl_features, sub_scl_features, device=device)
                print(f"CKA (Linear Kernel) between DCL and SCL features: {scl_cka_score:.4f}")
                ce_cka_score = cka_eval.cka_linear_kernel(sub_dcl_features, sub_ce_features, device=device)
                print(f"CKA (Linear Kernel) between DCL and CE features: {ce_cka_score:.4f}")
            except Exception as e:
                print(f"An error occurred during CKA calculation: {e}")
                print("This is likely due to memory limitations. Try reducing 'cka_sample_size' further.")

            print("\n--- CKA Computation Complete ---")
        
        # log results
        new_entry = {
            'Epoch': epoch,
            'NSCL_RSA': nscl_rsa_score if RSA else None,
            'NSCL_CKA': nscl_cka_score if CKA else None,
            'SCL_RSA': scl_rsa_score if RSA else None,
            'SCL_CKA': scl_cka_score if CKA else None,
            'CE_RSA': ce_rsa_score if RSA else None,
            'CE_CKA': ce_cka_score if CKA else None,
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df = df.sort_values(by='Epoch')
    df.to_csv(log_file, index=False)
    cleanup()