import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
torch.set_default_dtype(torch.float32)

import torchvision.models as models
import sys, os, argparse, yaml, pandas as pd
from tqdm import tqdm

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and models.
from data_utils.dataloaders import get_dataset
from eval_utils.feature_extractor import FeatureExtractor
from eval_utils.linear_probe_utils import LinearProbeEvaluator
from models.simclr import SimCLR, SimCLRWithClassificationHead

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
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="General Evluation Script")
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    parser.add_argument('--ckpt_path', '-ckpt', required=True, default=None,
                        help='path to model checkpoints')
    parser.add_argument('--output_path', '-out', required=True, default=None,
                        help='path to save logs')
    parser.add_argument('--supervision', default='dcl', type=str, choices=['dcl', 'nscl', 'scl', 'ce'],
                        help='which model to evaluate')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters required for evaluation
    experiment_name = config['experiment_name']
    method_type = config['method_type']

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
    train_dataset, train_loader, test_dataset, test_loader, train_labels, test_labels = get_dataset(dataset_name=dataset_name, 
                                                                    dataset_path=dataset_path,
                                                                    augment_both_views=augment_both,
                                                                    batch_size=batch_size, test=True)
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
        model = SimCLR(model=encoder,
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
    
    if args.supervision == 'ce':
        model = SimCLRWithClassificationHead(
            simclr_model=model,
            num_classes=num_output_classes
        )
    
    # load model checkpoint
    checkpoints_dir = f'{args.ckpt_path}/{args.supervision}'
    print(f"Loading checkpoints from {checkpoints_dir}")
    checkpoint_files = os.listdir(checkpoints_dir)
    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

    # Output logging
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    log_file = os.path.join(args.output_path, f'{args.supervision}_lp.csv')
    log_columns = ['Epoch', 'LP Train', 'LP Test']
    if not os.path.exists(log_file):
        df = pd.DataFrame(columns=log_columns)
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)

    for ssl_ckpt in sorted_checkpoints:
        epoch = int(ssl_ckpt.split('_')[-1].split('.')[0])
        epochs_to_eval = [800,1000,1200,1800,2000,2200,2400]
        if epoch in df['Epoch'].values or epoch not in epochs_to_eval: 
            print(f"Epoch {epoch} already evaluated. Skipping.")
            continue
        print(f'\nEvaluating Epoch {epoch}')
        snapshot_path = f'{args.ckpt_path}/{args.supervision}/snapshot_{epoch}.pth'
        model = load_snapshot(snapshot_path, model, device)
        model.eval()
        model = freeze_model(model)
        print("Models frozen for feature extraction")
        # --- Feature Extraction ---
        emb_layer = 0 # 0 for h and 1 for g(h)
        extractor = FeatureExtractor(model)
        train_features, train_labels = extractor.extract_features(train_loader)
        test_features, test_labels = extractor.extract_features(test_loader)

        # LP Evaluation
        evaluator = LinearProbeEvaluator(
            train_features[emb_layer], train_labels,
            test_features[emb_layer], test_labels,
            num_output_classes
        )
        train_acc, test_acc = evaluator.evaluate(repeat=1)
        print(f"LP Train Accuracy: {np.mean(train_acc)*100:.2f}")
        print(f"LP Test Accuracy: {np.mean(test_acc)*100:.2f}")
        # log results
        new_entry = {
            'Epoch': epoch,
            'NCCC Train': np.mean(train_acc)*100,
            'NCCC Test': np.mean(test_acc)*100
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df = df.sort_values(by='Epoch')
    df.to_csv(log_file, index=False)