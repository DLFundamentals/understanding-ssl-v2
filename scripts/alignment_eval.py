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

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and models.
from data_utils.dataloaders import get_dataset
from eval_utils.feature_extractor import FeatureExtractor
from eval_utils.similarity_metrics import compute_cka, compute_rsa
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

def initialize_logging(output_path, mode='train'):
    log_columns = ['Epoch', 'NSCL_RSA', 'SCL_RSA', 'NSCL_CKA', 'SCL_CKA', 'CE_RSA', 'CE_CKA']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_file = os.path.join(output_path, f'{mode}_alignment.csv')
    if not os.path.exists(log_file):
        df = pd.DataFrame(columns=log_columns)
        df.to_csv(log_file, index=False)
    else:
        print(f"Log file {log_file} exists. Resuming logging.")
        df = pd.read_csv(log_file)
    return log_file, df

RSA=True
CKA=True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="General Evluation Script")
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    parser.add_argument('--ckpt_path', '-ckpt', required=True,
                        help='path to model checkpoints')
    parser.add_argument('--output_path', '-out', required=True,
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
        use_old = True if dataset_name == 'imagenet' and supervision == 'SSL' else False,
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
    # deepcopy SCL model
    scl_model = deepcopy(ssl_model)
    scl_model.encoder.remove_hook()
    scl_model.encoder._register_hook()
    # deepcopy CE model
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
    train_log_file, train_df = initialize_logging(args.output_path, mode='train')
    test_log_file, test_df = initialize_logging(args.output_path, mode='test')

    def process_model(model_name, ssl_model, device, ckpt_path, epoch):
        """
        Loads, freezes, and extracts features for a given model.
        """
        model_path = f'{ckpt_path}/{model_name}/snapshot_{epoch}.pth'
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint not found for {model_name} at epoch {epoch}. Skipping.")
            return None, None

        model = load_snapshot(model_path, ssl_model, device)
        model = freeze_model(model)
        print(f"Model {model_name} frozen for feature extraction.")

        extractor = FeatureExtractor(model)
        train_features, _ = extractor.extract_features(train_loader)
        test_features, _ = extractor.extract_features(test_loader)
        return train_features, test_features
    
    models_to_evaluate = {
        'dcl': ssl_model,
        'nscl': nscl_model,
        'scl': scl_model,
        'ce': ce_model
    }

    for ssl_ckpt in sorted_checkpoints:
        epoch = int(ssl_ckpt.split('_')[-1].split('.')[0])
        if epoch in train_df['Epoch'].values:
            print(f"Epoch {epoch} already evaluated. Skipping.")
            continue
        print(f'\nEvaluating Epoch {epoch}')
        features = {}
        for name, model_arch in models_to_evaluate.items():
            train_feats, test_feats = process_model(name, model_arch, device, args.ckpt_path, epoch)
            if train_feats is not None:
                features[name] = {'train': train_feats, 'test': test_feats}
        
        emb_layer = 0 # 0 for h and 1 for g(h)
        if RSA:
            print("--- Starting RSA Computation ---")
            nscl_train_rsa_score = compute_rsa(features['dcl']['train'], features['nscl']['train'],
                                        model_name1='dcl', model_name2='nscl',
                                        embed_layer=emb_layer, device=device)
            nscl_test_rsa_score = compute_rsa(features['dcl']['test'], features['nscl']['test'],
                                        model_name1='dcl', model_name2='nscl',
                                        embed_layer=emb_layer, device=device)
            scl_train_rsa_score = compute_rsa(features['dcl']['train'], features['scl']['train'],
                                        model_name1='dcl', model_name2='scl',
                                        embed_layer=emb_layer, device=device)
            scl_test_rsa_score = compute_rsa(features['dcl']['test'], features['scl']['test'],
                                        model_name1='dcl', model_name2='scl',
                                        embed_layer=emb_layer, device=device)
            ce_train_rsa_score = compute_rsa(features['dcl']['train'], features['ce']['train'],
                                        model_name1='dcl', model_name2='ce',
                                        embed_layer=emb_layer, device=device)
            ce_test_rsa_score = compute_rsa(features['dcl']['test'], features['ce']['test'],
                                        model_name1='dcl', model_name2='ce',
                                        embed_layer=emb_layer, device=device)
            print("\n--- RSA Computation Complete ---")
        if CKA:
            # --- CKA Execution ---
            print("--- Starting CKA Computation ---")
            nscl_train_cka_score = compute_cka(features['dcl']['train'], features['nscl']['train'],
                                        model_name1='dcl', model_name2='nscl',
                                        embed_layer=emb_layer, device=device)
            nscl_test_cka_score = compute_cka(features['dcl']['test'], features['nscl']['test'],
                                        model_name1='dcl', model_name2='nscl',
                                        embed_layer=emb_layer, device=device)
            scl_train_cka_score = compute_cka(features['dcl']['train'], features['scl']['train'],
                                        model_name1='dcl', model_name2='scl',
                                        embed_layer=emb_layer, device=device)
            scl_test_cka_score = compute_cka(features['dcl']['test'], features['scl']['test'],
                                        model_name1='dcl', model_name2='scl',
                                        embed_layer=emb_layer, device=device)
            ce_train_cka_score = compute_cka(features['dcl']['train'], features['ce']['train'],
                                        model_name1='dcl', model_name2='ce',
                                        embed_layer=emb_layer, device=device)
            ce_test_cka_score = compute_cka(features['dcl']['test'], features['ce']['test'],
                                        model_name1='dcl', model_name2='ce',
                                        embed_layer=emb_layer, device=device)

            print("\n--- CKA Computation Complete ---")
        
        # log results
        train_new_entry = {
            'Epoch': epoch,
            'NSCL_RSA': nscl_train_rsa_score if RSA else None,
            'NSCL_CKA': nscl_train_cka_score if CKA else None,
            'SCL_RSA': scl_train_rsa_score if RSA else None,
            'SCL_CKA': scl_train_cka_score if CKA else None,
            'CE_RSA': ce_train_rsa_score if RSA else None,
            'CE_CKA': ce_train_cka_score if CKA else None,
        }
        train_df = pd.concat([train_df, pd.DataFrame([train_new_entry])], ignore_index=True)
        test_new_entry = {
            'Epoch': epoch,
            'NSCL_RSA': nscl_test_rsa_score if RSA else None,
            'NSCL_CKA': nscl_test_cka_score if CKA else None,
            'SCL_RSA': scl_test_rsa_score if RSA else None,
            'SCL_CKA': scl_test_cka_score if CKA else None,
            'CE_RSA': ce_test_rsa_score if RSA else None,
            'CE_CKA': ce_test_cka_score if CKA else None,
        }
        test_df = pd.concat([test_df, pd.DataFrame([test_new_entry])], ignore_index=True)
    train_df = train_df.sort_values(by='Epoch')
    train_df.to_csv(train_log_file, index=False)
    test_df = test_df.sort_values(by='Epoch')
    test_df.to_csv(test_log_file, index=False)