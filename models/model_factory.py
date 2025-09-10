import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from copy import deepcopy
from models.model_config import ModelConfig
from models.simclr import SimCLR, SimCLRWithClassificationHead
from utils.losses import NTXentLoss, DecoupledNTXentLoss, NegSupConLoss, SupConLoss, MultiViewCrossEntropyLoss
from utils.optimizer import LARS

MODEL_CONFIGS = {
    'dcl': {
        'criterion_type': 'primary', # primary model type
        'description': 'Decoupled Contrastive Learning'
    },
    'nscl': {
        'criterion_type': 'neg_sup_con',
        'description': 'Negatives-only Supervised Contrastive Learning'
    },
    'scl': {
        'criterion_type': 'sup_con',
        'description': 'Supervised Contrastive Learning'
    },
    'ce': {
        'criterion_type': 'cross_entropy',
        'description': 'Cross Entropy Learning'
    }
}

CRITERION_MAPPING = {
    'primary': {
        'DCL': DecoupledNTXentLoss,
        'CL': NTXentLoss,
    },
    'neg_sup_con': NegSupConLoss,
    'sup_con': SupConLoss,
    'cross_entropy': MultiViewCrossEntropyLoss,  # Easy to add
}

def generate_model_configs(encoder, supervision, temperature, device, effective_lr, total_epochs, gpu_id, **model_kwargs):
    """Factory function to create all model configurations"""
    base_simclr_model = SimCLR(
        model=encoder,
        **model_kwargs
    )
    
    base_simclr_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_simclr_model)
    model_configs = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"Setting up {config['description']}")
                
        if model_name == 'dcl':
            model = base_simclr_model
        elif model_name == 'ce':
            # For CE, wrap the copied architecture in our new supervised model
            model_arch_copy = deepcopy(base_simclr_model)
            model_arch_copy.encoder.remove_hook()
            model_arch_copy.encoder._register_hook()
            model = SimCLRWithClassificationHead(
                simclr_model=model_arch_copy,
                num_classes=model_kwargs.get('num_output_classes', 10)
            )
        else:
            # For contrastive methods, use the copied architecture directly
            model_arch_copy = deepcopy(base_simclr_model) 
            model = model_arch_copy
            model.encoder.remove_hook()
            model.encoder._register_hook()
                   
        model = model.to(f'cuda:{gpu_id}')
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        
        criterion_type = config['criterion_type']
        if criterion_type == 'primary':
            if supervision == 'DCL':
                criterion = DecoupledNTXentLoss(temperature, device)
            elif supervision == 'CL':
                criterion = NTXentLoss(temperature, device)
            else:
                raise NotImplementedError(f"{supervision} not implemented")
        else:
            criterion_class = CRITERION_MAPPING[criterion_type]
            criterion = criterion_class(temperature, device)
        
        optimizer, scheduler = _configure_optimizers(model, effective_lr, total_epochs)
        
        scaler = GradScaler()
        
        model_configs[model_name] = ModelConfig(
            model_name, model, criterion, optimizer, scheduler, scaler
        )
    
    return model_configs

def _configure_optimizers(model, effective_lr, total_epochs, warmup_epochs=10):
    optimizer = LARS(
        model.parameters(),
        lr=effective_lr,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"]
    )
    
    scheduler = lr_scheduler.LambdaLR(
        optimizer, 
        lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) * 0.5 * (1 + torch.cos(torch.tensor(epoch / total_epochs * 3.1416)))
    )
    
    return optimizer, scheduler
