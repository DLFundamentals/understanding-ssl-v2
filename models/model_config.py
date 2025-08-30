import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast
import os

class ModelConfig:
    def __init__(self, name, model, criterion, optimizer, scheduler, scaler):
        self.name = name
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
    
    def train_step(self, batch, gpu_id):
        self.optimizer.zero_grad()
        # get the inputs
        view1, view2, labels = batch
        # skip the batch with only 1 image
        if view1.size(0) < 2:
            return 0
        view1, view2 = view1.to(gpu_id), view2.to(gpu_id)
        labels = labels.to(gpu_id)
        
        torch.autograd.set_detect_anomaly(True)
        with autocast(device_type='cuda'):
            if self.name == 'ce':
                logits1 = self.model.module(view1, mode='train')
                logits2 = self.model.module(view2, mode='train')
                loss = self.criterion(logits1, logits2, labels)
            else:
                view1_features, view1_proj = self.model.module(view1)
                view2_features, view2_proj = self.model.module(view2)
                loss = self.criterion(view1_proj, view2_proj, labels)
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        torch.cuda.synchronize()
        
        return loss.item()
    
    def save_snapshot(self, epoch, snapshot_dir):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER": self.optimizer.state_dict(),
            "SCHEDULER": self.scheduler.state_dict()
        }
        model_snapshot_dir = f'{snapshot_dir}/{self.name}'
        os.makedirs(model_snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(model_snapshot_dir, f"snapshot_{epoch}.pth")
        torch.save(snapshot, snapshot_path)
        print(f"Saved {self.name} model to {snapshot_path} at epoch {epoch}")