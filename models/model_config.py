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
        
        with autocast(device_type='cuda'):
            if self.name == 'ce':
                loss = self.model.module.run_one_batch(batch, self.criterion, mode='train', device=gpu_id)
            else:
                loss = self.model.module.run_one_batch(batch, self.criterion, device=gpu_id)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        torch.cuda.synchronize()

        return loss.item()

    def save_snapshot(self, epoch: int, snapshot_dir: str) -> None:
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

    def load_snapshot(self, snapshot_dir: str, device: torch.device) -> int:
        if not os.path.exists(os.path.join(snapshot_dir, self.name)):
            os.makedirs(os.path.join(snapshot_dir, self.name))
            print(f"No snapshots found for {self.name} in {snapshot_dir}")
            return 0
        all_snapshots = os.listdir(os.path.join(snapshot_dir, self.name))
        if len(all_snapshots) == 0:
            print(f"No snapshots found for {self.name} in {snapshot_dir}")
            return 0
        sorted_snapshots = sorted(all_snapshots, key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_snapshot = sorted_snapshots[-1]
        snapshot_path = os.path.join(snapshot_dir, self.name, latest_snapshot)
        snapshot = torch.load(snapshot_path, map_location=device)
        self.model.module.load_state_dict(snapshot['MODEL_STATE'])
        self.optimizer.load_state_dict(snapshot['OPTIMIZER'])
        self.scheduler.load_state_dict(snapshot['SCHEDULER'])
        epochs_trained = snapshot['EPOCHS_RUN']
        print(f"Loaded {self.name} model from epoch {epochs_trained}")
        return epochs_trained
    