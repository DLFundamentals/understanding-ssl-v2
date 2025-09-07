import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.encoder import BaseEncoder, ResNetEncoder, ViTEncoder
from models.projector import SimCLR_Projector

import os
from tqdm import tqdm
from collections import defaultdict
import wandb

class SimCLR(nn.Module):
    def __init__(self, model, layer = -2, dataset = 'imagenet',
                 width_multiplier = 1, pretrained = False,
                 hidden_dim = 512, projection_dim = 128, **kwargs):
        
        super().__init__()

        if isinstance(model, models.ResNet):
            self.encoder = ResNetEncoder(model, layer = layer, dataset = dataset,
                                        width_multiplier = width_multiplier,
                                        pretrained = pretrained)
        elif isinstance(model, models.VisionTransformer):
            self.encoder = ViTEncoder(model, layer ='encoder',
                                    image_size=kwargs.get('image_size', 224),
                                    patch_size=kwargs.get('patch_size', 16),
                                    stride=kwargs.get('stride', 16),
                                    hidden_dim=kwargs.get('token_hidden_dim', 768),
                                    mlp_dim=kwargs.get('mlp_dim', 3072))
            
        else:
            raise NotImplementedError(f"Model {model} not implemented. Use a ResNet or ViT model.")
        
        # run a mock image tensor to instantiate parameters
        with torch.no_grad():
            if 'imagenet' in dataset:
                h = self.encoder(torch.randn(1, 3, 224, 224))
            elif 'cifar' in dataset:
                h = self.encoder(torch.randn(1, 3, 32, 32))
            elif dataset == 'svhn':
                h = self.encoder(torch.randn(1, 3, 32, 32))
            else:
                raise NotImplementedError(f"{dataset} not implemented")
            
        input_dim = h.shape[1]
        
        self.projector = SimCLR_Projector(input_dim, hidden_dim, projection_dim)

    def forward(self, X):
  
        h = self.encoder(X)
        h = h.view(h.size(0), -1) # flatten the tensor
        g_h = self.projector(h)

        return h, F.normalize(g_h, dim = -1)

    def run_one_batch(self, batch, criterion, device='cuda'):
        # get the inputs
        view1, view2, labels = batch
        # skip the batch with only 1 image
        if view1.size(0) < 2:
            return 0

        view1, view2 = view1.to(device), view2.to(device)
        labels = labels.to(device)

        # forward pass
        view1_features, view1_proj = self(view1)
        view2_features, view2_proj = self(view2)

        # compute contrastive loss
        loss = criterion(view1_proj, view2_proj, labels)

        return loss

class SimCLRWithClassificationHead(nn.Module):
    """
    A supervised model that wraps a SimCLR-style architecture (encoder + projector).
    - During training, it passes the projector's output through a classifier.
    - During evaluation, it returns the encoder and projector outputs directly.
    """
    def __init__(self, simclr_model: SimCLR, num_classes: int):
        super().__init__()
        self.simclr_model = simclr_model
        
        # Get the projector's output dimension for the classifier input
        projector_output_dim = self.simclr_model.projector.projection_dim
        self.classifier = nn.Linear(projector_output_dim, num_classes)

    def forward(self, x, mode='eval'):
        # Always get the base representations from the SimCLR architecture
        h, g_h = self.simclr_model(x)
        
        # Mode-dependent output
        if mode == 'train':
            # During training, return logits for the loss function
            logits = self.classifier(g_h)
            return logits
        else:
            # During evaluation, return representations for downstream tasks
            return h, g_h

    def run_one_batch(self, batch, criterion, mode='eval', device='cuda'):
        # get the inputs
        view1, view2, labels = batch
        # skip the batch with only 1 image
        if view1.size(0) < 2:
            return 0
        view1, view2 = view1.to(device), view2.to(device)
        labels = labels.to(device)

        logits1 = self(view1, mode='train')
        logits2 = self(view2, mode='train')
        loss = criterion(logits1, logits2, labels)

        return loss