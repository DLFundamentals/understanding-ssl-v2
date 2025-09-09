import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.gather import GatherLayer

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device="cuda"):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device = self.device)
        mask = mask.fill_diagonal_(0) # self-similarity is not useful
        # register the mask as a buffer
        self.register_buffer("mask", mask)
        return mask

    def forward(self, z_i, z_j, labels):
        # distributed version
        if dist.is_initialized():
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        
        z = torch.cat([z_i, z_j], dim=0)
        
        N = z.size(0)
        self.batch_size = N // 2
        # Mask correlated samples (positive and self pairs)
        self.mask = self.mask_correlated_samples(self.batch_size)

        # Compute the NxN similarity matrix (cosine similarity)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Extract the positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # Concatenate positive samples (for each view i and j)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # Extract negative samples by masking out the positive pairs
        negative_samples = sim[self.mask].reshape(N, -1)  # Masking positive pairs

        # Labels for cross-entropy loss (all zeros for correct classification)
        labels = torch.zeros(N).to(positive_samples.device).long()

        # Concatenate positive and negative samples for logits
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N x (N-1)

        # Calculate the loss using cross-entropy
        loss = self.criterion(logits, labels)
        loss /= N  # Normalize by batch size

        return loss
    
class DecoupledNTXentLoss(nn.Module):
    """
    Implementation of DCL
    """
    def __init__(self, temperature=0.5, device="cuda"):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device = self.device)
        mask = mask.fill_diagonal_(0) # self-similarity is not useful
        for i in range(batch_size):
            mask[i, batch_size + i] = 0 # mask the positive pair
            mask[batch_size + i, i] = 0

        return mask

    def forward(self, z_i, z_j, 
                labels=None): # labels are not used in this loss
        # distributed version
        if dist.is_initialized():
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        
        z = torch.cat([z_i, z_j], dim=0)
        
        N = z.size(0)
        self.batch_size = N // 2
        # Mask correlated samples (positive and self pairs)
        self.mask = self.mask_correlated_samples(self.batch_size)

        # Compute the NxN similarity matrix (cosine similarity)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Extract the positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # Concatenate positive samples (for each view i and j)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # Extract negative samples by masking out the positive pairs
        negative_samples = sim[self.mask].reshape(N, -1)  # Masking positive pairs

        # Labels for cross-entropy loss (all zeros for correct classification)
        labels = torch.zeros(N).to(positive_samples.device).long()

        # Concatenate positive and negative samples for logits
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N x (N-1)

        # numerical stability
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]

        # Calculate the loss using cross-entropy
        loss = self.criterion(logits, labels)
        loss /= N  # Normalize by batch size

        return loss
    
class NegSupConLoss(nn.Module):
    """
    Implementation of the negatives-only supervised contrastive loss function
    proposed in our work. 
    """
    def __init__(self, temperature=0.5, device="cuda"):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, labels):
        N = 2 * batch_size
        labels = labels.contiguous().view(-1, 1)
        # create a mask where negative pairs from the same class are masked 1s
        mask = torch.ne(labels, labels.T).to(self.device)
        return mask

    def forward(self, z_i, z_j, labels):
        
        # distributed version
        if dist.is_initialized():
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0)
        
        z = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        
        N = z.size(0)
        self.batch_size = N // 2
        # Mask correlated samples (positive and self pairs)
        self.mask = self.mask_correlated_samples(self.batch_size, labels)

        # Compute the NxN similarity matrix (cosine similarity)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Extract the positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # Concatenate positive samples (for each view i and j)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # Extract negative samples
        negative_samples = sim.masked_fill(~self.mask, float("-inf"))

        # Labels for cross-entropy loss (all zeros for correct classification)
        labels_hack = torch.zeros(N).to(positive_samples.device).long()

        # Concatenate positive and negative samples for logits
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N x (N-1)

        # numerical stability
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]

        # Calculate the loss using cross-entropy
        loss = self.criterion(logits, labels_hack)
        loss /= N  # Normalize by batch size

        return loss
    
class SupConLoss(nn.Module):
    """
    Implementation of the supervised contrastive loss function (Khosla et al. 2020).
    """
    def __init__(self, temperature=0.5, device="cuda"):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, labels):
        """
        Creates a mask to identify all positive pairs (same label, excluding self).
        """
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).to(self.device)
        mask.fill_diagonal_(0)
        return mask  # [2N, 2N] boolean mask

    def forward(self, z_i, z_j, labels):
        # Distributed version
        if dist.is_initialized():
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0)

        # Shape: [2N, d], labels: [2N]
        z = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        N = z.size(0)

        # Similarity matrix
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Masks
        pos_mask = self.mask_correlated_samples(labels)          # positives
        denominator_mask = ~torch.eye(N, dtype=torch.bool, device=self.device)

        # Numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Log-softmax
        exp_logits = torch.exp(logits) * denominator_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Mean log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-9)

        # Final loss
        loss = -mean_log_prob_pos.mean()
        return loss

class HybridSupConLoss(nn.Module):
    def __init__(self, temperature=0.5, device="cuda"):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
    
    def mask_correlated_samples(self, labels):
        """
        Creates a mask to identify all positive pairs (same label, excluding self).
        """
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).to(self.device)
        mask.fill_diagonal_(0)
        return mask  # [2N, 2N] boolean mask

    def forward(self, z_i, z_j, labels):
        # Distributed version
        if dist.is_initialized():
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0)

        # Shape: [2N, d], labels: [2N]
        z = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        N = z.size(0)

        # Similarity matrix
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Masks
        pos_mask = self.mask_correlated_samples(labels)          # positives
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(0)

        # Numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Log-softmax
        exp_logits = torch.exp(logits) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Mean log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-9)

        # Final loss is the negative mean of the mean log-probabilities
        loss = -mean_log_prob_pos
        loss = loss[~torch.isnan(loss) & ~torch.isinf(loss)].mean()
        return loss
    
class MultiViewCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=None,device="cuda"):
        super().__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, logits_i, logits_j, labels):
        # Distributed version
        if dist.is_initialized():
            logits_i = torch.cat(GatherLayer.apply(logits_i), dim=0)
            logits_j = torch.cat(GatherLayer.apply(logits_j), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0)

        # Shape: [2N, d], labels: [2N]
        logits = torch.cat([logits_i, logits_j], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        N = logits.size(0)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class LossFactory:
    @staticmethod
    def get_loss(loss_name, **kwargs):
        if loss_name == "cl":
            return NTXentLoss(**kwargs)
        elif loss_name == "dcl":
            return DecoupledNTXentLoss(**kwargs)
        elif loss_name == "nscl":
            return NegSupConLoss(**kwargs)
        elif loss_name == "scl":
            return SupConLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
