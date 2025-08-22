import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from typing import Literal, Tuple

class RepresentationSimilarityAnalysis:
    """
    A class to compute Representational Similarity Analysis (RSA) between feature sets.
    
    This involves creating Representational Dissimilarity Matrices (RDMs) from features
    and then correlating these RDMs.
    """

    def __init__(self, metric: Literal['cosine', 'euclidean'] = 'cosine'):
        """
        Initializes the RSA calculator.

        Args:
            metric (str, optional): The default dissimilarity metric. 
                                    Defaults to 'cosine'.
        """
        self.metric = metric

    def compute_rdm(self, features: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
        """
        Computes a Representational Dissimilarity Matrix (RDM) from a feature tensor.

        Args:
            features (torch.Tensor): A tensor of shape (num_samples, feature_dim).
            metric (str): The dissimilarity metric to use. Options: 'euclidean', 'cosine'.

        Returns:
            torch.Tensor: The RDM of shape (num_samples, num_samples).
        """
        num_samples = int(features.shape[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = features.to(device)
        
        # Pre-normalize features for cosine similarity
        if self.metric == 'cosine':
            features = F.normalize(features, p=2, dim=1)

        rdm = torch.zeros((num_samples, num_samples), device='cpu')
        
        # Iterate through chunks of features to compute pairwise distances
        print(f"Computing RDM with {self.metric} distance on device: {device} (chunk_size: {chunk_size})...")
        for i in tqdm(range(0, num_samples, chunk_size)):
            for j in range(0, num_samples, chunk_size):
                features_i = features[i:min(i + chunk_size, num_samples)]
                features_j = features[j:min(j + chunk_size, num_samples)]

                if self.metric == 'euclidean':
                    # torch.cdist computes pairwise distances between vectors in two batches
                    # output shape (len(features_i), len(features_j))
                    dissimilarities = torch.cdist(features_i, features_j, p=2)
                elif self.metric == 'cosine':
                    # F.cosine_similarity expects (N, C) and (N, C) but computes pairwise if given (N,C) and (M,C)
                    # We need to explicitly compute all pairs
                    # Normalize features first for robust cosine similarity
                    features_i_norm = F.normalize(features_i, p=2, dim=1)
                    features_j_norm = F.normalize(features_j, p=2, dim=1)
                    # The dot product of normalized vectors is cosine similarity
                    similarities = torch.mm(features_i_norm, features_j_norm.T)
                    dissimilarities = 1 - similarities # Convert similarity to dissimilarity
                else:
                    raise ValueError("Unsupported metric. Choose 'euclidean' or 'cosine'.")
                
                # Ensure dissimilarities are non-negative and finite
                dissimilarities[torch.isnan(dissimilarities)] = 0.0
                dissimilarities[torch.isinf(dissimilarities)] = 0.0

                rdm[i:min(i + chunk_size, num_samples), j:min(j + chunk_size, num_samples)] = dissimilarities.cpu()
        
        # Ensure diagonal is zero (dissimilarity of an item with itself)
        rdm.fill_diagonal_(0)
        rdm.clamp_(min=0.0) # Ensure no negative values from floating point errors
        
        return rdm


    def vectorize_rdm(self, rdm: torch.Tensor) -> np.ndarray:
        """
        Extracts the unique lower triangular elements (excluding diagonal) of an RDM.

        Args:
            rdm (torch.Tensor): The RDM matrix.

        Returns:
            np.ndarray: A 1D array of the unique elements.
        """
        # Get the indices of the lower triangle, excluding the diagonal
        lower_triangle_indices = torch.tril_indices(row=rdm.shape[0], col=rdm.shape[1], offset=-1)
        return rdm[lower_triangle_indices[0], lower_triangle_indices[1]].numpy()


    def compute_rsa(self, 
                    rdm1: torch.Tensor, 
                    rdm2: torch.Tensor, 
                    correlation_type: Literal['pearson', 'spearman'] = 'pearson'
                   ) -> Tuple[float, float]:
        """
        Computes the RSA score (correlation between two RDMs).

        Args:
            rdm1 (torch.Tensor): The first RDM.
            rdm2 (torch.Tensor): The second RDM.
            correlation_type (str): The type of correlation to use ('pearson' or 'spearman').

        Returns:
            Tuple[float, float]: A tuple containing the correlation coefficient and the p-value.
        """
        if rdm1.shape != rdm2.shape:
            raise ValueError("RDMs must have the same shape.")
        
        vec_rdm1 = self.vectorize_rdm(rdm1)
        vec_rdm2 = self.vectorize_rdm(rdm2)

        if correlation_type == 'pearson':
            correlation, p_value = pearsonr(vec_rdm1, vec_rdm2)
        elif correlation_type == 'spearman':
            correlation, p_value = spearmanr(vec_rdm1, vec_rdm2)
        else:
            raise ValueError("Unsupported correlation_type. Choose 'pearson' or 'spearman'.")
        
        return correlation, p_value

class CenteredKernelAlignment:
    def __init__(self, kernel: str = 'linear'):
        self.kernel = kernel

    def center_gram_matrix(self, K: torch.Tensor) -> torch.Tensor:
        """
        Centers a Gram matrix K.
        K_c = H K H, where H = I - 1/N * 1_N 1_N^T
        """
        N = K.shape[0]
        # Using matrix multiplication for centering
        # H = torch.eye(N, device=K.device) - (1/N) * torch.ones((N, N), device=K.device)
        # K_c = H @ K @ H

        # More numerically stable and common way to center K:
        mean_rows = torch.mean(K, dim=1, keepdim=True)
        mean_cols = torch.mean(K, dim=0, keepdim=True)
        mean_all = torch.mean(K)
        K_c = K - mean_rows - mean_cols + mean_all
        return K_c


    def cka_linear_kernel(self, X: torch.Tensor, Y: torch.Tensor, device: torch.device) -> float:
        """
        Computes Centered Kernel Alignment (CKA) with a linear kernel.

        Args:
            X (torch.Tensor): Feature tensor 1 of shape (num_samples, feature_dim_X).
            Y (torch.Tensor): Feature tensor 2 of shape (num_samples, feature_dim_Y).
            device (torch.device): Device to perform computations ('cpu' or 'cuda').

        Returns:
            float: The CKA similarity score.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must be the same for CKA.")

        X = X.to(device)
        Y = Y.to(device)

        # 1. Compute Gram Matrices (linear kernel: X @ X.T)      
        K = torch.matmul(X, X.T)
        L = torch.matmul(Y, Y.T)

        # 2. Center Gram Matrices
        K_c = self.center_gram_matrix(K)
        L_c = self.center_gram_matrix(L)

        # 3. Compute HSIC numerator and denominators
        # trace(K_c L_c)
        numerator = torch.trace(torch.matmul(K_c, L_c))

        # trace(K_c K_c) and trace(L_c L_c)
        denom_K = torch.trace(torch.matmul(K_c, K_c))
        denom_L = torch.trace(torch.matmul(L_c, L_c))

        # 4. CKA Score
        cka_score = numerator / torch.sqrt(denom_K * denom_L)

        return cka_score.item()