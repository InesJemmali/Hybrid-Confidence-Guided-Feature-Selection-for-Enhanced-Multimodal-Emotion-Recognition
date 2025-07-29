import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture





class VarianceConfidenceNN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, feature_dim)
        self.activation = nn.ReLU()

    def forward(self, features):  # features: [B, T, C]
        B, T, C = features.shape
        x = features.view(B * T, C)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # [B*T, C]
        return x.view(B, T, C)  # back to [B, T, C]

    


def get_gmm_confidence(features, gmm):
    """
    Compute GMM-based confidence scores for features.
    
    Args:
        features: Tensor of shape [B, T, C]
        gmm: Trained Gaussian Mixture Model

    Returns:
        confidence_scores: Tensor of shape [B, T, C]
    """
    B, T, C = features.shape
    features_flat = features.view(B * T, C)
    #Debug
    #print("GMM input shape:", features_flat.shape)
    #print("GMM expects:", gmm.means_.shape[1])

    log_probs = gmm.score_samples(features_flat.cpu().detach().numpy())  # [B*T]
    confidence_scores = torch.tensor(log_probs, dtype=torch.float32).to(features.device)
    confidence_scores = confidence_scores.unsqueeze(1).expand(-1, C).view(B, T, C)

    # Normalize to [0, 1]
    min_val, max_val = confidence_scores.min(), confidence_scores.max()
    if max_val - min_val > 1e-6:
        confidence_scores = (confidence_scores - min_val) / (max_val - min_val)

    return confidence_scores








class ConfidenceFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # Learnable weight fusion

    def forward(self, variance_conf, gmm_conf):
        """
        Args:
            variance_conf (torch.Tensor): FFN-based confidence scores, shape [B,T, C]
            gmm_conf (torch.Tensor): GMM-based confidence scores, shape [B,T, C]
        Returns:
            fused_confidence_scores (torch.Tensor): Combined confidence scores, shape [B,T, C]
        """
        x = torch.stack([variance_conf, gmm_conf], dim=-1)  # [B, T, C, 2]
        weight = torch.sigmoid(self.fc(x)).squeeze(-1)  # [B, T, C]
        return weight * variance_conf + (1 - weight) * gmm_conf

def compute_audio_confidence_scores(x, variance_ffn, gmm, fusion_model):
    """
    Compute dynamic confidence scores per frame and apply weights.
    
    Args:
        x (Tensor): [B, T, C] features
        variance_ffn (nn.Module): VarianceConfidenceNN
        gmm (GaussianMixture): trained GMM
        fusion_model (nn.Module): ConfidenceFusion
    
    Returns:
        confidence_scores: [B, T, C]
        weighted_x: [B, T, C]
    """
    B, T, C = x.shape
    device = x.device

    # Step 1: Compute per-frame FFN confidence
    diff = torch.abs(x[:, 1:] - x[:, :-1])                 # [B, T-1, C]
    padded_diff = F.pad(diff, pad=(0, 0, 1, 0))            # [B, T, C]
    variance_conf = variance_ffn(padded_diff)              # pass to FFN

    # Step 2: GMM confidence
    if gmm is not None:
        frame_diff = torch.abs(x[:, 1:] - x[:, :-1])  # [B, T-1, C]
        padded_diff = F.pad(frame_diff, pad=(0, 0, 1, 0))  # [B, T, C]
        gmm_conf = get_gmm_confidence(padded_diff, gmm)
        
    else:
        gmm_conf = torch.ones_like(variance_conf, device=device)

    # Step 3: Fuse & normalize
    confidence_scores = fusion_model(variance_conf, gmm_conf)
    #normalized_conf = confidence_scores / (confidence_scores.sum(dim=-1, keepdim=True) + 1e-8)
    normalized_conf = confidence_scores
    # Step 4: Apply weights
    weighted_x = x * normalized_conf  # [B, T, C]

    return confidence_scores, weighted_x









def compute_visual_confidence_scores(faces, variance_ffn, gmm, fusion_model):
    """
    Confidence-aware visual weighting with per-frame scores.
    
    Args:
        faces (Tensor): [B, T, C, H, W]
        ...
    Returns:
        fused_conf (Tensor): [B, T, C]
        weighted_features (Tensor): [B, T, C*H*W]
    """
    B, T, C, H, W = faces.shape
    device = faces.device

    # [B, T, C] from spatial mean
    spatial_mean = faces.view(B, T, C, -1).mean(dim=-1)

    # [B, T-1, C] → temporal difference
    frame_diff = torch.abs(spatial_mean[:, 1:] - spatial_mean[:, :-1])
    frame_diff = F.pad(frame_diff, (0, 0, 1, 0))  # Pad to [B, T, C]

    # FFN and GMM
    variance_conf = variance_ffn(frame_diff)
    gmm_conf = get_gmm_confidence(frame_diff, gmm) if gmm is not None else torch.ones_like(variance_conf, device=device)
    fused_conf = fusion_model(variance_conf, gmm_conf)
    #norm_conf = fused_conf / (fused_conf.sum(dim=-1, keepdim=True) + 1e-8)
    norm_conf = fused_conf
    # Apply weights to original features
    weighted_faces = faces * norm_conf.unsqueeze(-1).unsqueeze(-1)  # [B, T, C, H, W]
    weighted_features = weighted_faces.view(B, T, -1)  # Flatten to [B, T, C*H*W]

    return fused_conf, weighted_features


