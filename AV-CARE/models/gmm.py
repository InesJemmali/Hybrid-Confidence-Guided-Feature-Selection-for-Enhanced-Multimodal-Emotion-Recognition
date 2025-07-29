import os
import torch.nn.functional as F 

from sklearn.mixture import GaussianMixture
import pickle

import torch

# Define the correct GMM save/load path
GMM_SAVE_PATH_VIDEO = r"C:\Users\jemma\OneDrive\Documents\pfe\mmer - with modifications - best final results 4\savings\gmm\gmm_model_video"
GMM_SAVE_PATH_AUDIO = r"C:\Users\jemma\OneDrive\Documents\pfe\mmer - with modifications - best final results 4\savings\gmm\gmm_model_audio" 



def train_gmm_audio(features_list, num_components=4):
    """
    Train a GMM on per-frame audio temporal differences (deltas).

    Args:
        features_list: List of tensors with shape [B, T, C]
        num_components: Number of GMM components

    Returns:
        Trained GMM model
    """
    all_deltas = []

    for ft in features_list:
        # Compute frame-to-frame differences
        deltas = torch.abs(ft[:, 1:] - ft[:, :-1])  # Shape: [B, T-1, C]
        # Pad the first frame to keep original length
        deltas = F.pad(deltas, pad=(0, 0, 1, 0))     # Shape: [B, T, C]
        # Flatten across batch and time
        all_deltas.append(deltas.reshape(-1, ft.shape[-1]))  # Shape: [B*T, C]

    features = torch.cat(all_deltas, dim=0)  # Shape: [N, C]
    features_np = features.detach().cpu().numpy()

    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type='diag',
        random_state=42,
        max_iter=300
    )
    gmm.fit(features_np)

    return gmm

def train_gmm_video(features_list, num_components=4):
    """
    Train a GMM on per-frame visual temporal differences (after spatial pooling).

    Args:
        features_list: List[Tensor] with shape [B, T, C, H, W]
        num_components: Number of GMM components

    Returns:
        Trained GMM model
    """
    all_deltas = []

    for ft in features_list:
        B, T, C, H, W = ft.shape

        # Step 1: Spatial mean pooling → [B, T, C]
        spatial_mean = ft.view(B, T, C, -1).mean(dim=-1)

        # Step 2: Compute deltas over time → [B, T-1, C]
        deltas = torch.abs(spatial_mean[:, 1:] - spatial_mean[:, :-1])

        # Step 3: Pad first frame → [B, T, C]
        deltas = F.pad(deltas, (0, 0, 1, 0))

        # Step 4: Flatten across batch and time
        all_deltas.append(deltas.reshape(-1, C))  # Shape: [B*T, C]

    features = torch.cat(all_deltas, dim=0)
    features_np = features.detach().cpu().numpy()

    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type='diag',
        random_state=42,
        max_iter=300
    )
    gmm.fit(features_np)

    return gmm


def save_gmm_video(gmm, epoch, file_path=GMM_SAVE_PATH_VIDEO):
    """
    Save the trained GMM model to a file with the epoch number appended.

    Parameters:
        gmm (GaussianMixture): The trained GMM model.
        epoch (int): Current epoch number to include in filename.
        file_path (str): file file path without extension.
    """
    # Construct the full path with epoch number and .pth extension  
    full_path = f"{file_path}{epoch}.pkl"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Save the GMM model
    with open(full_path, "wb") as f:
        pickle.dump(gmm, f)

    print(f"GMM model saved to {full_path}")

def load_gmm_video(epoch,file_path=GMM_SAVE_PATH_VIDEO):
    """Load a pretrained GMM model if it exists, otherwise return None."""

    file_path = f"{file_path}{epoch}.pkl"
   
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            print(f"✅ Loaded GMM model from: {file_path}")
            return pickle.load(f)
    else:
        print("GMM model not found. Using default confidence values.")
        return None  # Return None if GMM is missing
    
def save_gmm_audio(gmm,epoch, file_path=GMM_SAVE_PATH_AUDIO):
    """Save the trained GMM model to the correct directory."""
    file_path = f"{file_path}{epoch}.pkl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with open(file_path, "wb") as f:
        pickle.dump(gmm, f)
    print(f"GMM model saved to {file_path}")

def load_gmm_audio(epoch,file_path=GMM_SAVE_PATH_AUDIO):
    """Load a pretrained GMM model if it exists, otherwise return None."""

    file_path = f"{file_path}{epoch}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            print(f"✅ Loaded GMM model from: {file_path}")
            return pickle.load(f)
    else:
        print("GMM model not found. Using default confidence values.")
        return None  # Return None if GMM is missing    
 
    
    