import torch
import torch.nn as nn
import torch.nn.functional as F




    
class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss with confidence-aware correlation loss.

    Parameters:
    - alpha (Tensor): Class weighting factor.
    - gamma (float): Focusing parameter for hard example mining.
    - beta (float): Weight for correlation loss component.
    - num_classes (int): Number of classes (needed for confidence score projection).
    """
    
    def __init__(self, alpha=None, gamma=2, beta=0.2, num_classes=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.beta = beta
        self.num_classes = num_classes

      

    def correlation_loss(self, confidence_scores, uncertainty):
        """
        Computes correlation loss between confidence scores and uncertainty.

        Args:
            confidence_scores (torch.Tensor): Projected confidence scores `[B, num_classes]`.
            uncertainty (torch.Tensor): Model uncertainty estimate `[B, num_classes]`.

        Returns:
            torch.Tensor: Mean correlation loss across batch.
        """
        if uncertainty.dim() == 1:
            uncertainty = uncertainty.unsqueeze(1)

        # Fix NaN Handling
        confidence_scores = torch.where(torch.isnan(confidence_scores), torch.zeros_like(confidence_scores), confidence_scores)
        uncertainty = torch.where(torch.isnan(uncertainty), torch.zeros_like(uncertainty), uncertainty)

        # *Normalize Confidence Scores & Uncertainty
        mean_conf = confidence_scores.mean()
        std_conf = confidence_scores.std().clamp(min=1e-3)  # Avoid division by zero
        mean_uncert = uncertainty.mean()
        std_uncert = uncertainty.std().clamp(min=1e-3)  

        confidence_scores = (confidence_scores - mean_conf) / std_conf
        uncertainty = (uncertainty - mean_uncert) / std_uncert

        # Compute Correlation
        correlation = torch.mean(confidence_scores * uncertainty, dim=1) 
        correlation = torch.clamp(correlation, -1, 1)  # Keep within valid range

        return -torch.mean(correlation)  # Minimize correlation
    
    def error_alignment_loss(self, confidence_scores, inputs, targets):
        """
        Penalize positive correlation between confidence and prediction error.
        Encourages: high confidence ↔ correct prediction.
        """
        sample_conf = confidence_scores.mean(dim=1)  # [B]
        preds = torch.argmax(inputs, dim=1)
        true = torch.argmax(targets, dim=1)
        error = (preds != true).float()  # [B], 1 if wrong

        # Normalize
        sample_conf = (sample_conf - sample_conf.mean()) / (sample_conf.std().clamp(min=1e-3))
        error = (error - error.mean()) / (error.std().clamp(min=1e-3))

        corr = torch.mean(sample_conf * error)
        return corr  # minimize this (penalize if confidence aligns with being wrong)

    def forward(self, inputs, targets, class_visual_confidence_scores=None, class_audio_confidence_scores=None):
        """
        Computes the modified Focal Loss with correlation loss.

        Args:
                inputs (torch.Tensor): Model predictions `[B, num_classes]`.
                targets (torch.Tensor): Ground truth labels `[B, num_classes]` (one-hot encoded).
                class_visual_confidence_scores (torch.Tensor): Visual confidence `[B, num_classes]`.
                class_audio_confidence_scores (torch.Tensor): Audio confidence `[B, num_classes]`.

        Returns:
                torch.Tensor: Combined loss (Focal Loss + correlation loss).
        """
        # Convert targets to indices
        targets_long = torch.argmax(targets, dim=1)    # [B]

        # Compute standard focal loss
        ce_loss = F.cross_entropy(inputs, targets_long, reduction='none')    # [B]
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha[targets_long] * (1 - pt) ** self.gamma * ce_loss).mean()

        # Compute entropy-based uncertainty (per class)
        probabilities = F.softmax(inputs, dim=-1).clamp(min=1e-6, max=1.0)    # [B, num_classes]
        uncertainty = -probabilities * torch.log(probabilities + 1e-6)    # [B, num_classes]

        # Fix NaNs if any
        uncertainty = torch.where(torch.isnan(uncertainty), torch.zeros_like(uncertainty), uncertainty)

        # Compute correlation loss
        corr_loss = 0.0
        error_corr_loss = 0.0
        modality_count = 0

        if class_audio_confidence_scores is not None:
                 
                # Uncertainty-based correlation 
                corr_loss_audio = self.correlation_loss(class_audio_confidence_scores, uncertainty)
                corr_loss += corr_loss_audio

                # Prediction-error correlation
                error_corr_loss += self.error_alignment_loss(class_audio_confidence_scores, inputs, targets)
                modality_count += 1

        if class_visual_confidence_scores is not None:
                # Uncertainty-based correlation
                corr_loss_visual = self.correlation_loss(class_visual_confidence_scores, uncertainty)
                corr_loss += corr_loss_visual

                # Prediction-error correlation
                error_corr_loss += self.error_alignment_loss(class_visual_confidence_scores, inputs, targets)
                modality_count += 1

        if modality_count > 0:
                corr_loss = corr_loss / modality_count    # average correlation loss across modalities
                error_corr_loss /= modality_count

        # Clamp for numerical stability
        corr_loss = torch.abs(corr_loss)
        corr_loss = torch.clamp(corr_loss, min=-4.0, max=4.0)
        error_corr_loss = torch.clamp(error_corr_loss, min=-4.0, max=4.0)

        # Final total loss
        total_loss = focal_loss + self.beta * corr_loss + self.beta * torch.abs(error_corr_loss)

        # Safety check
        if torch.isnan(total_loss).any():
                print("NaN detected in total loss!")
                print(f"Inputs: {inputs}")
                print(f"Targets: {targets}")
                print(f"Uncertainty: {uncertainty}")
                print(f"Conf Audio: {class_audio_confidence_scores}")
                print(f"Conf Visual: {class_visual_confidence_scores}")
                exit(1)

        return total_loss 
    

class ConfidenceFocalLossBCE(nn.Module):
    def __init__(self, alpha=None, gamma=2, beta=0.2):
        """
        Focal Loss with Binary Cross-Entropy and confidence-error alignment.

        Args:
            alpha (Tensor): Class-wise alpha weights for BCE. Should be broadcastable to target shape.
            gamma (float): Focusing parameter for hard samples
            beta (float): Weight for the confidence alignment regularization
        """
        super(ConfidenceFocalLossBCE, self).__init__()
        self.alpha = alpha.cuda() if alpha is not None else None
        self.gamma = gamma
        self.beta = beta

    def confidence_alignment(self, confidence, inputs, targets):
        """
        Penalize confident but incorrect predictions.
        Args:
            confidence: Tensor [B, T, C] (e.g., frame-feature confidence)
            inputs: Tensor [B, num_classes]
            targets: Tensor [B, num_classes]
        """
        B, T, C = confidence.shape

        pred_classes = (inputs >= 0.5).float()
        errors = (pred_classes != targets).float().mean(dim=1, keepdim=True)  # [B, 1]
        error_mask = errors.unsqueeze(-1).expand(B, T, C)  # [B, T, C]

        # Normalize confidence and error
        conf_norm = (confidence - confidence.mean()) / (confidence.std() + 1e-6)
        error_norm = (error_mask - error_mask.mean()) / (error_mask.std() + 1e-6)

        corr = torch.mean(conf_norm * error_norm)  # High corr = bad → encourage anti-correlation
        return -corr

    def forward(self, inputs, targets, audio_conf=None, video_conf=None):
        """
        Compute Focal BCE Loss with optional confidence supervision.

        Args:
            inputs: Raw logits before sigmoid, shape [B, num_classes]
            targets: Binary labels, shape [B, num_classes]
            audio_conf, video_conf: confidence maps, shape [B, T, C]
        """
        inputs = torch.sigmoid(inputs)

        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Focal Modulation
        pt = torch.exp(-bce_loss)
        if self.alpha is not None:
            focal_weight = ((1 - pt) ** self.gamma) * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        else:
            focal_weight = ((1 - pt) ** self.gamma)

        focal_bce_loss = (focal_weight * bce_loss).mean()

        # Confidence alignment
        total_corr_loss = 0.0
        modality_count = 0

        if audio_conf is not None:
            total_corr_loss += self.confidence_alignment(audio_conf, inputs, targets)
            modality_count += 1

        if video_conf is not None:
            total_corr_loss += self.confidence_alignment(video_conf, inputs, targets)
            modality_count += 1

        if modality_count > 0:
            total_corr_loss /= modality_count
            total_loss = focal_bce_loss + self.beta * torch.abs(total_corr_loss.abs)
        else:
            total_loss = focal_bce_loss

        return total_loss    
  
    
class FocalLossWithoutConfidence(nn.Module):
    
    """
    Implementation of the Focal Loss as a PyTorch module.

    Parameters:
    - alpha (Tensor): Weighting factor for the positive class.
    - gamma (float): Focusing parameter to adjust the rate at which easy examples contribute to the loss.
    
    """
    
    def __init__(self, alpha=None, gamma=2):
        super(FocalLossWithoutConfidence, self).__init__()
        self.alpha = alpha.to("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma

        
    def forward(self, inputs, targets):
       
            
        targets_long = targets.clone().long()
        targets_long  = torch.argmax(targets_long , dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets_long, reduction='mean')
        pt = torch.exp(-ce_loss)
       

        loss = (self.alpha[targets_long] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss    

class FocalLossBCE(nn.Module):
    """
    Implementation of the Focal Loss with Binary Cross-Entropy (BCE) as a PyTorch module.

    Parameters:
    - alpha (Tensor): Weighting factor for the positive class.
    - gamma (float): Focusing parameter to adjust the rate at which easy examples contribute to the loss.
    - reduction (str): Specifies the reduction type: 'none', 'mean', or 'sum'.
    """

    def __init__(self, alpha=None, gamma=2):
        super(FocalLossBCE, self).__init__()
        self.alpha = alpha.cuda() if alpha is not None else None
        self.gamma = gamma
       
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid for binary classification
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute the focal weight
        pt = torch.exp(-bce_loss)
        focal_weight = ((1 - pt) ** self.gamma) * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        
        loss = (focal_weight * bce_loss).mean()
        
        
        return loss     