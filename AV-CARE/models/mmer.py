from facenet_pytorch import MTCNN
import torch
import cv2

from torch import nn
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from models.confidences import ConfidenceFusion,  VarianceConfidenceNN, compute_visual_confidence_scores
from models.gmm import load_gmm_video
from models.text_model import ALBERT
from models.video_model import RepVGGBlock
from models.audio_model import CNN18biXLSTM, CNN_BiGRU
from models.transformers_module import WrappedTransformerEncoder, WrappedTransformerEncoderVideo
from models.attention_module import CrossModalAttention, CrossModalAttentionBlock, SpatialAttentionBlock, TemporalAttentionBlock


    



class MODEL(nn.Module):
    """
    Multi-Modal Emotion Recognition Model.

    Args:
        args (Dict[str, Any]): Dictionary containing model configurations and hyperparameters.
            - 'num_emotions' (int): Number of emotion classes for classification.
        device (torch.device): The device (CPU or GPU) on which computations will be performed.
        text_model_size (str, optional): Size variant of the ALBERT model to use ('base', 'large', 'xlarge').
            Defaults to 'base'.

    Attributes:
        num_classes (int): Number of emotion classes.
        args (Dict[str, Any]): Model configuration parameters.
        mod (List[str]): List of modalities to be used ('t' for text, 'v' for visual, 'a' for audio).
        device (torch.device): Device for computations.
        feature_dim (int): Dimensionality of feature representations.
        T (ALBERT): Text encoder using the ALBERT model.
        mtcnn (MTCNN): Face detection model for preprocessing visual inputs.
        normalize (transforms.Normalize): Normalization layer for visual features.
        V (nn.Sequential): Visual encoder composed of convolutional and RepVGG blocks.
        A (nn.Sequential): Audio encoder using a CNN-LSTM hybrid architecture.
        v_flatten (nn.Sequential): Flattening and projection layers for visual features.
        a_flatten (nn.Sequential): Flattening and projection layers for audio features.
        v_transformer (WrappedTransformerEncoder): Transformer encoder for visual features.
        v_cross_modal_attention (CrossModalAttentionT): Cross-modal attention layer for text attending to visual features.
        a_cross_modal_attention (CrossModalAttentionT): Cross-modal attention layer for text attending to audio features.
        v_out (nn.Linear): Output layer for visual modality.
        t_out (nn.Linear): Output layer for text modality.
        a_out (nn.Linear): Output layer for audio modality.
        weighted_fusion (nn.Linear): Weighted fusion layer to aggregate logits from all modalities.
    """
    def __init__(self, args, device, text_model_size='base'):
        super(MODEL, self).__init__()
        self.num_classes = args['num_emotions']
        self.args = args

        self.mod = args['modalities'] #['t', 'v', 'a'] :  Modalities: text, visual, audio
        self.device = device
        self.feature_dim = 256
         
        # Transformer configuration
        nlayers, nheads, trans_dim, audio_dim = 4, 4, 512,72
        text_cls_dim = 1024 if text_model_size == 'large' else 2048 if text_model_size == 'xlarge' else 768  # Dimension for text classification

        # Text encoder
        if 't' in self.mod:
          self.T = ALBERT(feature_dim=self.feature_dim, size=text_model_size)


        if 'v' in self.mod:
           # Face detection using MTCNN
           self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)

           # Normalization for face images
           self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

           # Visual encoder
    
           self.V = nn.Sequential(
                nn.Conv2d(3, 32, 5, padding=2),  # Reduced from 64 to 32
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                RepVGGBlock(32, 32), 
                RepVGGBlock(32, 32), # Keep output channels the same
                RepVGGBlock(32, 64),  # Reduced from 128 to 64
                nn.MaxPool2d(2, stride=2),
                RepVGGBlock(64, 128),  # Reduced from 256 to 128
                nn.MaxPool2d(2, stride=2),
      
             )
           
           # Transformers for visual and audio features
           self.v_transformer = WrappedTransformerEncoderVideo(
                   dim=trans_dim,
                  num_layers=nlayers,
                  num_heads=nheads
             )

           self.visual_feature_proj = nn.Sequential(
               nn.Linear(4608, 2048),  # Step 1: Reduce from 3456 → 2048
               nn.ReLU(),
               nn.Linear(2048, 1024),  # Step 2: Reduce from 2048 → 1024
               nn.ReLU(),
               nn.Linear(1024, 512),  # Step 3: Reduce from 1024 → 512
               nn.ReLU()
             )
           
           # The SAB BLocks for video modality
           self.sab=SpatialAttentionBlock(embed_dim=128,num_heads=2,height=6, width=6)
    

           # FFN to compute per-frame channel-wise variance-based confidence
           self.confidence_net_visual_ffn = VarianceConfidenceNN(feature_dim=128)  
           # Trainable confidence fusion module
           self.confidence_fusion_visual = ConfidenceFusion()
           # Gaussian Mixture Model trained on visual frame features (trained externally)
           self.gmm_visual = None


        if 'a' in self.mod:
            # Audio encoder configuration
            config = {
                "n_input": 1,
                "hidden_dim": 16,
                "n_layers": 18,
                "n_output": 4,
                "lr": self.args['learning_rate'],
               
            }

            # Audio encoder
            self.A = CNN_BiGRU(
                        n_input=config["n_input"],
                        hidden_dim=config["hidden_dim"],
                        n_layers=config["n_layers"],
                        n_output=config["n_output"],
                   
                )
        
        
        if 'v' in self.mod and 'a' in self.mod and 't' in self.mod:
            self.av_attn = CrossModalAttentionBlock(
               query_dim=512 ,                 
               key_value_dim=72,   
               embed_dim=256,
               num_heads=4
            )
            self.va_attn = CrossModalAttentionBlock(
               query_dim=72 ,                 
               key_value_dim=512,   
               embed_dim=256,
               num_heads=4
            )
            self.tv_attn = CrossModalAttentionBlock(
               query_dim=text_cls_dim,               
               key_value_dim=256,   
               embed_dim=256,
               num_heads=4
            )
            self.ta_attn = CrossModalAttentionBlock(
               query_dim=text_cls_dim,                 
               key_value_dim=256,   
               embed_dim=256,
               num_heads=4
            )
            
  
            # Output layers for text, visual, and audio
            self.v_out = nn.Linear(256, self.num_classes)
            self.ta_out = nn.Linear(256, self.num_classes)
            self.tv_out = nn.Linear(256, self.num_classes)
            self.a_out = nn.Linear(256, self.num_classes)

            # Weighted fusion layer
            self.weighted_fusion = nn.Linear(4, 1, bias=False)

    def crop_img_center(self, img, target_size=48):

        """
        Crops the center of the image to the target size.

        Args:
            img (torch.Tensor): Image tensor with shape (C, H, W).
            target_size (int, optional): Desired size for height and width. Defaults to 48.

        Returns:
            torch.Tensor: Cropped image tensor.
        """
        current_size = img.size(1)
        off = (current_size - target_size) // 2
        return img[:, off:off + target_size, off:off + target_size]
    
    

    
    def reload_gmm_video(self,epoch):
        """
        Reload the GMM model from disk.
        This should be called after each epoch to ensure we use the latest GMM.
        """
        self.gmm_visual = load_gmm_video(epoch)    
        

    def forward(self, imgs, imgs_lens, waves,  text,return_features=True):
        """
        Processes input data from text, visual, and audio modalities, applies respective models, 
        and fuses their outputs using cross-modal attention for classification.

        Args:
            imgs (List[torch.Tensor]): List of input images (faces) for the visual modality.
            imgs_lens (List[int]): Sequence lengths for the images (batch-wise).
            waves (torch.Tensor): Raw waveforms for the audio modality.
            text (Dict[str, torch.Tensor]): Tokenized input text for the text modality.

        Returns:
            torch.Tensor: Fused logits from different modalities for classification.

        Notes:
            Returns a single modality's logits if only one modality is specified.
        """
        
        all_logits = []
        imgs_lens2 = []

        # Process text modality
        if 't' in self.mod:
            text_features = self.T(text)
            #all_logits.append(self.t_out(text_features))

        # Process visual modality
        if 'v' in self.mod:
                faces_per_sample = [[] for _ in range(len(imgs_lens))]
                k = [0] * len(imgs_lens)

                for i, img in enumerate(imgs):
                                detected_face = None
                                if img is not None and img.any():
                                                detected_face = self.mtcnn(img)
                                else:
                                                detected_face = self.crop_img_center(torch.tensor(img).permute(2, 0, 1))

                                if detected_face is not None:
                                                detected_face = self.normalize(detected_face)
                                                for j in range(len(imgs_lens)):
                                                                if i < sum(imgs_lens[:j+1]):
                                                                                faces_per_sample[j].append(detected_face)
                                                                                break
                                else:
                                                for j in range(len(imgs_lens)):
                                                                if i < sum(imgs_lens[:j+1]):
                                                                                k[j] += 1
                                                                                break

                # Fallback dummy face tensor if no faces found
                dummy_face = torch.zeros((1, 3, 48, 48)).to(self.device)

                # Ensure every sample has at least 1 frame
                for j in range(len(faces_per_sample)):
                                if len(faces_per_sample[j]) == 0:
                                                faces_per_sample[j] = [dummy_face.squeeze(0)]
                                                k[j] = imgs_lens[j] - 1  # Artificially set valid length = 1

                faces_per_sample = [torch.stack(sample_faces) for sample_faces in faces_per_sample]
                faces_padded = pad_sequence(faces_per_sample, batch_first=True, padding_value=0).to(self.device)  # [B, max_frames, 3, 48, 48]

                # Adjust lengths (ensuring min length = 1)
                imgs_lens2 = [max(imgs_lens[j] - k[j], 1) for j in range(len(imgs_lens))]

                
                B, max_frames, C, H, W =  faces_padded.shape  # Shape: [B, max_frames, 3, 48, 48]
                
                # Flatten frames into batch format for CNN processing
                faces_reshaped =faces_padded.view(B * max_frames, C, H, W)  # Shape: [B * max_frames, 3, 48, 48]

                faces_features = self.V( faces_reshaped)

                # Reshape back to batch format
                C_out, H_out, W_out = faces_features.shape[1:]  # Extract output channels, height, width
                faces_features = faces_features.view(B, max_frames, C_out, H_out, W_out)  # [B, max_frames, C_out, H_out, W_out]
                
                #print(faces_features.shape)
                # SAB block
                faces_features=self.sab(faces_features)

                extrcated_face_features=faces_features
                B, T, H, W, C = faces_features.shape
             
               
                if self.gmm_visual is None:
                   print("gmm video none")
               
                # Compute confidence scores 
                visual_confidence_scores, weighted_faces = compute_visual_confidence_scores(
                        faces_features,
                        self.confidence_net_visual_ffn,
                        self.gmm_visual,
                        self.confidence_fusion_visual
                )  # [B,T,C] , → [B, T, C], [B, T, C*H*W]
            
                class_visual_confidence_scores= visual_confidence_scores # [B,T, C]

                # Feature Reduction
                faces_features = self.visual_feature_proj(weighted_faces) # [B,T,C]
                
                # Apply Transformer encoder to capture temporal dependencies
                visual_features = self.v_transformer(faces_features, imgs_lens2, get_cls=False)
        
                


               

        # Process audio modality
        if 'a' in self.mod:
            audio_features,class_audio_confidence_scores,extracted_audio_features = self.A(waves,return_features)
            
            
           
            #audio_features = self.a_cross_modal_attention(query=text_features,key=audio_features,value=audio_features).squeeze(0)
            video_fused = self.av_attn(query=visual_features, key=audio_features, value=audio_features) # Video attends to audio
            audio_fused = self.va_attn(query=audio_features, key=visual_features, value=visual_features) # Audio attends to video
    
            #video_repr = video_fused.mean(dim=1)   # [B, 256]
            video_repr = torch.zeros(B, 256).to(self.device)
            for i, l in enumerate(imgs_lens2):
                 video_repr[i] = video_fused[i, :l].mean(dim=0)

            audio_repr = audio_fused.mean(dim=1)   # [B, 256]

            # Text attends to audio and video
            text_audio = self.ta_attn(query=text_features.unsqueeze(1), key=audio_repr.unsqueeze(1), value=audio_repr.unsqueeze(1))  # [B, Tt, D]
            text_video = self.tv_attn(query=text_features.unsqueeze(1), key=video_repr.unsqueeze(1), value=video_repr.unsqueeze(1))  # [B, Tt, D]


         

            all_logits.append(self.v_out(video_repr)) 
            all_logits.append(self.tv_out(text_video.squeeze(1))) 
            all_logits.append(self.ta_out(text_audio.squeeze(1))) 
            all_logits.append(self.a_out(audio_repr))
        
        
        

        # Fuse logits from different modalities
        if len(self.mod) == 1:
            return all_logits[0] , class_audio_confidence_scores,extracted_audio_features
        else:
            return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1) ,class_audio_confidence_scores,class_visual_confidence_scores,extracted_audio_features,extrcated_face_features
        
    
   

