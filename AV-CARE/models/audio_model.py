
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention_module import SpatialAttentionBlock, TemporalAttentionBlock
from models.confidences import  ConfidenceFusion, VarianceConfidenceNN, compute_audio_confidence_scores
from models.gmm import  load_gmm_audio




class CNN_BiGRU(nn.Module):
    """
    A **lighter** CNN + BiGRU model for emotion recognition from raw audio.
    Reduces depth while maintaining performance.
    """
    def __init__(self, n_input=1, hidden_dim=64, n_layers=4, n_output=4, stride=4, n_channel=18):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #  Convolutional Feature Extraction (Reduced Depth)**
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=160, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel * 2)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv1d(n_channel * 2, n_channel * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_channel * 2)
        self.relu3 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(n_channel * 2, n_channel * 4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(n_channel * 4)
        self.relu4 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.05)

        self.conv5 = nn.Conv1d(n_channel * 4, n_channel * 4, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(n_channel * 4)
        self.relu5 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool1d(4)

        self.conv6 = nn.Conv1d(n_channel * 4, n_channel * 8, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(n_channel * 8)
        self.relu6 = nn.LeakyReLU()
        self.conv7 = nn.Conv1d(n_channel * 8, n_channel * 8, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(n_channel * 8)
        self.relu7 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.pool4 = nn.MaxPool1d(4)

        # **Initialize GMM Placeholder**
        self.gmm = None  # Will be assigned later after training

        # FFN for confidence learning
        self.variance_ffn = VarianceConfidenceNN(feature_dim=144)
        
        # Confidence Fusion Model
        self.fusion_model = ConfidenceFusion()

        # Bi-GRU 
        self.gru = nn.GRU(
            input_size=n_channel * 8, 
            hidden_size=72, 
            num_layers=4, 
            batch_first=True, 
            bidirectional=True
        )

        self.tab=TemporalAttentionBlock(embed_dim=144,seq_dim=81, num_heads=4)
        self.fc1 = nn.Linear(144, 72)  # Fully connected layer

    def reload_gmm_audio(self,epoch):
        """
        Reload the GMM model from disk.
        This should be called after each epoch to ensure we use the latest GMM.
        """
        self.gmm = load_gmm_audio(epoch)    

    def forward(self, x,return_features):
        """ Forward pass through CNN and BiGRU """

        x = x.unsqueeze(1)  # Shape: [B, C=1, T]

        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(self.bn2(x))

        x = self.conv3(x)
        x = self.relu3(self.bn3(x))
        x = self.pool2(x)

        x = self.conv4(x)
        x = self.relu4(self.bn4(x))
        #x = self.dropout1(x)

        x = self.conv5(x)
        x = self.relu5(self.bn5(x))
        x = self.pool3(x)

        x = self.conv6(x)
        x = self.relu6(self.bn6(x))
        x = self.conv7(x)
        x = self.relu7(self.bn7(x))
        #x = self.dropout2(x)
        x = self.pool4(x)


        x = x.permute(0, 2, 1)  # Change shape  [B, T, C]

        cnn_features=x
        

        if self.gmm is None:
           print("audio gmm none")
        
        # **Compute Confidence Scores and Select Features**
        confidence_scores, x = compute_audio_confidence_scores(
        x,   self.variance_ffn, self.gmm,self.fusion_model
        ) #[B,T,C]

       
        
        
        # **Bi-GRU**
        gru_out, _ = self.gru(x)

        

        gru_out=self.tab(gru_out)

        gru_out = self.fc1(gru_out)
        

        out=gru_out

        

        if return_features:
            return out, confidence_scores, cnn_features  # Return features for GMM
        return out, confidence_scores
    
    
    


