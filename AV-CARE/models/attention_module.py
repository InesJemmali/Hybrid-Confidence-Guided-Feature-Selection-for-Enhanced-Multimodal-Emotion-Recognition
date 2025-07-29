import torch
import torch.nn as nn


class CrossModalAttentionT(nn.Module):
    """
    Cross-Modal Attention Layer for Text attending to Visual-Audio Features.


    Parameters:
        text_dim (int): Dimension of the text input features.
        va_dim (int): Dimension of the visual/audio input features.
        num_heads (int): Number of attention heads for the multi-head attention mechanism. Default is 4.
        

    """    

    def __init__(self, text_dim, va_dim, num_heads):
        super(CrossModalAttentionT, self).__init__()
        self.text_dim = text_dim
        self.va_dim = va_dim
        self.num_heads = num_heads

        # Linear layers to transform visual features to the same dimension as text features
        self.visual_to_text_dim = nn.Linear(va_dim, text_dim)
        self.attention = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(text_dim)
        
        # Linear layer to project the output back to the desired dimension
        self.output_projection = nn.Linear(text_dim, va_dim)

    def forward(self, query, key, value):
        """
        Forward pass of the CrossModalAttentionVA module. Applies multi-head attention 
        where text features attend to visual/audio features.

        Arguments:
            query (Tensor): Visual/audio features, shape (batch_size, va_dim).
            key (Tensor): Text features for attention, shape (batch_size, text_dim).
            value (Tensor): Text features for attention, shape (batch_size, text_dim).
            


        Returns:
            Tensor: The output features after attention and projection, shape (batch_size, text_dim).
        """

        # Transform visual features to the same dimension as text features
        key = self.visual_to_text_dim(key).unsqueeze(0)
        value = self.visual_to_text_dim(value).unsqueeze(0)
        query=query.unsqueeze(0)
       
 
        # Apply multi-head attention: text features (query) attend to visual features (key, value)
        attn_output, _ = self.attention(query, key, value)
        
        # Add & Norm: Add the attention output to the query and normalize
        attn_output = self.norm( value+attn_output)
        return  self.output_projection(attn_output)

class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Layer for Visual-Audio attending to Text Features.

    
    Parameters:
        text_dim (int): The dimensionality of the text features.
        va_dim (int): The dimensionality of the visual-audio features.
        num_heads (int): The number of attention heads for the multi-head attention.
        emb_dim (int): Dimension of the shared embedding space. Default is 256.

    """      
    def __init__(self,text_dim, va_dim, num_heads=4,emb_dim=256):  # Adjusted dimensions
        super(CrossModalAttention, self).__init__()
        self.text_dim = text_dim
        self.va_dim = va_dim
        self.num_heads = num_heads

        # Linear layers to transform visual features to the same dimension as text features
        self.va_to_emb_dim = nn.Linear(va_dim, emb_dim)
        self.text_to_emb_dim = nn.Linear(text_dim, emb_dim)
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(emb_dim)
        
        # Linear layer to project the output back to the desired dimension
        self.output_projection = nn.Linear(emb_dim, va_dim)

    def forward(self, query, key, value):

        """
        Forward pass of the CrossModalAttentionVA module. Applies multi-head attention 
        where text features attend to visual/audio features.

        Arguments:
            query (Tensor): Visual/audio features, shape (batch_size, va_dim).
            key (Tensor): Text features for attention, shape (batch_size, text_dim).
            value (Tensor): Text features for attention, shape (batch_size, text_dim).
            


        Returns:
            Tensor: The output features after attention and projection, shape (batch_size, text_dim).
        """
        

        # Transform visual features to the same dimension as text features
        key = self.va_to_emb_dim(key).unsqueeze(0)
        value = self.va_to_emb_dim(value).unsqueeze(0)
        query = self.text_to_emb_dim(query).unsqueeze(0)
       
        # Apply multi-head attention: text features (query) attend to visual features (key, value)
        attn_output, _ = self.attention(query, key, value)
        
        # Add & Norm: Add the attention output to the query and normalize
        attn_output = self.norm(value + attn_output)
        return self.output_projection(attn_output)



class CrossModalAttentionVA(nn.Module):
    """
    Cross-Modal Attention Layer for Visual-Audio attending to Text Features.

    
    Parameters:
        text_dim (int): The dimensionality of the text features.
        va_dim (int): The dimensionality of the visual-audio features.
        num_heads (int): The number of attention heads for the multi-head attention.
        emb_dim (int): Dimension of the shared embedding space. Default is 256.

    """      
    def __init__(self,text_dim, va_dim, num_heads=4,emb_dim=256):  # Adjusted dimensions
        super(CrossModalAttentionVA, self).__init__()
        self.text_dim = text_dim
        self.va_dim = va_dim
        self.num_heads = num_heads

        # Linear layers to transform visual features to the same dimension as text features
        self.va_to_emb_dim = nn.Linear(va_dim, emb_dim)
        self.text_to_emb_dim = nn.Linear(text_dim, emb_dim)
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(emb_dim)
        
        # Linear layer to project the output back to the desired dimension
        self.output_projection = nn.Linear(emb_dim, text_dim)

    def forward(self, query, key, value):

        """
        Forward pass of the CrossModalAttentionVA module. Applies multi-head attention 
        where text features attend to visual/audio features.

        Arguments:
            query (Tensor): Visual/audio features, shape (batch_size, va_dim).
            key (Tensor): Text features for attention, shape (batch_size, text_dim).
            value (Tensor): Text features for attention, shape (batch_size, text_dim).
            


        Returns:
            Tensor: The output features after attention and projection, shape (batch_size, text_dim).
        """
        
        # Transform visual features to the same dimension as text features
        query = self.va_to_emb_dim(query).unsqueeze(0)
        value = self.text_to_emb_dim(value).unsqueeze(0)
        key = self.text_to_emb_dim(key).unsqueeze(0)
       
        # Apply multi-head attention: text features (query) attend to visual features (key, value)
        attn_output, _ = self.attention(query, key, value)
        
        # Add & Norm: Add the attention output to the query and normalize
        attn_output = self.norm(value + attn_output)
        return self.output_projection(attn_output)

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for sequences"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pos_embed[:, :x.size(1), :]

class TemporalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, seq_dim,num_heads):
        super(TemporalAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.pos_encoding = LearnablePositionalEncoding(seq_dim, embed_dim)

    def forward(self, x):
        # Debugging print
       
        x = self.pos_encoding(x)  # [B, T, D]

        x = x.permute(1, 0, 2)  # Now shape is [T, B, C]
        # x must have shape [T, B, embed_dim]
        attn_output, _ = self.attention(x, x, x) # → [T, B, D]
        
        x = x.permute(1, 0, 2) 
        attn_output =  attn_output.permute(1, 0, 2) 

        # Residual Connection & Normalization
        x1 = self.norm( x+ attn_output)  # Ensure proper residual connection
        x2 = self.ffn(x)
        x = self.norm(x1 + x2)
        
        return x
  



class SpatialAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, height, width):
        """
        Applies attention over spatial positions (H×W) per frame.

        Args:
            embed_dim: number of channels (C)
            num_heads: number of attention heads
            height, width: spatial dimensions of the input
        """
        super().__init__()
        self.height = height
        self.width = width
        self.spatial_tokens = height * width  # S = H×W

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Learnable 2D positional encoding [1, H*W, C]
        self.pos_embed = nn.Parameter(torch.randn(1, self.spatial_tokens, embed_dim))

    def forward(self, x):
        """
        x: Tensor of shape [B, T, C, H, W]
        Returns: Tensor of shape [B, T, C, H, W]
        """
       
        B, T, C, H, W = x.shape
        assert H == self.height and W == self.width, f"Expected spatial size ({self.height}, {self.width}), got ({H}, {W})"

        # Flatten spatial grid → [B*T, C, H*W]
        x = x.view(B * T, C, H * W).permute(0, 2, 1)  # [B*T, S, C]

        # Add 2D positional encoding
        x = x + self.pos_embed  # [B*T, S, C]

        # Prepare for MultiheadAttention: [S, B*T, C]
        x = x.permute(1, 0, 2)  # [S, B*T, C]
        attn_output, _ = self.attention(x, x, x)  # [S, B*T, C]
        x = x.permute(1, 0, 2)                    # [B*T, S, C]
        attn_output = attn_output.permute(1, 0, 2)

        # Residual + norm
        x = self.norm1(x + attn_output)
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)

        # Restore to [B, T, C, H, W]
        x = x.permute(0, 2, 1).contiguous().view(B, T, C, H, W)
        return x
    




class  MultimodalAttentionBlock(nn.Module):
    """
    Generalized Multimodal Attention Block (MAB) for fusing Text with Audio OR Video.

    - Text provides context for the secondary modality.
    - The text representation remains unchanged.
    - Can be used for either Text-Audio or Text-Video fusion.

    Parameters:
        text_dim (int): Dimension of text features.
        modality_dim (int): Dimension of the secondary modality (audio/video).
        embed_dim (int): Common embedding dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, text_dim, modality_dim, embed_dim, num_heads=4):
        super(MultimodalAttentionBlock, self).__init__()

        # Projection layers to align text and secondary modality to shared embedding space
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.modality_proj = nn.Linear(modality_dim, embed_dim)

        # Cross-Attention: The modality attends to the text (text guides the modality)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Normalization and Feedforward layers
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, text_feat, modality_feat):
        """
        Forward pass for multimodal fusion.

        Inputs:
            text_feat: (batch, seq_len_text, text_dim)
            modality_feat: (batch, seq_len_modality, modality_dim)  # Can be audio or video features

        Returns:
            refined_modality: The refined representation of the secondary modality influenced by text.
        """

        # Project features to shared embedding space
        text_feat = self.text_proj(text_feat)  # (batch, seq_len_text, embed_dim)
        modality_feat = self.modality_proj(modality_feat)  # (batch, seq_len_modality, embed_dim)

        # Cross-Attention: Modality attends to Text (Text as Key & Value)
        attn_output, _ = self.cross_attention(modality_feat, text_feat, text_feat)

        # Residual Connection & Normalization (applied to the secondary modality)
        refined_modality = self.norm(modality_feat + attn_output)

        # Feedforward Network for feature transformation
        refined_modality = self.norm(refined_modality + self.ffn(refined_modality))

        return refined_modality  # The updated audio/video feature representation


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_value_dim, embed_dim, num_heads=4, dropout=0.1):
        """
        Cross-modal attention block with internal projection.

        Args:
            query_dim (int): Input dimension of query modality (e.g., video).
            key_value_dim (int): Input dimension of key/value modality (e.g., audio or text).
            embed_dim (int): Shared attention dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(CrossModalAttentionBlock, self).__init__()

        # Independent projections for each modality
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_value_dim, embed_dim)
        self.value_proj = nn.Linear(key_value_dim, embed_dim)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Post-attention residual + normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        """
        Args:
            query: [B, T_q, D_q]
            key:   [B, T_k, D_k]
            value: [B, T_k, D_k]
        Returns:
            attended: [B, T_q, embed_dim]
        """
        # Project to shared embedding space
        query_proj = self.query_proj(query).permute(1, 0, 2)  # [T_q, B, D]
        key_proj = self.key_proj(key).permute(1, 0, 2)        # [T_k, B, D]
        value_proj = self.value_proj(value).permute(1, 0, 2)  # [T_k, B, D]

      

        # Apply attention
        attn_output, _ = self.attn(query_proj, key_proj, value_proj)  # [B, T_q, embed_dim]

        # Convert back to [B, T_q, D]
        attn_output = attn_output.permute(1, 0, 2)

        # Residual connection and normalization
        out = self.norm(query_proj.permute(1, 0, 2) + self.dropout(attn_output))  # [B, T_q, embed_dim]
        return out
