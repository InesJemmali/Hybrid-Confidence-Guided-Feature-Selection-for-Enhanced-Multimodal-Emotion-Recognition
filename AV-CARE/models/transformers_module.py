import math
from typing import Optional, List
import torch
from torch import nn


#add zeros to tensors to augment its len
def padTensor(t: torch.tensor, targetLen: int) -> torch.tensor:
    """
    Pads a tensor with zeros along the first dimension to reach the target length.

    This function augments the length of the input tensor by appending zeros to it. It's useful for
    ensuring that all tensors in a batch have the same length, which is often required for batch processing.

    Args:
        t (torch.Tensor): The input tensor to be padded with shape `(original_length, dim)`.
        targetLen (int): The desired length after padding.

    Returns:
        torch.Tensor: The padded tensor with shape `(targetLen, dim)`.

    """
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)




class WrappedTransformerEncoder(nn.Module):

    
    """
    A wrapped Transformer Encoder module with optional classification token prepending and padding support.

    This module encapsulates PyTorch's `TransformerEncoder`, providing additional functionality such as
    prepending a classification token and handling variable-length input sequences with padding.

    Args:
        dim (int): The number of expected features in the encoder input (model dimension).
        num_layers (int): The number of `TransformerEncoderLayer` layers in the encoder.
        num_heads (int): The number of heads in the multihead attention models.

    Attributes:
        dim (int): The model dimension.
        encoder (nn.TransformerEncoder): The Transformer encoder composed of multiple layers.
        cls_emb (nn.Embedding): Embedding layer for the classification token.
    """

    def __init__(self, dim, num_layers, num_heads):
        super(WrappedTransformerEncoder, self).__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=dim)

    def prepend_cls(self, inputs):
        """
        Prepends a classification token embedding to each sequence in the batch.

        This method adds a learnable classification token (`CLS`) at the beginning of each input sequence,
        which can be used to aggregate information for classification tasks.

        Args:
            inputs (torch.Tensor): Input tensor with shape `(batch_size, seq_length, dim)`.

        Returns:
            torch.Tensor: Tensor with the classification token prepended, shape `(batch_size, seq_length + 1, dim)`.
        """
        
        index = torch.LongTensor([0]).to(device=inputs.device)
        
        cls_emb = self.cls_emb(index)
       
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.dim)
      
        outputs = torch.cat((cls_emb, inputs), dim=1)
       
        return outputs

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, get_cls: Optional[bool] = False):
        """
        Processes the input sequence through the model, optionally prepending a classification token
        and handling variable sequence lengths with padding.

        Args:
            inputs (torch.Tensor): Input tensor with shape `(total_seq_length, dim)` or `(batch_size, seq_length, dim)`.
            lens (Optional[List[int]]): List of sequence lengths for each batch element. If provided, the input will 
                be split, padded, and a mask will be created to handle variable lengths. Default is `None`.
            get_cls (Optional[bool]): If `True`, a classification token will be prepended to each sequence, 
                and only the output of this token will be returned. Default is `False`.

        Returns:
            torch.Tensor: 
                - If `get_cls` is `True`: The output corresponding to the classification token, with shape `(batch_size, dim)`.
                - Otherwise: The processed sequence without the classification token, with shape `(batch_size, seq_length, dim)`.
        """
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * (l + int(get_cls)) + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)

            inputs = list(inputs.split(lens, dim=0))
            inputs = [padTensor(inp, max_len) for inp in inputs]
            inputs = torch.stack(inputs, dim=0)
            # print(inputs.shape)
        else:
            mask = None

        if get_cls:
            inputs = self.prepend_cls(inputs)

        inputs = inputs.permute(1, 0, 2) 
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        if get_cls:
            return inputs[0]

        return inputs[1:].permute(1, 0, 2)
    

class WrappedTransformerEncoderVideo(nn.Module):
    """
    Transformer Encoder for visual features with optional CLS token prepending.

    This module encapsulates PyTorch's `TransformerEncoder`, adding:
    - A learnable CLS token for capturing global sequence representation.
    - Handling of variable-length sequences via padding masks.

    Args:
        dim (int): Feature dimension (e.g., 512).
        num_layers (int): Number of Transformer layers.
        num_heads (int): Number of attention heads.

    Attributes:
        encoder (nn.TransformerEncoder): Transformer with multiple layers.
        cls_emb (nn.Embedding): CLS token embedding.
    """

    def __init__(self, dim: int, num_layers: int, num_heads: int):
        super(WrappedTransformerEncoderVideo, self).__init__()
        self.dim = dim
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable CLS token
        self.cls_emb = nn.Embedding(1, dim)

    def prepend_cls(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Prepends a CLS token to the input sequence.

        Args:
            inputs (torch.Tensor): Shape `(batch_size, seq_length, feature_dim)`

        Returns:
            torch.Tensor: Shape `(batch_size, seq_length + 1, feature_dim)`
        """
        batch_size = inputs.shape[0]
        cls_token = self.cls_emb.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 1, dim)
        outputs = torch.cat((cls_token, inputs), dim=1)  # Prepend CLS token at position 0
        return outputs

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, get_cls: bool = False):
        """
        Passes visual features through the Transformer Encoder.

        Args:
            inputs (torch.Tensor): Shape `(batch_size, seq_length=8, dim)`
            lens (Optional[List[int]]): Sequence lengths for padding handling.
            get_cls (bool): If `True`, returns only the CLS token output.

        Returns:
            torch.Tensor:
                - If `get_cls=True`: Output CLS token `(batch_size, dim)`.
                - Otherwise, full output `(batch_size, seq_length, dim)`.
        """
        batch_size, seq_length, feature_dim = inputs.shape  # (4, 8, 512)

        # Handle padding mask
        if lens is not None:
            max_len = max(lens)
            mask = torch.tensor([[False] * l + [True] * (max_len - l) for l in lens]).to(inputs.device)

            # Ensure mask matches the actual seq_len
            mask = mask[:, :seq_length]
        else:
            mask = None

        # Add CLS token if needed
        if get_cls:
            inputs = self.prepend_cls(inputs)  # Adds CLS token

            if mask is not None:
                cls_mask = torch.zeros(mask.shape[0], 1, dtype=torch.bool).to(mask.device)  # No padding for CLS
                mask = torch.cat((cls_mask, mask), dim=1)  # Update mask for new seq_len

        # Transformer expects (seq_len, batch_size, feature_dim)
        inputs = inputs.permute(1, 0, 2)  

        # Forward through Transformer Encoder
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask)

        if get_cls:
            return inputs[0]  # Return CLS token output

        return inputs.permute(1, 0, 2)  # Convert back to (batch_size, seq_length, feature_dim)
