import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm


class PatchEmbedding(nn.Module):
    """
    PatchEmbedding module for breaking down input images into patches.

    Args:
        embed_dim (int): Dimension of the embedding for each patch.
        patch_size (int): Size of each square patch.
        num_patches (int): Number of patches in the image.
        dropout (float): Dropout probability for regularization.
        in_channels (int): Number of input channels (e.g., 3 for RGB images).

    Attributes:
        patcher (nn.Sequential): Convolutional layer for patch extraction.
        cls_token (nn.Parameter): Learnable classification token.
        position_embeddings (nn.Parameter): Learnable positional embeddings.
        dropout (nn.Dropout): Dropout layer for regularization.

    Returns:
        torch.Tensor: Embeddings for the patches.
    """

    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        # Define a patcher using a convolutional layer
        self.patcher = nn.Sequential(
            nn.Conv2d(  # Look into swapping it with nn.Conv3d
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            nn.Flatten(2),
        )

        # Initialize the classification token and positional embeddings
        self.cls_token = nn.Parameter(
            torch.randn(size=(1, 1, embed_dim)), requires_grad=True
        )
        self.positoin_embeddings = nn.Parameter(
            torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Expand the classification token for all samples in the batch
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.positoin_embeddings + x
        x = self.dropout(x)
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model.

    Args:
        num_patches (int): Number of patches in the image.
        img_size (int): Size of the input image.
        num_classes (int): Number of output classes.
        patch_size (int): Size of each patch.
        embed_dim (int): Dimension of patch embeddings.
        num_encoders (int): Number of transformer encoders.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the encoder.
        dropout (float): Dropout probability.
        activation (str): Activation function for the encoder.
        in_channels (int): Number of input channels.

    Attributes:
        embeddings_block (PatchEmbedding): PatchEmbedding module.
        embeddings_blocks (nn.TransformerEncoder): Transformer encoder layers.
        mlp_head (nn.Sequential): Multilayer perceptron head for classification.
    """

    def __init__(
        self,
        num_patches,
        img_size,
        num_classes,
        patch_size,
        embed_dim,
        num_encoders,
        num_heads,
        hidden_dim,
        dropout,
        activation,
        in_channels,
    ) -> None:
        super().__init__()

        # Initialize the patch embeddings
        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        # Create a stack of Transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.embeddings_blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoders
        )

        # Define the MLP head for final classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.embeddings_blocks(x)
        x = self.mlp_head(x[:, 0, :])
        return x
