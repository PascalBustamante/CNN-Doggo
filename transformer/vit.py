import torch
from torch import nn


learning_rate = 1e-4
num_classes = 10
patch_size = 4
img_size = 28
in_channels = 1
dropout = 0.001
hidden_dim = 768
adam_weight_decay = 0
adam_betas = (0.9, 0.999)
activaition = "gelu"
num_encoder = 4
embed_dim = (patch_size ** 2) * in_channels
num_patches = (img_size // patch_size) ** 2 

device = "cuda" if torch.cuda.is_available() else "cpu"


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.positoin_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0,2,1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.dropout(x)
        return x
    

model = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels).to(device)
x = torch.randn(512, 1, 28, 28)
print(model(x).shape)