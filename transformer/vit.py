import torch
from torch import nn, optim
import pandas as pd
from torch .utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm


RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 40
LEARNING_RATE = 1E-4
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNELS = 1
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVAITION = "gelu"
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2 

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        x = self.positoin_embeddings + x
        x = self.dropout(x)
        return x
    

#model = PatchEmbedding(EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS).to(device)
#x = torch.randn(512, 1, 28, 28).to(device)
#print(model(x).shape)


class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim,  dropout, activation, in_channels) -> None:
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,activation=activation, batch_first=True, norm_first=True)
        self.embeddings_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )


    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.embeddings_blocks(x)
        x = self.mlp_head(x[:, 0, :])
        return x 
    
#model = ViT(NUM_PATCHES, IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMBED_DIM, NUM_ENCODERS, NUM_HEADS, HIDDEN_DIM, DROPOUT, ACTIVAITION, IN_CHANNELS).to(device)
#x = torch.randn(512, 1, 28, 28).to(device)
#print(model(x).shape)


train_df = pd.read_csv(r"C:\Users\pasca\CNN Doggo\MNIST\train.csv")
test_df = pd.read_csv(r"C:\Users\pasca\CNN Doggo\MNIST\test.csv")
submission_df = pd.read_csv(r"C:\Users\pasca\CNN Doggo\MNIST\sample_submission.csv")

#print(train_df.head())


train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED, shuffle=True)


class MNISTTrainDataset(Dataset):
    def __init__(self, images, labels, indicies):
        super().__init__()
        self.images = images
        self.labels = labels
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        image = self.images[index].reshape((28, 28)).astype(np.uint8)
        label = self.labels[index]
        index = self.indicies[index]
        image = self.transform(image)

        return {"image": image, "label": label, "index": index}


class MNISTValDataset(Dataset):
    def __init__(self, images, labels, indicies):
        super().__init__()
        self.images = images
        self.labels = labels
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        image = self.images[index].reshape((28, 28)).astype(np.uint8)
        label = self.labels[index]
        index = self.indicies[index]
        image = self.transform(image)

        return {"image": image, "label": label, "index": index}
    

class MNISTSubmitDataset(Dataset):
    def __init__(self, images, indicies):
        super().__init__()
        self.images = images
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        image = self.images[index].reshape((28, 28)).astype(np.uint8)
        label = self.labels[index]
        index = self.indicies[index]
        image = self.transform(image)

        return {"image": image, "index": index}
    

plt.figure()
f, axarr = plt.subplots(1, 3)

train_dataset = MNISTTrainDataset(train_df.iloc[:, 1:].values.astype(np.uint8), train_df.iloc[:, 0].values, train_df.index.values) ##ommit label
print(len(train_dataset))
print(train_dataset[0])
axarr[0].imshow(train_dataset[0]["image"].squeeze(), cmap="gray")
axarr[0].set_title("Train Image")
print("-"*30)


val_dataset = MNISTValDataset(val_df.iloc[:, 1:].values.astype(np.uint8), val_df.iloc[:, 0].values, val_df.index.values) ##ommit label
print(len(val_dataset))
print(val_dataset[0])
axarr[0].imshow(val_dataset[0]["image"].squeeze(), cmap="gray")
axarr[0].set_title("Val Image")
print("-"*30)


test_dataset = MNISTSubmitDataset(test_df.values.astype(np.uint8), test_df.index.values) ##ommit label
print(len(test_dataset))
print(test_dataset[0])
axarr[0].imshow(test_dataset[0]["image"].squeeze(), cmap="gray")
axarr[0].set_title("test Image")
print("-"*30)


plt.show()