import torch
import random
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import random_split

from models.model import ViT
from utils.data_loading import DataLoaderManager
from train.train import Trainer


RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 40
LEARNING_RATE = 1E-4
NUM_CLASSES = 10     #for the 1- digits
PATCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNELS = 1      #1 because MNIST is gray scale 
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

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images to 224x224
    transforms.ToTensor(),  # convert image to PyTorch tensor
])

# Load the dataset
train_val_df = datasets.ImageFolder(root=r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\train", transform=transform)
test_df = datasets.ImageFolder(root=r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\test", transform=transform)

# Split the dataset
lengths = [int(0.9 * len(train_val_df)), len(train_val_df) - int(0.9 * len(train_val_df))]
train_df, val_df = random_split(train_val_df, lengths)


model = ViT(NUM_PATCHES, IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMBED_DIM, NUM_ENCODERS, NUM_HEADS, HIDDEN_DIM, DROPOUT, ACTIVAITION, IN_CHANNELS).to(device)