import torch
import random
import glob
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt


from models.model import ViT
from utils.data_loading import DataLoaderManager, create_dog_breed_enum
from train.train import Trainer


RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_CLASSES = 10  # for the 1- digits
PATCH_SIZE = 16
IMG_SIZE = 244
IN_CHANNELS = 3 
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVAITION = "gelu"
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE**2) * IN_CHANNELS
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
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # resize images to 224x224
        transforms.ToTensor(),  # convert image to PyTorch tensor
    ]
)

# Load the dataset
image_files_train_val = glob.glob(
    r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\train/*.jpg"
)
image_files_test = glob.glob(
    r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\test/*.jpg"
)
labels_df = pd.read_csv(
    r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\labels.csv"
)

breeds_list = labels_df["breed"].unique().tolist()
id_to_breed = dict(zip(labels_df["id"], labels_df["breed"]))
breeds = create_dog_breed_enum(breeds_list)

# Split the dataset
lengths = [
    int(0.9 * len(image_files_train_val)),
    len(image_files_train_val) - int(0.9 * len(image_files_train_val)),
]
image_files_train, image_files_val = random_split(image_files_train_val, lengths)

model = ViT(
    NUM_PATCHES,
    IMG_SIZE,
    NUM_CLASSES,
    PATCH_SIZE,
    EMBED_DIM,
    NUM_ENCODERS,
    NUM_HEADS,
    HIDDEN_DIM,
    DROPOUT,
    ACTIVAITION,
    IN_CHANNELS,
).to(device)

DM = DataLoaderManager(
    batch_size=BATCH_SIZE,
    train_files=image_files_train,
    val_files=image_files_val,
    test_files=image_files_test,
    id_to_breed=id_to_breed,
)

train_dataloader = DM.get_dataloader("train")
val_dataloader = DM.get_dataloader("val")
test_dataloader = DM.get_dataloader("test")

ViTTrainer = Trainer(
    model=model,
    device=device,
    adam_betas=ADAM_BETAS,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    weight_decay=ADAM_WEIGHT_DECAY,
)


ViTTrainer.train()