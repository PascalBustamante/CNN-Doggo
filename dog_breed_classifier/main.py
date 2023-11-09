import torch
import random
import glob
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt

# Import custom modules
from models.model import ViT
from utils.data_loading import DataLoaderManager, create_dog_breed_enum
from train.train import Trainer
from utils.logger import Logger

# Define hyperparameters and configurations
RANDOM_SEED = 42
NUM_WORKERS = 8  # number of CPU cores for parallel data loading
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_CLASSES = 121
PATCH_SIZE = 16
IMG_SIZE = 224
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

# Set random seed for various libraries
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Choose the device for training (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

log_file_path = "logs/main.txt"
logger = Logger(__name__, log_file_path)

# Log hyperparameters
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Number of epochs: {EPOCHS}")
logger.info(f"Learning rate: {LEARNING_RATE}")
logger.info(f"Device: {device}")

# Define the transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # resize images to 224x224
        transforms.ToTensor(),  # convert image to PyTorch tensor
    ]
)

# Load image file paths, and labels
image_files_train_val = glob.glob(
    r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\train/*.jpg"
)
image_files_test = glob.glob(
    r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\test/*.jpg"
)
labels_df = pd.read_csv(
    r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\labels.csv"
)

# Create a list of unique dog breeds and a mapping of image IDs to breed labels
breeds_list = labels_df["breed"].unique().tolist()
id_to_breed = dict(zip(labels_df["id"], labels_df["breed"]))
breeds = create_dog_breed_enum(breeds_list)

# Split the dataset into training and validation sets
lengths = [
    int(0.9 * len(image_files_train_val)),
    len(image_files_train_val) - int(0.9 * len(image_files_train_val)),
]
image_files_train, image_files_val = random_split(image_files_train_val, lengths)

model = ViT(
    num_patches=NUM_PATCHES,
    img_size=IMG_SIZE,
    num_classes=NUM_CLASSES,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_encoders=NUM_ENCODERS,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT,
    activation=ACTIVAITION,
    in_channels=IN_CHANNELS,
).to(device)

DM = DataLoaderManager(
    batch_size=BATCH_SIZE,
    train_files=image_files_train,
    val_files=image_files_val,
    test_files=image_files_test,
    id_to_breed=id_to_breed,
    num_workers=NUM_WORKERS,
)

# Get data loaders
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

# Train the ViT model
ViTTrainer.train()
