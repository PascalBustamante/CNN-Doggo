import optuna
import torch
import glob
import random
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import random_split

from models.model import ViT
from train.train import Trainer
from utils.data_loading import DataLoaderManager, create_dog_breed_enum
from utils.logger import Logger


def objective(trial, data_subset_ratio, image_files):
    # Static hyperparameters
    RANDOM_SEED = 42
    NUM_WORKERS = 8  # number of CPU cores for parallel data loading
    EPOCHS = 40
    NUM_CLASSES = 121
    PATCH_SIZE = 16
    IMG_SIZE = 224
    IN_CHANNELS = 3
    DROPOUT = 0.001
    HIDDEN_DIM = 768
    ADAM_WEIGHT_DECAY = 0
    ADAM_BETAS = (0.9, 0.999)
    ACTIVAITION = "gelu"
    NUM_ENCODERS = 4
    EMBED_DIM = (PATCH_SIZE**2) * IN_CHANNELS
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

    # Define the hyperparameters to be tuned
    LEARNING_RATE = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [8, 16, 32])
    NUM_HEADS = trial.suggest_int("num_heads", 4, 8, 16)

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

    log_file_path = "logs/hyperparameter_optimazation.txt"
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
    labels_df = pd.read_csv(
        r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\labels.csv"
    )

    # Create a list of unique dog breeds and a mapping of image IDs to breed labels
    breeds_list = labels_df["breed"].unique().tolist()
    id_to_breed = dict(zip(labels_df["id"], labels_df["breed"]))
    breeds = create_dog_breed_enum(breeds_list)

    # Take the {data_subset_ratio} and split the dataset into training and validation sets
    num_files_to_sample = int(
        len(image_files_train_val) * (data_subset_ratio)
    )  # check and make sure that the ratio is a ratio!!
    lengths = [
        int(0.9 * len(image_files_train_val)),
        len(image_files_train_val) - int(0.9 * len(image_files_train_val)),
    ]
    image_files_train_val_subset = random.sample(
        image_files_train_val, num_files_to_sample
    )
    image_files_train, image_files_val = random_split(
        image_files_train_val_subset, lengths
    )

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
        test_files=[],  # Leave empty
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

    # Evaluate your model on the validation set and get the validation loss
    validation_loss = evaluate_model(model, val_dataloader)

    # Return the metric to be optimized (e.g., validation loss)
    return validation_loss
