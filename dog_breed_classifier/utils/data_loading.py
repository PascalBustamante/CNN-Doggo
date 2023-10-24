import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm
from PIL import Image
from enum import Enum, auto

class DataLoaderManager:
    def __init__(
        self, batch_size, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    ):
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def split_data(self):
        total_size = len(self.dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self):
        train_dataset, val_dataset, test_dataset = self.split_data()
        self.train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def get_dataloaders(self):
        return self.train_dataloader, self.val_dataloader, self.test_dataloader


class DOGGOTrainDataset(Dataset):
    def __init__(self, image_paths, labels):
        super().__init__()
        self.image_paths = image_paths
        self.labels, self.label_indicies = torch.unique(labels, return_inverse=True)
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                #transforms.ToPILImage(),
                #transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(
        self, index
    ):  ##this would have to change for it to take coloured images
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]
        image = self.transform(image)

        return {"image": image, "label": label}
    

def create_dog_breed_enum(breeds):
    return Enum("Dog Breed", {breed.upper(): i for i, breed in enumerate(breeds, start=1)})