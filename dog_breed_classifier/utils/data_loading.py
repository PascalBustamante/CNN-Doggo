import torch
import os
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

from utils.data_exploration import calculate_mean_std


class DataLoaderManager:
    def __init__(self, batch_size, train_files, val_files, test_files, id_to_breed):
        self.batch_size = batch_size
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.id_to_breed = id_to_breed
        self.mean, self.std = self.calculate_mean_std(self.train_files)

    def calculate_mean_std(self, image_files):
        # function from utils
        return calculate_mean_std(image_files)

    ##build get_dataset(self, dataset_type)

    def get_dataloader(self, dataset_type):
        if dataset_type == "train":
            transform = transforms.Compose(
                [
                    #transforms.ToPILImage(),
                    #transforms.RandomRotation(15),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            dataset = DOGGODataset(
                self.train_files,
                self.id_to_breed,
                dataset_type="train",
                transform=transform,
            )

        elif dataset_type in ["val", "test"]:
            transform = transforms.Compose(
                [
                    #transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            if dataset_type == "val":
                dataset = DOGGODataset(
                    self.val_files,
                    self.id_to_breed,
                    dataset_type="val",
                    transform=transform,
                )
            else:
                dataset = DOGGODataset(
                    self.test_files,
                    self.id_to_breed,
                    dataset_type="test",
                    transform=transform,
                )

        else:
            raise ValueError("dataset_type should be 'train', 'val', or 'test'")

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader


class DOGGODataset(Dataset):
    def __init__(self, image_files, id_to_breed, dataset_type, transform=None):
        super().__init__()
        self.image_files = image_files
        self.id_to_breed = id_to_breed
        self.dataset_type = dataset_type
        self.transform = transform

        #Create Enum of breeds
        self.breeds_enum = create_dog_breed_enum(set(id_to_breed.values()))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_id = os.path.splitext(os.path.basename(image_file))[0]

        image = Image.open(image_file)
        if self.transform:
            image = self.transform(image)

        if self.dataset_type == "test": #for later differences, like maybe not returning labels for future test
            breed = self.id_to_breed[image_id]
            label = self.breeds_enum[breed.upper()].value
            label = torch.tensor(label, dtype=torch.long)  #Convert the label to a tensor   
            return {'image': image, 'label': label}   #turn it into a dict 
        else:
            breed = self.id_to_breed[image_id]
            label = self.breeds_enum[breed.upper()].value
            label = torch.tensor(label, dtype=torch.long)  #Convert the label to a tensor
            return {'image': image, 'label': label}  #turn it into a dict 


def create_dog_breed_enum(breeds):
    return Enum(
        "Dog Breed", {breed.upper(): i for i, breed in enumerate(breeds, start=1)}
    )

