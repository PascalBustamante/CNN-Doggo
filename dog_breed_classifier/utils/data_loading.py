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
        # return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def get_dataloaders(self):
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

