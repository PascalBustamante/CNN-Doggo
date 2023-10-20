import torch
from torch import nn, optim
import pandas as pd
<<<<<<< HEAD
from torch.utils.data import DataLoader, Dataset, random_split
=======
from torch .utils.data import DataLoader, Dataset
>>>>>>> dad2c1da0521503b8c09c08b2cce9b51e8d1b8a7
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm


<<<<<<< HEAD
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

=======
train_df = pd.read_csv(r"C:\Users\pasca\CNN Doggo\MNIST\train.csv")
test_df = pd.read_csv(r"C:\Users\pasca\CNN Doggo\MNIST\test.csv")
submission_df = pd.read_csv(r"C:\Users\pasca\CNN Doggo\MNIST\sample_submission.csv")

#print(train_df.head())


train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED, shuffle=True)


class MNISTTrainDataset(Dataset):      ##change it to doggos DS
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
    

    def __getitem__(self, index):        ##this would have to change for it to take coloured images
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
        index = self.indicies[index]
        image = self.transform(image)

        return {"image": image, "index": index}
>>>>>>> dad2c1da0521503b8c09c08b2cce9b51e8d1b8a7
