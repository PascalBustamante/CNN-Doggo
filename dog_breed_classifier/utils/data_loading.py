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