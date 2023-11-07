import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        device,
        epochs,
        adam_betas,
        learning_rate,
        weight_decay,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.adam_betas = adam_betas
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            betas=self.adam_betas,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train_epoch(self):
        self.model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0
        for idx, img_label in enumerate(
            tqdm(self.train_dataloader, position=0, leave=True)
        ):
            #print(img_label["label"])
            img = img_label["image"].float().to(self.device)
            #print(img)
            label = img_label["label"].type(torch.uint8).to(self.device)
            y_pred = self.model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())

            loss = self.criterion(y_pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_running_loss += loss.item()
        train_loss = train_running_loss / (idx + 1)
        return train_loss, train_labels, train_preds

    def val_epoch(self):
        self.model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_label in enumerate(
                tqdm(self.val_dataloader, position=0, leave=True)
            ):
                img = img_label["image"].float().to(self.device)
                label = img_label["label"].type(torch.uint8).to(self.device)
                y_pred = self.model(img)
                y_pred_label = torch.argmax(y_pred, dim=1)

                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())

                loss = self.criterion(y_pred, label)
                val_running_loss += loss.item()
        val_loss = val_running_loss / (idx + 1)
        return val_loss, val_labels, val_preds

    def train(self):
        start = timeit.default_timer()
        for epoch in tqdm(range(self.epochs), position=0, leave=True):
            train_loss, train_labels, train_preds = self.train_epoch()
            val_loss, val_labels, val_preds = self.val_epoch()

            print("-" * 30)
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Validation Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print(
                f"Train Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}"
            )
            print(
                f"Validation Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}"
            )
            print("-" * 30)
        stop = timeit.default_timer()
        print(f"Training Time: {stop-start:.2f}s")

        torch.cuda.empty_cache()
