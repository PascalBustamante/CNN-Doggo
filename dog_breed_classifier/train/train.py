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

from utils.logger import Logger


# Initialize the logger with a log file path
log_file_path = "logs/training_log.txt"
logger = Logger(__name__, log_file_path)


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
        log_interval=10,
        early_stopping_patience=5,
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

        self.log_interval = log_interval
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.no_improvement_count = 0

    def train_epoch(self):
        self.model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0
        for idx, img_label in enumerate(
            tqdm(self.train_dataloader, position=0, leave=True)
        ):
            img = img_label["image"].float().to(self.device)
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

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Save the best model checkpoint
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if (epoch + 1) % self.log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}]")
                logger.info(f"Train Loss: {train_loss:.4f}")
                logger.info(f"Validation Loss: {val_loss:.4f}")

                # Log validation metrics
                val_accuracy = sum(
                    1 for x, y in zip(val_preds, val_labels) if x == y
                ) / len(val_labels)
                logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Learning Rate: {current_lr:.6f}")

                # Log model checkpoints
                if self.no_improvement_count == 0:
                    logger.info("Model checkpoint saved (best so far)")

                # Log epoch duration
                elapsed_time = timeit.default_timer() - start
                logger.info(f"Epoch Duration: {elapsed_time:.2f}s")

                # Log batch processing
                total_batches = len(self.train_dataloader)
                logger.info(f"Total Batches: {total_batches}")
                logger.info(
                    f"Average Batch Processing Time: {elapsed_time / total_batches:.2f}s"
                )

                if self.no_improvement_count >= self.early_stopping_patience:
                    # Log early stopping message (Early Stopping Messages)
                    logger.info(
                        "Early stopping triggered. No improvement for {self.early_stopping_patience} epochs."
                    )
                    break

                logger.info("-" * 30)

        stop = timeit.default_timer()
        logger.info(f"Training Time: {stop-start:.2f}s")
        logger.info("Training finished")

        torch.cuda.empty_cache()
