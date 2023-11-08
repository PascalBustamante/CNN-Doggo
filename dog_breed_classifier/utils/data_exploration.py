import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def calculate_mean_std(image_files):
    # Define the transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Calculate the mean and standard deviation
    mean = 0.0
    std = 0.0
    for image_file in image_files:
        image = Image.open(image_file)
        image_tensor = transform(image)
        mean += image_tensor.mean()
        std += image_tensor.std()

    # Calculate the average mean and standard deviation
    mean /= len(image_files)
    std /= len(image_files)

    return mean, std


def calculate_label_distribution(id_to_breed, image_files):
    label_distribution = {}

    for image_file in image_files:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        label = id_to_breed[image_id]

        if label is not None:
            label_distribution[label] = label_distribution.get(label, 0) + 1

    return label_distribution
