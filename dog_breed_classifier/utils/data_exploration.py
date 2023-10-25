import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Create variables containing paths to train / test data.
train_path = os.path.join('..', 'data', 'train')

# Step 2: Check the Distribution of Labels
train_labels_path = os.path.join('..', 'data', 'labels.csv')
train_labels = pd.read_csv(r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\labels.csv")

print(train_labels.describe())

hist = train_labels['breed'].hist(figsize=(16, 8), bins=120, xrot=90, xlabelsize=8)
plt.show()

# Step 3: Check the Size of the Images
image_sizes = []
for filename in os.listdir(train_path):
    with Image.open(os.path.join(train_path, filename)) as img:
        width, height = img.size
        image_sizes.append((width, height))

# Print out stats for viewing pleasure.
widths, heights = zip(*image_sizes)
print('Training data average (width, height): ', (np.mean(widths), np.mean(heights)))
print('Training data (minimum width, maximum width): ', (np.min(widths), np.max(widths)))
print('Training data (minimum height, maximum height): ', (np.min(heights), np.max(heights)))
print('Training data (STD of widths, STD of heights): ', (np.std(widths), np.std(heights)))

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
fig.suptitle('Histograms of training image widths, heights')
ax1.hist(widths, bins=100); ax1.set_xlabel('Width'); ax1.set_ylabel('No. of images')
ax2.hist(heights, bins=100); ax2.set_xlabel('Height'); ax2.set_ylabel('No. of images')
plt.show()

# Step 4: Check the Number of Images for Each Class
breed_counts = train_labels['breed'].value_counts()
print(breed_counts)



def calculate_mean_std(image_files):
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Calculate the mean and standard deviation
    mean = 0.
    std = 0.
    for image_file in image_files:
        image = Image.open(image_file)
        image_tensor = transform(image)
        mean += image_tensor.mean()
        std += image_tensor.std()

    # Calculate the average mean and standard deviation
    mean /= len(image_files)
    std /= len(image_files)

    return mean, std


