# Dog Breed Classifier 
## Overview
The Dog Breed Classifier is a machine learning project that aims to classify dog breeds using a combination of ViT and CNN. The project includes various modules such as model training, hyperparameter optimization, and data pipeline.

## Data Loading
The data loading module manages the loading and processing of data for the Dog Breed Classifier. It includes functionality for creating data loaders, calculating dataset statistics, and implementing a custom dataset class.

Data Loader Manager
The DataLoaderManager class orchestrates the loading of training, validation, and test datasets. It calculates dataset statistics, such as mean, standard deviation, and label distribution, which are crucial for normalization and understanding the data distribution.
The get_dataloader method creates a PyTorch DataLoader for a specified dataset type (train, val, or test). It includes data augmentation for training data, such as random rotation.

Custom Dataset Class
The DOGGODataset class defines a custom PyTorch dataset for the Dog Breed Classifier. It handles the loading of images and labels, providing necessary transformations for training and evaluation.

DataPrefetcher Class
The DataPrefetcher class, although currently not implemented, is designed to prefetch data using CUDA streams for enhanced data loading efficiency.

## Model Training and Optimization
The train module includes scripts for training the ViT/CNN models (train.py) and optimizing hyperparameters (hyperparameter_optimization.py). The main script (main.py) integrates these modules for a complete training pipeline.

## TODO
- Finish implementing preprocessing.
- Hook model to production data pipeline.
- Train and publish model.
