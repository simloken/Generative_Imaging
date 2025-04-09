import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import shutil

class MNISTDataset(Dataset):
    """
    Custom PyTorch Dataset for MNIST data stored in CSV format.

    The first column is assumed to be the labels and the remaining columns are pixel values.
    Pixel values are scaled from the original 0-255 range. By default, images are scaled to [-1, 1]
    unless the `skip` flag is set (for models such as RBM).

    Args:
        csv_file (str): Path to the CSV file containing MNIST data.
        skip (bool): Flag indicating whether to skip scaling to [-1, 1]. Defaults to False.
    """
    def __init__(self, csv_file, skip):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values  # First column as labels.
        # Reshape images and scale pixel values to [0, 1].
        self.images = self.data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32) / 255.0
        if skip:  # For methods (e.g., RBM) that do not require scaling to [-1, 1].
            return
        self.images = 2 * self.images - 1  # Scale to [-1, 1].

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label) where image is a 28x28 numpy array and label is an integer.
        """
        return self.images[idx], self.labels[idx]

def get_data_make_dataset():
    # Transform to convert image to tensor
    transform = transforms.ToTensor()
    
    # Load training and test datasets
    train_set = datasets.MNIST(root='./../Data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./../Data', train=False, download=True, transform=transform)
    
    # Function to convert dataset to DataFrame
    def dataset_to_dataframe(dataset):
        data = []
        for image, label in dataset:
            # Flatten the image (1, 28, 28) to (784,)
            pixels = image.view(-1).numpy() * 255  # Convert from [0, 1] to [0, 255]
            row = [label] + pixels.astype(int).tolist()
            data.append(row)
        # First column is label, then 784 pixels
        columns = ['label'] + [f'pixel{i}' for i in range(784)]
        return pd.DataFrame(data, columns=columns)
    
    # Convert and save to CSV
    train_df = dataset_to_dataframe(train_set)
    test_df = dataset_to_dataframe(test_set)
    
    train_df.to_csv('./../Data/mnist_train.csv', index=False)
    test_df.to_csv('./../Data/mnist_test.csv', index=False)
    
    shutil.rmtree('./../Data/MNIST')