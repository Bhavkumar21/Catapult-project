import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
from utils import generate_highrank_matrix, generate_lowrank_matrix

np.random.seed(23)
torch.manual_seed(23)

class CustomDataset(Dataset):
    """Custom dataset for handling the neural network input/output pairs."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class CustomDatasetClassifier(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_relu_quadratic(X, A):
    return np.array([x.T @ A @ x for x in X])


def create_A(input_dim, type):
  if type == 'hd':
      A = generate_highrank_matrix(dim=input_dim, target_condition=1, sparsity=0.01)
  elif type == 'hs':
      A = generate_highrank_matrix(dim=input_dim, target_condition=1, sparsity=0.9)
      
  elif type == 'ld':
      A = generate_lowrank_matrix(input_dim, 3, 0.01)
  else:
      A = np.zeros((input_dim, input_dim), dtype=int)
      A[0, 0] = 1
      A[1, 1] = 1
  return A

def generate_data(num_samples, input_dim, type='ls'):
    A = create_A(input_dim, type)
    
    X_data = np.random.randn(num_samples, input_dim)
    y_data = compute_relu_quadratic(X_data, A)

    train_size = int(num_samples * 0.8)

    X_train, X_test = X_data[:train_size], X_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]

    # # Normalize data
    # X_mean = np.mean(X_train, axis=0)
    # X_std = np.std(X_train, axis=0) + 1e-8
    # X_train = (X_train - X_mean) / X_std
    # X_test = (X_test - X_mean) / X_std
    
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    print("Finished Generating Dataset...")

    return train_dataset, test_dataset

def generate_data_mnist(num_samples):
    mnist_train = datasets.MNIST(root='./mnist', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='./mnist', train=False, download=True, transform=None)
    
    train_size = int(num_samples * 0.8)
    test_size = num_samples - train_size
    
    # Convert to numpy arrays and float type
    X_train = np.array(mnist_train.data[:train_size], dtype=np.float32)
    y_train = np.array(mnist_train.targets[:train_size])
    X_test = np.array(mnist_test.data[:test_size], dtype=np.float32)
    y_test = np.array(mnist_test.targets[:test_size])
    
    # Scale pixel values to [0, 1]
    X_train /= 255.0
    X_test /= 255.0
    
    # Calculate mean and std from training data
    train_pixels = X_train.reshape(-1)
    mean = np.mean(train_pixels)
    std = np.std(train_pixels)
    
    print(f"MNIST Mean: {mean}")
    print(f"MNIST Std: {std}")
    
    # Reshape to (N, C, H, W) format, MNIST is grayscale so C=1
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    # Apply normalization
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    train_dataset = CustomDatasetClassifier(X_train, y_train)
    test_dataset = CustomDatasetClassifier(X_test, y_test)
    
    print("Finished Generating Normalized MNIST Dataset...")
    return train_dataset, test_dataset



def generate_data_cifar(num_samples):
    cifar_train = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=None)
    cifar_test = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=None)
    
    train_size = int(num_samples * 0.8)
    test_size = num_samples - train_size
    
    # Convert to numpy arrays and float type
    X_train = np.array(cifar_train.data[:train_size], dtype=np.float32)
    y_train = np.array(cifar_train.targets[:train_size])
    X_test = np.array(cifar_test.data[:test_size], dtype=np.float32)
    y_test = np.array(cifar_test.targets[:test_size])
    
    # Scale pixel values to [0, 1]
    X_train /= 255.0
    X_test /= 255.0
    
    # Calculate mean and std per channel from training data
    # Reshape to get all pixels for each channel
    train_pixels = X_train.reshape(-1, 3)
    mean = np.mean(train_pixels, axis=0)
    std = np.std(train_pixels, axis=0)
    
    print(f"CIFAR-10 Channel Mean: {mean}")
    print(f"CIFAR-10 Channel Std: {std}")
    
    # Transpose to (N, C, H, W) format
    X_train = X_train.transpose((0, 3, 1, 2))
    X_test = X_test.transpose((0, 3, 1, 2))
    
    # Apply normalization to each channel
    for i in range(3):
        X_train[:, i] = (X_train[:, i] - mean[i]) / std[i]
        X_test[:, i] = (X_test[:, i] - mean[i]) / std[i]
    
    train_dataset = CustomDatasetClassifier(X_train, y_train)
    test_dataset = CustomDatasetClassifier(X_test, y_test)
    
    print("Finished Generating Normalized CIFAR-10 Dataset...")
    return train_dataset, test_dataset
