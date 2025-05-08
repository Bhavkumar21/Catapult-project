import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
from utils import generate_highrank_matrix, generate_lowrank_matrix

np.random.seed(24)
torch.manual_seed(24)

class CustomDataset(Dataset):
    """Custom dataset for handling the neural network input/output pairs."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
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
      A = generate_lowrank_matrix(input_dim, 3)
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

