import torch
from torch.utils.data import Dataset
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def compute_sigmoid_quadratic(X, A):
    quadratic_values = np.array([x.T @ A @ x for x in X])
    return sigmoid(quadratic_values)

def create_matrix_A(input_dim, rank_type):
    """Create matrix A with specified rank"""
    if rank_type == 'low':
        # Low rank matrix (rank 2)
        U = np.random.randn(input_dim, 2)
        V = np.random.randn(2, input_dim)
        A = U @ V
    elif rank_type == 'medium':
        # Medium rank matrix (rank 10)
        U = np.random.randn(input_dim, 10)
        V = np.random.randn(10, input_dim)
        A = U @ V
    elif rank_type == 'high':
        # High rank matrix (rank 20)
        U = np.random.randn(input_dim, 20)
        V = np.random.randn(20, input_dim)
        A = U @ V
    else:
        raise ValueError("rank_type must be 'low', 'medium', or 'high'")
    
    return A

def generate_data(num_samples, input_dim, rank_type='low'):
    A = create_matrix_A(input_dim, rank_type)
    X_data = np.random.randn(num_samples, input_dim)
    y_data = compute_sigmoid_quadratic(X_data, A)
    
    train_size = int(num_samples * 0.8)
    X_train, X_test = X_data[:train_size], X_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    
    # Normalize targets
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    
    print(f"Generated {rank_type} rank dataset with {num_samples} samples")
    return train_dataset, test_dataset
