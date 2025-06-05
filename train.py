import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import compute_ranks, singular_value_entropy

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

np.random.seed(24)
torch.manual_seed(24)

INPUT_DIM = 20

class NN_TOY(nn.Module):
  """Simple two-layer neural network with ReLU activation."""
  def __init__(self, input_dim=INPUT_DIM, hidden_dim=256):
      super().__init__()
      self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
      self.act1 = nn.ReLU() 
      self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
      
  def forward(self, x):
      x = self.fc1(x)
      x = self.act1(x)
      x = self.fc2(x)
      return x

class NN_TOY_Imbalanced(nn.Module):
    """Two-layer neural network with imbalanced weight initialization."""
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=512, imbalance_factor=10.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        
        # Initialize with imbalanced weights
        with torch.no_grad():
            # Standard initialization
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            
            # Scale up first layer weights to create imbalance
            self.fc1.weight.data *= imbalance_factor
            
            w1_norm = torch.norm(self.fc1.weight).item()
            w2_norm = torch.norm(self.fc2.weight).item()
            print(f"Initial weight norm ratio ||W1||/||W2|| = {w1_norm/w2_norm:.4f}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def train_model(train_dataset, test_dataset, lr, epochs, weight_decay=0.00, imb = False):
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    quad_matrix = torch.zeros((INPUT_DIM, INPUT_DIM))
    for i in range(len(X_train)):
        x = X_train[i]
        quad_matrix += torch.abs(y_train[i]) * torch.outer(x, x)

    quad_matrix /= len(X_train)
    eigenvalues, eigenvectors = torch.linalg.eigh(quad_matrix)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    e_1 = eigenvectors[:, 0]
    e_1 = e_1 / torch.norm(e_1)

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    if imb:
        model = NN_TOY_Imbalanced()
    else:
        model = NN_TOY()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Metrics storage
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'W1WT_rank':[],
        'frobenius_norms': [],
        'similarity_e1_v1':[],
        'norm_ratios': [],
        'svs':[],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(-1, INPUT_DIM)
            outputs = model(images)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_size = images.shape[0]
            train_loss += loss.item() * current_batch_size
            train_samples += current_batch_size
            
            with torch.no_grad():
                W1 = model.fc1.weight.detach().cpu()
                U, s, V = torch.svd(W1)
                sv_entropy = singular_value_entropy(s)
                metrics['svs'].append(sv_entropy)
                W1WT = W1 @ W1.T
                metrics['W1WT_rank'].append(compute_ranks(W1WT))

                W2 = model.fc2.weight.detach().cpu()
                w1_norm = torch.norm(W1).item()
                w2_norm = torch.norm(W2).item()
                if w2_norm == 0:
                    norm_ratio = 0
                else:
                    norm_ratio = w1_norm / w2_norm
                metrics['norm_ratios'].append(norm_ratio)

                eigenvalues, eigenvectors = torch.linalg.eigh(W1WT)
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                v_1 = eigenvectors[:, 0]
                projected_e_1 = W1 @ e_1
                projected_e_1 = projected_e_1 / torch.norm(projected_e_1)
                v_1 = v_1 / torch.norm(v_1)
                
                # Calculate cosine similarity
                cosine_sim = torch.nn.functional.cosine_similarity(
                    projected_e_1.unsqueeze(0), 
                    v_1.unsqueeze(0)
                ).item()
        
                # Store the similarity
                metrics['similarity_e1_v1'].append(abs(cosine_sim))
                
                U, s, V = torch.svd(W1)
                
                frobenius_norm = torch.sqrt(torch.sum(s**2)).item()
                metrics['frobenius_norms'].append(frobenius_norm)

        metrics['train_loss'].append(train_loss/train_samples)
        model.eval()
        test_loss = 0.0
        test_samples = 0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, INPUT_DIM)
                outputs = model(images)
                labels = labels.view(-1, 1)
                all_labels.append(labels)
                all_outputs.append(outputs)

                batch_size = images.shape[0]
                batch_loss = criterion(outputs, labels).item() * batch_size
                test_loss += batch_loss
                test_samples += batch_size
            
            metrics['test_loss'].append(test_loss / test_samples)
        all_labels = torch.cat(all_labels, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)

        # Calculate R²
        mean_label = torch.mean(all_labels)
        tss = torch.sum((all_labels - mean_label) ** 2).item()
        mse = metrics['test_loss'][-1]
        n = len(test_dataset)
        r_squared = max(0, 1 - (mse * n) / tss)
        print(f"Final Test Loss: {mse:.6f}, R²: {r_squared:.6f}")

    print("Finished Training...")
    return metrics


def train_model_exp(train_dataset, test_dataset, lr, epochs, weight_decay=0.00):
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    quad_matrix = torch.zeros((INPUT_DIM, INPUT_DIM))
    for i in range(len(X_train)):
        x = X_train[i]
        quad_matrix += torch.abs(y_train[i]) * torch.outer(x, x)

    quad_matrix /= len(X_train)
    eigenvalues, eigenvectors = torch.linalg.eigh(quad_matrix)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    e_1 = eigenvectors[:, 0]
    e_1 = e_1 / torch.norm(e_1)

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


    model = NN_TOY()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    global_iteration = 0

    # Metrics storage
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'W1WT_rank':[],
        'frobenius_norms': [],
        'similarity_e1_v1':[],
        'norm_ratios': [],
        'svs':[],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if global_iteration == 15:
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = 0.0
                print(f"Turned off weight decay at iteration {global_iteration}")
            images = images.view(-1, INPUT_DIM)
            outputs = model(images)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_iteration += 1

            current_batch_size = images.shape[0]
            train_loss += loss.item() * current_batch_size
            train_samples += current_batch_size
            
            with torch.no_grad():
                W1 = model.fc1.weight.detach().cpu()
                U, s, V = torch.svd(W1)
                sv_entropy = singular_value_entropy(s)
                metrics['svs'].append(sv_entropy)
                W1WT = W1 @ W1.T
                metrics['W1WT_rank'].append(compute_ranks(W1WT))

                W2 = model.fc2.weight.detach().cpu()
                w1_norm = torch.norm(W1).item()
                w2_norm = torch.norm(W2).item()
                if w2_norm == 0:
                    norm_ratio = 0
                else:
                    norm_ratio = w1_norm / w2_norm
                metrics['norm_ratios'].append(norm_ratio)

                eigenvalues, eigenvectors = torch.linalg.eigh(W1WT)
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                v_1 = eigenvectors[:, 0]
                projected_e_1 = W1 @ e_1
                projected_e_1 = projected_e_1 / torch.norm(projected_e_1)
                v_1 = v_1 / torch.norm(v_1)
                
                # Calculate cosine similarity
                cosine_sim = torch.nn.functional.cosine_similarity(
                    projected_e_1.unsqueeze(0), 
                    v_1.unsqueeze(0)
                ).item()
        
                # Store the similarity
                metrics['similarity_e1_v1'].append(abs(cosine_sim))
                
                U, s, V = torch.svd(W1)
                
                frobenius_norm = torch.sqrt(torch.sum(s**2)).item()
                metrics['frobenius_norms'].append(frobenius_norm)

        metrics['train_loss'].append(train_loss/train_samples)
        model.eval()
        test_loss = 0.0
        test_samples = 0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, INPUT_DIM)
                outputs = model(images)
                labels = labels.view(-1, 1)
                all_labels.append(labels)
                all_outputs.append(outputs)

                batch_size = images.shape[0]
                batch_loss = criterion(outputs, labels).item() * batch_size
                test_loss += batch_loss
                test_samples += batch_size
            
            metrics['test_loss'].append(test_loss / test_samples)
        all_labels = torch.cat(all_labels, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)

        # Calculate R²
        mean_label = torch.mean(all_labels)
        tss = torch.sum((all_labels - mean_label) ** 2).item()
        mse = metrics['test_loss'][-1]
        n = len(test_dataset)
        r_squared = max(0, 1 - (mse * n) / tss)
        print(f"Final Test Loss: {mse:.6f}, R²: {r_squared:.6f}")

    print("Finished Training...")
    return metrics