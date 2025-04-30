import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import compute_ranks

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

np.random.seed(23)
torch.manual_seed(23)

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

def train_model(train_dataset, test_dataset, lr, epochs, weight_decay=0.00):
  train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

  model = NN_TOY()
  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

  # Metrics storage
  metrics = {
      'train_loss': [],
      'test_loss': [],
      'top_svs': np.zeros((epochs * len(train_loader), 20)),
      'W1WT_rank':[],
      'frobenius_norms': [],
  }

  sv_counter = 0

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
              W1WT = W1 @ W1.T
              metrics['W1WT_rank'].append(compute_ranks(W1WT))
              
              U, s, V = torch.svd(W1)
              
              top_svs = s.numpy()
              
              frobenius_norm = torch.sqrt(torch.sum(s**2)).item()
              metrics['frobenius_norms'].append(frobenius_norm)
              
              metrics['top_svs'][sv_counter] = top_svs              
              sv_counter += 1

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

class NN_MNIST(nn.Module):
    """
    3-hidden layer fully-connected network with ReLU non-linearity for MNIST
    classification, as used in Lewkowycz et al. 2020.
    """
    def __init__(self, input_dim=784, hidden_dim=9408, num_classes=10):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act2 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, num_classes, bias=True)
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc4(x)    
        return x


def train_model_mnist(train_dataset, test_dataset, lr, epochs, weight_decay=0.00):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

    model = NN_MNIST().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    one_hot_lookup = torch.eye(10).to(DEVICE)
    
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'W1WT_rank': [],
        'W2WT_rank': [],
        # 'W3WT_rank': [],
        'frobenius_norm1': [],
        'frobenius_norm2': [],
        # 'frobenius_norm3': [],
        'accuracy': [],
    }
    
    # Track metrics less frequently to improve speed
    metric_tracking_interval = 5
    
    print(f"Training on {DEVICE}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device efficiently
            images = images.view(-1, 784).to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            # Use pre-computed one-hot encoding
            labels_onehot = one_hot_lookup[labels]
            
            outputs = model(images)
            loss = criterion(outputs, labels_onehot)

            optimizer.zero_grad(set_to_none=True)  # More efficient than just zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_size = images.size(0)
            train_loss += loss.item() * current_batch_size
            train_samples += current_batch_size
            
            # Only track metrics occasionally to reduce overhead
            if batch_idx % metric_tracking_interval == 0:
                with torch.no_grad():
                    # Move to CPU for analysis, detach to avoid tracking history
                    W1 = model.fc1.weight.detach().cpu()
                    W1WT = W1 @ W1.T
                    metrics['W1WT_rank'].append(compute_ranks(W1WT))
                    metrics['frobenius_norm1'].append(torch.norm(W1, p='fro').item())
                    
                    W2 = model.fc2.weight.detach().cpu()
                    W2WT = W2 @ W2.T
                    metrics['W2WT_rank'].append(compute_ranks(W2WT))
                    metrics['frobenius_norm2'].append(torch.norm(W2, p='fro').item())
                    
                    # W3 = model.fc3.weight.detach().cpu()
                    # W3WT = W3 @ W3.T
                    # metrics['W3WT_rank'].append(compute_ranks(W3WT))
                    # metrics['frobenius_norm3'].append(torch.norm(W3, p='fro').item())

        metrics['train_loss'].append(train_loss / train_samples)

        model.eval()
        test_loss = 0.0
        test_samples = 0
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, 784).to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                labels_onehot = one_hot_lookup[labels]
                
                outputs = model(images)
                loss = criterion(outputs, labels_onehot)

                batch_size = images.size(0)
                test_loss += loss.item() * batch_size
                test_samples += batch_size

                _, predicted = torch.max(outputs, 1)
                # Now both tensors are on the same device
                correct += (predicted == labels).sum().item()

        metrics['test_loss'].append(test_loss / test_samples)
        accuracy = correct / test_samples
        metrics['accuracy'].append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {metrics['train_loss'][-1]:.6f}, "
              f"Test Loss: {metrics['test_loss'][-1]:.6f}, Accuracy: {accuracy:.4f}")

    print("Finished Training MNIST Model...")
    return metrics


class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=304, kernel_size=3, padding='same', bias=False)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 304, 10, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_model_cifar(train_dataset, test_dataset, lr, epochs, weight_decay=0.00):

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

    model = CNN_CIFAR10().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    one_hot_lookup = torch.eye(10).to(DEVICE)
    
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'WWT_fc_rank': [],
        'WWT_conv_rank': [],
        'frobenius_conv_norm': [],
        'frobenius_fc_norm': [],
        'accuracy': [],
    }
    
    metric_tracking_interval = 5
    
    print(f"Training on {DEVICE}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            labels_onehot = one_hot_lookup[labels]
            
            outputs = model(images)
            loss = criterion(outputs, labels_onehot)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            current_batch_size = images.size(0)
            train_loss += loss.item() * current_batch_size
            train_samples += current_batch_size
            
            # Track metrics occasionally to reduce overhead
            if batch_idx % metric_tracking_interval == 0:
                with torch.no_grad():
                    # Compute WWT for fc layer
                    W_fc = model.fc.weight.detach().cpu()
                    WWT_fc = W_fc @ W_fc.T
                    metrics['WWT_fc_rank'].append(compute_ranks(WWT_fc))
                    metrics['frobenius_fc_norm'].append(torch.norm(W_fc, p='fro').item())
                    
                    # Compute WWT for conv layer
                    W_conv = model.conv1.weight.detach().cpu()
                    WWT_conv = W_conv.flatten(1) @ W_conv.flatten(1).T
                    metrics['WWT_conv_rank'].append(compute_ranks(WWT_conv))
                    metrics['frobenius_conv_norm'].append(torch.norm(W_conv, p='fro').item())

        metrics['train_loss'].append(train_loss / train_samples)

        model.eval()
        test_loss = 0.0
        test_samples = 0
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                labels_onehot = one_hot_lookup[labels]
                
                outputs = model(images)
                loss = criterion(outputs, labels_onehot)

                batch_size = images.size(0)
                test_loss += loss.item() * batch_size
                test_samples += batch_size

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        metrics['test_loss'].append(test_loss / test_samples)
        accuracy = correct / test_samples
        metrics['accuracy'].append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {metrics['train_loss'][-1]:.6f}, "
              f"Test Loss: {metrics['test_loss'][-1]:.6f}, Accuracy: {accuracy:.4f}")

    print("Finished Training CIFAR Model...")
    return metrics
