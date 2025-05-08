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