import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.linalg import svdvals


INPUT_DIM = 20
NUM_SAMPLES = 500

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

def compute_ranks(matrix):
  """Compute effective rank of a matrix using nuclear norm / operator norm."""
  if not isinstance(matrix, np.ndarray):
      matrix = np.array(matrix)
  if np.any(~np.isfinite(matrix)):
      return np.nan
  try:
      s = svdvals(matrix)
  except np.linalg.LinAlgError:
      return np.nan
  
  s = np.abs(s)
  if len(s) == 0 or np.max(s) <= 1e-12:
      return 0.0
  
  nuclear_norm = np.sum(s)
  operator_norm = np.max(s)
  return nuclear_norm / operator_norm

def generate_data(num_samples, input_dim, type='ls'):
    A = np.zeros((input_dim, input_dim), dtype=int)
    A[0, 0] = 1
    A[1, 1] = 1
    
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

colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
def add_annotation_2(ax, exp_val, ctrl_val, exp_color, ctrl_color, fmt):
  ax.text(0.95, 0.30, f'Large lr: {exp_val:{fmt}}', color=exp_color, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

  ax.text(0.95, 0.21, f'Small lr: {ctrl_val:{fmt}}', color=ctrl_color, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

def generate_viz_2(exp, ctrl, name='ls'):
  colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['train_loss'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['train_loss'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['train_loss'][-1], ctrl['train_loss'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Training Loss')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('MSE Loss')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"train-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['test_loss'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['test_loss'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['test_loss'][-1], ctrl['test_loss'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.set_yscale('log') 
  ax1.set_title('Test Loss (log scaled)')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('MSE Loss')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"test-{name}.png")
  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(15, 8))
  gs = GridSpec(1, 1)
  lw = 2.5
  large_lr_colors = ['#08519c', '#3182bd', '#6baed6']  # More saturated blues
  small_lr_colors = ['#a63603', '#e6550d', '#fd8d3c']  # More saturated oranges
  line_styles = ['-', '--', ':']
  ax = plt.subplot(gs[0, 0])
  for i in range(3):
      ax.plot(exp['top_svs'][:40, i], 
              color=large_lr_colors[i%3],
              linestyle=line_styles[i%3],
              linewidth=lw,
              alpha=0.9,
              label=f'large lr σ_{i+1}')
      
  for i in range(3):
      ax.plot(ctrl['top_svs'][:, i], 
              color=small_lr_colors[i],
              linestyle=line_styles[i],
              linewidth=lw,
              alpha=1.0,
              label=f'small lr σ_{i+1}')
      
  for i in range(20):
     print('top svs', i, exp['top_svs'][:40, i])
  
  ax.set_title('Largest Singular Values of W1', fontsize=16)
  ax.set_ylabel('Singular Values (W1)', fontsize=14)
  ax.set_xlabel('Training Iterations', fontsize=14)
  # ax.legend(loc='upper right')
  ax.grid(True, alpha=0.3) 

  plt.savefig(f"svs-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['frobenius_norms'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['frobenius_norms'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['frobenius_norms'][-1], ctrl['frobenius_norms'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Frobenius Norm of Weight 1 Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Frobenius Norm Value')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"norm-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['W1WT_rank'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['W1WT_rank'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['W1WT_rank'][-1], ctrl['W1WT_rank'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Effective Rank of Weight 1 Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Effective Rank')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"rank-{name}.png")
  plt.tight_layout()
  plt.show()

  print("Finished Generating Visaulizations...")

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

def main():

  for i in range(23, 28):
    np.random.seed(i)
    torch.manual_seed(i)
    print("Generating Dataset...")
    train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type='ls')

    print("Training Catapult Model...")
    exp = train_model(train_dataset, test_dataset, 0.1, 500, weight_decay=0.00)

    print("Training Control Model...")
    ctrl = train_model(train_dataset, test_dataset, 0.001, 500, weight_decay=0.01)

    print("Generating Visaulizations...")
    generate_viz_2(exp, ctrl, name=f'{i}-randseed')

main()