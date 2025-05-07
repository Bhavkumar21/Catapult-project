import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.linalg import svdvals

import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Now use a regular import
from data import generate_highrank_matrix, generate_lowrank_matrix

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

def compute_condition_number(singular_values):
    """Calculate the condition number (ratio of largest to smallest non-zero singular value)."""
    filtered_sv = singular_values[singular_values > 1e-12]
    if len(filtered_sv) == 0:
        return np.nan
    return np.max(filtered_sv) / np.min(filtered_sv)

def compute_singular_value_entropy(singular_values):
    """Calculate entropy of normalized singular value distribution."""
    sv_sum = np.sum(singular_values)
    if sv_sum == 0:
        return 0
    normalized_sv = singular_values / sv_sum
    return -np.sum(normalized_sv * np.log(normalized_sv + 1e-12))
    
def train_model_imbalanced(train_dataset, test_dataset, lr, epochs, weight_decay=0.0, imbalance_factor=10.0):
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    model = NN_TOY_Imbalanced(imbalance_factor=imbalance_factor)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Metrics storage
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'top_svs': np.zeros((epochs * len(train_loader), 20)),
        'W1WT_rank': [],
        'W2WT_rank': [],
        'frobenius_norms': [],
        'w1_norms': [],
        'w2_norms': [],
        'norm_ratios': [],
        'condition_numbers': [],
        'sv_entropies': [],
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
                W2 = model.fc2.weight.detach().cpu()
                
                W1WT = W1 @ W1.T
                metrics['W1WT_rank'].append(compute_ranks(W1WT))

                W2WT = W2 @ W2.T
                metrics['W2WT_rank'].append(compute_ranks(W2WT))
                
                U, s, V = torch.svd(W1)
                top_svs = s.numpy()

                cond_num = compute_condition_number(top_svs)
                sv_entropy = compute_singular_value_entropy(top_svs)
                metrics['condition_numbers'].append(cond_num)
                metrics['sv_entropies'].append(sv_entropy)
                
                w1_norm = torch.norm(W1).item()
                w2_norm = torch.norm(W2).item()
                norm_ratio = w1_norm / w2_norm
                
                metrics['w1_norms'].append(w1_norm)
                metrics['w2_norms'].append(w2_norm)
                metrics['norm_ratios'].append(norm_ratio)
                metrics['frobenius_norms'].append(w1_norm)  # For compatibility
                
                metrics['top_svs'][sv_counter] = top_svs[:20]
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

    print(f"Final Norm Ratio ||W1||/||W2|| = {metrics['norm_ratios'][-1]:.4f}")
    print("Finished Training...")
    return metrics

colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
def add_annotation_2(ax, exp_val, ctrl_val, exp_color, ctrl_color, fmt, wd=False):
    ax.text(0.95, 0.30, f'Large lr: {exp_val:{fmt}}', color=exp_color, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    if wd:
        ax.text(0.95, 0.21, f'Small lr w/ 0.1 WD: {ctrl_val:{fmt}}', color=ctrl_color, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    else:
        ax.text(0.95, 0.21, f'Small lr: {ctrl_val:{fmt}}', color=ctrl_color, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

def generate_imbalance_viz(exp, ctrl, name='imbalanced', wd=False):
    colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['train_loss'], color=colors['exp'], label='Larger Lr')
    ax1.plot(ctrl['train_loss'], color=colors['ctrl'], label='Small Lr')
    add_annotation_2(ax1, exp['train_loss'][-1], ctrl['train_loss'][-1], colors['exp'], colors['ctrl'], '.2e', wd)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('MSE Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.savefig(f"img/train-{name}.png")
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['test_loss'], color=colors['exp'], label='Larger Lr')
    ax1.plot(ctrl['test_loss'], color=colors['ctrl'], label='Small Lr')
    add_annotation_2(ax1, exp['test_loss'][-1], ctrl['test_loss'][-1], colors['exp'], colors['ctrl'], '.2e', wd)
    ax1.set_yscale('log') 
    ax1.set_title('Test Loss (log scaled)')
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('MSE Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.savefig(f"img/test-{name}.png")
    plt.tight_layout()
    plt.show()

    # Plot Condition Number
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['condition_numbers'], color=colors['exp'], label='Large LR')
    ax1.plot(ctrl['condition_numbers'], color=colors['ctrl'], label='Small LR')
    
    add_annotation_2(ax1, exp['condition_numbers'][-1], ctrl['condition_numbers'][-1], 
                    colors['exp'], colors['ctrl'], '.2f', wd)
    
    ax1.set_title('Condition Number of Weight Matrix')
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('Condition Number')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"img/condition_number-{name}.png")
    plt.tight_layout()
    plt.show()

    # Plot Singular Value Entropy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['sv_entropies'], color=colors['exp'], label='Large LR')
    ax1.plot(ctrl['sv_entropies'], color=colors['ctrl'], label='Small LR')
    
    add_annotation_2(ax1, exp['sv_entropies'][-1], ctrl['sv_entropies'][-1], 
                    colors['exp'], colors['ctrl'], '.4f', wd)
    
    ax1.set_title('Singular Value Entropy')
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('Entropy')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"img/sv_entropy-{name}.png")
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['norm_ratios'], color=colors['exp'], label='Large LR')
    ax1.plot(ctrl['norm_ratios'], color=colors['ctrl'], label='Small LR')
    
    ax1.set_title('Weight Norm Ratio (||W1||/||W2||)')
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('Norm Ratio')
    add_annotation_2(ax1, exp['norm_ratios'][-1], ctrl['norm_ratios'][-1], colors['exp'], colors['ctrl'], '.2e', wd)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"img/norm_ratio-{name}.png")
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

    plt.savefig(f"img/rank-{name}.png")
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['W2WT_rank'], color=colors['exp'], label='Larger Lr')
    ax1.plot(ctrl['W2WT_rank'], color=colors['ctrl'], label='Small Lr')
    add_annotation_2(ax1, exp['W2WT_rank'][-1], ctrl['W2WT_rank'][-1],colors['exp'], colors['ctrl'], '.2e')
    ax1.set_title('Effective Rank of Weight 2 Matrix')
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('Effective Rank')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.savefig(f"img/rank2-{name}.png")
    plt.tight_layout()
    plt.show()
    
    # NEW: Individual Layer Norms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # W1 norm
    ax1.plot(exp['w1_norms'], color=colors['exp'], label='Large LR')
    ax1.plot(ctrl['w1_norms'], color=colors['ctrl'], label='Small LR')
    ax1.set_title('Layer 1 Weight Norm (||W1||)')
    add_annotation_2(ax1, exp['w1_norms'][-1], ctrl['w1_norms'][-1], colors['exp'], colors['ctrl'], '.2e', wd)
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('Norm Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # W2 norm
    ax2.plot(exp['w2_norms'], color=colors['exp'], label='Large LR')
    ax2.plot(ctrl['w2_norms'], color=colors['ctrl'], label='Small LR')
    ax2.set_title('Layer 2 Weight Norm (||W2||)')
    add_annotation_2(ax2, exp['w2_norms'][-1], ctrl['w2_norms'][-1], colors['exp'], colors['ctrl'], '.2e', wd)
    ax2.set_xlabel('Training Iterations')
    ax2.set_ylabel('Norm Value')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(f"img/layer_norms-{name}.png")
    plt.tight_layout()
    plt.show()


def add_annotation_4(ax, exp_val, ctrl_val, ctrl1_val, ctrl2_val, exp_color, ctrl_color, fmt):
    ax.text(1.05, 0.30, f'Larger Lr: {exp_val:{fmt}}', color=exp_color, ha='left', va='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1.05, 0.23, f'Small Lr + WD (0.001): {ctrl_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1.05, 0.16, f'Small Lr + WD (0.01): {ctrl1_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1.05, 0.09, f'Small Lr + WD (0.1): {ctrl2_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

def generate_imbalance_viz2(exp, ctrl, ctrl1, ctrl2, name='imbalanced'):
    line_styles = ['-', '--', ':']
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['norm_ratios'], color=colors['exp'], label='Larger Lr')
    ax1.plot(ctrl['norm_ratios'], color=colors['ctrl'], label='Small Lr + WD (0.001)', linestyle=line_styles[0])
    ax1.plot(ctrl1['norm_ratios'], color=colors['ctrl'], label='Small Lr + WD (0.01)', linestyle=line_styles[1])
    ax1.plot(ctrl2['norm_ratios'], color=colors['ctrl'], label='Small Lr + WD (0.1)', linestyle=line_styles[2])

    add_annotation_4(ax1, exp['norm_ratios'][-1], ctrl['norm_ratios'][-1], ctrl1['norm_ratios'][-1], ctrl2['norm_ratios'][-1], colors['exp'], colors['ctrl'], '.2e')

    ax1.set_title('Weight Norm Ratio (||W1||/||W2||)')
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('Norm Ratio')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"img/norm_ratio-{name}.png")
    plt.show()

def main():
    # Set random seed for reproducibility
    SEED = 1
    TYPE = 'ls'
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    imbalance_factor = 10.0
    for i in ['ls', 'ld', 'hs', 'hd']:
        TYPE = i
        print(f"Generating Dataset {i}...")
        train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=TYPE)
        
        print("Training with Large Learning Rate (Catapult Regime)...")
        exp = train_model_imbalanced(
            train_dataset, test_dataset, 
            lr=0.002,
            epochs=500, 
            imbalance_factor=imbalance_factor
        )
        
        # print("Training with Small Learning Rate (Control)...")
        # ctrl = train_model_imbalanced(
        #     train_dataset, test_dataset, 
        #     lr=0.0001,
        #     epochs=500,
        #     imbalance_factor=imbalance_factor
        # )
        
        # print("Generating Visualizations...")
        # generate_imbalance_viz(exp, ctrl, name=f'imbalance_{TYPE}')
        
        print("Training with Small Learning w 0.001 wd Rate (Control)...")
        ctrl = train_model_imbalanced(
            train_dataset, test_dataset, 
            lr=0.0001,
            epochs=500,
            weight_decay=0.001,
            imbalance_factor=imbalance_factor
        )

        print("Training with Small Learning w 0.01 wd Rate (Control)...")

        ctrl1 = train_model_imbalanced(
            train_dataset, test_dataset, 
            lr=0.0001,
            epochs=500,
            weight_decay=0.01,
            imbalance_factor=imbalance_factor
        )

        print("Training with Small Learning w 0.1 wd Rate (Control)...")

        ctrl2 = train_model_imbalanced(
            train_dataset, test_dataset, 
            lr=0.0001,
            epochs=500,
            weight_decay=0.1,
            imbalance_factor=imbalance_factor
        )
        
        print("Generating Visualizations...")
        generate_imbalance_viz2(exp, ctrl, ctrl1, ctrl2, name=f'imbalance_{TYPE}-wd')
        

if __name__ == "__main__":
    main()