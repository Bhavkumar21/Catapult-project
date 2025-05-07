import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.linalg import svdvals
import sys
from pathlib import Path

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

class ActivationSparsityHook:
    """Hook to capture activation sparsity."""
    def __init__(self):
        self.sparsity = []
        self.activations = None
    
    def hook_fn(self, module, input, output):
        # Calculate sparsity (percentage of zeros)
        zeros = (output == 0).float().mean().item()
        self.sparsity.append(zeros)
        self.activations = output.detach().clone()

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

colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
def add_annotation_2(ax, exp_val, ctrl_val, exp_color, ctrl_color, fmt, wd=False):
    ax.text(0.95, 0.30, f'Large lr: {exp_val:{fmt}}', color=exp_color, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    if wd:
        ax.text(0.95, 0.21, f'Small lr w/ 0.1 WD: {ctrl_val:{fmt}}', color=ctrl_color, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    else:
        ax.text(0.95, 0.21, f'Small lr: {ctrl_val:{fmt}}', color=ctrl_color, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

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
    """Two-layer neural network with imbalanced weight initialization and activation tracking."""
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=512, imbalance_factor=10.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        
        # Initialize with imbalanced weights
        with torch.no_grad():
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            
            # Scale up first layer weights to create imbalance
            self.fc1.weight.data *= imbalance_factor
            
            w1_norm = torch.norm(self.fc1.weight).item()
            w2_norm = torch.norm(self.fc2.weight).item()
            print(f"Initial weight norm ratio ||W1||/||W2|| = {w1_norm/w2_norm:.4f}")
        
        # Create hook for measuring activation sparsity
        self.activation_hook = ActivationSparsityHook()
        self.act1.register_forward_hook(self.activation_hook.hook_fn)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x
    
    def get_latest_sparsity(self):
        """Return the most recent activation sparsity value."""
        if len(self.activation_hook.sparsity) > 0:
            return self.activation_hook.sparsity[-1]
        return 0.0
    
    def reset_sparsity_history(self):
        """Clear sparsity history."""
        self.activation_hook.sparsity = []

def train_model_imbalanced(train_dataset, test_dataset, lr, epochs, weight_decay=0.0, imbalance_factor=10.0):
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    model = NN_TOY_Imbalanced(imbalance_factor=imbalance_factor)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Add activation sparsity to metrics
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
        'activation_sparsity': [], # Add sparsity tracking
        'batch_activation_sparsity': [],  # Track per batch for visualization
    }
    
    sv_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        epoch_sparsity = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(-1, INPUT_DIM)
            
            # Forward pass will trigger the activation hook
            outputs = model(images)
            
            # Record batch sparsity
            batch_sparsity = model.get_latest_sparsity()
            metrics['batch_activation_sparsity'].append(batch_sparsity)
            epoch_sparsity.append(batch_sparsity)
            
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_batch_size = images.shape[0]
            train_loss += loss.item() * current_batch_size
            train_samples += current_batch_size
            
            with torch.no_grad():
                # Existing weight metrics collection
                W1 = model.fc1.weight.detach().cpu()
                W2 = model.fc2.weight.detach().cpu()
                
                # W1WT = W1 @ W1.T
                # metrics['W1WT_rank'].append(compute_ranks(W1WT))
                
                # W2WT = W2 @ W2.T
                # metrics['W2WT_rank'].append(compute_ranks(W2WT))
                
                U, s, V = torch.svd(W1)
                top_svs = s.numpy()
                
                # cond_num = compute_condition_number(top_svs)
                # sv_entropy = compute_singular_value_entropy(top_svs)
                # metrics['condition_numbers'].append(cond_num)
                # metrics['sv_entropies'].append(sv_entropy)
                
                w1_norm = torch.norm(W1).item()
                w2_norm = torch.norm(W2).item()
                norm_ratio = w1_norm / w2_norm
                
                metrics['w1_norms'].append(w1_norm)
                metrics['w2_norms'].append(w2_norm)
                metrics['norm_ratios'].append(norm_ratio)
                metrics['frobenius_norms'].append(w1_norm)
                
                metrics['top_svs'][sv_counter] = top_svs[:20]
                sv_counter += 1
        
        # Record average sparsity for the epoch
        avg_epoch_sparsity = np.mean(epoch_sparsity)
        metrics['activation_sparsity'].append(avg_epoch_sparsity)
        
        # Existing epoch-level logic
        metrics['train_loss'].append(train_loss/train_samples)
        
        # Test evaluation
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
        
        # Calculate R² and print progress
        all_labels = torch.cat(all_labels, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        
        mean_label = torch.mean(all_labels)
        tss = torch.sum((all_labels - mean_label) ** 2).item()
        mse = metrics['test_loss'][-1]
        n = len(test_dataset)
        r_squared = max(0, 1 - (mse * n) / tss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Test Loss: {mse:.6f}, R²: {r_squared:.6f}, Sparsity: {avg_epoch_sparsity:.4f}")
    
    print(f"Final Test Loss: {mse:.6f}, R²: {r_squared:.6f}")
    print(f"Final Norm Ratio ||W1||/||W2|| = {metrics['norm_ratios'][-1]:.4f}")
    print(f"Final Activation Sparsity: {metrics['activation_sparsity'][-1]:.4f} (fraction of zeros)")
    
    return metrics

# Add function to plot activation sparsity
def plot_activation_sparsity(exp, ctrl, name='sparsity', wd=False):
    colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
    
    # Plot epoch-level sparsity
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['activation_sparsity'], color=colors['exp'], label='Large LR')
    ax1.plot(ctrl['activation_sparsity'], color=colors['ctrl'], label='Small LR')
    
    add_annotation_2(ax1, exp['activation_sparsity'][-1], ctrl['activation_sparsity'][-1], 
                    colors['exp'], colors['ctrl'], '.4f', wd)
    
    ax1.set_title('Activation Sparsity (Fraction of Zeros)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Sparsity')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"img/activation_sparsity-{name}.png")
    plt.tight_layout()
    plt.show()
    
    # Plot batch-level sparsity
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(exp['batch_activation_sparsity'], color=colors['exp'], alpha=0.5)
    ax1.plot(ctrl['batch_activation_sparsity'], color=colors['ctrl'], alpha=0.5)
    
    # Add smoothed lines
    window = min(50, len(exp['batch_activation_sparsity'])//10)
    if window > 0:
        smooth_exp = np.convolve(exp['batch_activation_sparsity'], np.ones(window)/window, mode='valid')
        smooth_ctrl = np.convolve(ctrl['batch_activation_sparsity'], np.ones(window)/window, mode='valid')
        offset = window//2
        ax1.plot(range(offset, offset+len(smooth_exp)), smooth_exp, color=colors['exp'], linewidth=2, label='Large LR')
        ax1.plot(range(offset, offset+len(smooth_ctrl)), smooth_ctrl, color=colors['ctrl'], linewidth=2, label='Small LR')
    
    ax1.set_title('Batch-level Activation Sparsity')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Sparsity')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"img/batch_activation_sparsity-{name}.png")
    plt.tight_layout()
    plt.show()
    
    # Plot relationship between sparsity and layer balance
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(exp['norm_ratios'], exp['batch_activation_sparsity'], 
                color=colors['exp'], alpha=0.5, label='Large LR')
    ax1.scatter(ctrl['norm_ratios'], ctrl['batch_activation_sparsity'], 
                color=colors['ctrl'], alpha=0.5, label='Small LR')
    
    ax1.set_title('Relationship Between Weight Balance and Activation Sparsity')
    ax1.set_xlabel('Weight Norm Ratio (||W1||/||W2||)')
    ax1.set_ylabel('Activation Sparsity')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"img/sparsity_vs_balance-{name}.png")
    plt.tight_layout()
    plt.show()
    
    # Plot relationship between sparsity and rank
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(exp['W1WT_rank'], exp['batch_activation_sparsity'], 
                color=colors['exp'], alpha=0.5, label='Large LR')
    ax1.scatter(ctrl['W1WT_rank'], ctrl['batch_activation_sparsity'], 
                color=colors['ctrl'], alpha=0.5, label='Small LR')
    
    ax1.set_title('Relationship Between Weight Rank and Activation Sparsity')
    ax1.set_xlabel('Effective Rank of W1')
    ax1.set_ylabel('Activation Sparsity')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"img/sparsity_vs_rank-{name}.png")
    plt.tight_layout()
    plt.show()

# Modify main function to include sparsity analysis
def main():
    # Set random seed for reproducibility
    SEED = 1
    TYPE = 'ls'
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    imbalance_factor = 10
    
    for i in ['ls', 'ld', 'hs', 'hd']:
        TYPE = i
        print(f"Generating Dataset {i}...")
        train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=TYPE)
        
        print("Training with Large Learning Rate (Catapult Regime)...")
        exp = train_model_imbalanced(
            train_dataset, test_dataset,
            lr=0.002,  # Large learning rate
            epochs=500,
            imbalance_factor=imbalance_factor
        )
        
        print("Training with Small Learning Rate (Control)...")
        ctrl = train_model_imbalanced(
            train_dataset, test_dataset,
            lr=0.0001,  # Small learning rate 
            epochs=500,
            imbalance_factor=imbalance_factor
        )
        
        print("Generating Visualizations...")
        plot_activation_sparsity(exp, ctrl, name=f'sparse-thingy')
        
        # print("Training with Small Learning Rate + Weight Decay...")
        # ctrl_wd = train_model_imbalanced(
        #     train_dataset, test_dataset,
        #     lr=0.0001,
        #     epochs=500,
        #     weight_decay=0.01,
        #     imbalance_factor=imbalance_factor
        # )
        
        # print("Generating Visualizations with Weight Decay...")
        # plot_activation_sparsity(exp, ctrl_wd, name=f'imbalance_{TYPE}-wd', wd=True)

if __name__ == "__main__":
    main()
