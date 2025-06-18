import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def create_output_dir():
    """Create output directory for plots"""
    os.makedirs('results', exist_ok=True)

def plot_metrics(exp_metrics, ctrl_metrics, name, num_layers):
    """Plot all four metrics"""
    colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
    
    # Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(exp_metrics['train_loss'], color=colors['exp'], label='Large LR', linewidth=2)
    plt.plot(ctrl_metrics['train_loss'], color=colors['ctrl'], label='Small LR + WD Exp', linewidth=2)
    plt.title(f'{num_layers}-Layer NN: Training Loss ({name} rank)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/train_loss_{num_layers}layer_{name}.png', dpi=150)
    plt.close()
    
    # Test Loss
    plt.figure(figsize=(10, 6))
    plt.plot(exp_metrics['test_loss'], color=colors['exp'], label='Large LR', linewidth=2)
    plt.plot(ctrl_metrics['test_loss'], color=colors['ctrl'], label='Small LR + WD Exp', linewidth=2)
    plt.title(f'{num_layers}-Layer NN: Test Loss ({name} rank)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/test_loss_{num_layers}layer_{name}.png', dpi=150)
    plt.close()
    
    # Frobenius Norms
    plt.figure(figsize=(10, 6))
    plt.plot(exp_metrics['frobenius_norms'], color=colors['exp'], label='Large LR', linewidth=2)
    plt.plot(ctrl_metrics['frobenius_norms'], color=colors['ctrl'], label='Small LR + WD Exp', linewidth=2)
    plt.title(f'{num_layers}-Layer NN: Frobenius Norm of First Layer ({name} rank)')
    plt.xlabel('Epoch')
    plt.ylabel('Frobenius Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/frobenius_{num_layers}layer_{name}.png', dpi=150)
    plt.close()
    
    # Sharpness
    plt.figure(figsize=(10, 6))
    plt.plot(exp_metrics['sharpness'], color=colors['exp'], label='Large LR', linewidth=2)
    plt.plot(ctrl_metrics['sharpness'], color=colors['ctrl'], label='Small LR + WD Exp', linewidth=2)
    plt.title(f'{num_layers}-Layer NN: Sharpness (Largest Hessian Eigenvalue) ({name} rank)')
    plt.xlabel('Epoch')
    plt.ylabel('Sharpness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/sharpness_{num_layers}layer_{name}.png', dpi=150)
    plt.close()
    
    print(f"Plots saved for {num_layers}-layer NN with {name} rank data")
