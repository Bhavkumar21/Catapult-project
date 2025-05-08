import torch
import numpy as np
from scipy.linalg import svdvals
from numpy.linalg import svd, cond, matrix_rank
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(24)
torch.manual_seed(24)

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

def generate_highrank_matrix(dim=1000, target_condition=1, sparsity=0.1):
  """Generate a high rank matrix with controlled condition number and sparsity."""
  # Create random orthogonal matrices
  Q1, _ = np.linalg.qr(np.random.randn(dim, dim))
  Q2, _ = np.linalg.qr(np.random.randn(dim, dim))
  
  # Create singular values with controlled condition number
  singular_values = np.linspace(target_condition, 1, dim)
  S = np.diag(singular_values)
  
  # Form matrix A = Q1 @ S @ Q2.T
  A = Q1 @ S @ Q2.T
  
  # Apply sparsity pattern while preserving rank
  mask = np.random.rand(dim, dim) < sparsity
  np.fill_diagonal(mask, False)  # Don't zero out diagonal
  
  A_sparse = A.copy()
  A_sparse[mask] = 0
  
  # If rank dropped too much, reduce sparsity
  current_rank = matrix_rank(A_sparse)
  attempts = 0
  while current_rank < dim and attempts < 10:
      sparsity /= 2
      mask = np.random.rand(dim, dim) < sparsity
      np.fill_diagonal(mask, False)
      
      A_sparse = A.copy()
      A_sparse[mask] = 0
      
      current_rank = matrix_rank(A_sparse)
      attempts += 1
  
  # Calculate final properties
  final_condition = cond(A_sparse)
  final_rank = matrix_rank(A_sparse)
  final_sparsity = np.sum(A_sparse == 0) / (dim * dim)
  
  print(f"Final matrix properties:")
  print(f"- Dimension: {dim}x{dim}")
  print(f"- Rank: {compute_ranks(A_sparse)}")
  print(f"- Condition number: {final_condition:.2f}")
  print(f"- Sparsity: {final_sparsity:.2%}")
  
  return A_sparse

def generate_lowrank_matrix(dim, target_rank, sparsity=0):
  # Step 1: Create factors that will give us a low rank matrix
  # Using random normal matrices of appropriate dimensions
  A_left = np.random.randn(dim, target_rank)
  A_right = np.random.randn(target_rank, dim)
  
  # Step 2: Matrix multiplication to create the low rank matrix
  # This ensures rank ≤ target_rank
  A = np.dot(A_left, A_right)
  
  # Step 3: Add minimal noise to ensure numerical stability
  # The small magnitude ensures we don't significantly affect the rank
  noise = np.random.randn(dim, dim) * 1e-10
  A += noise
  
  # Return as sparse matrix format for consistency
  A_sparse = csr_matrix(A)
  A_dense = A_sparse.toarray()
  actual_cond = cond(A_dense)
  nnz = np.count_nonzero(A_dense)
  
  print(f"Dimension: {A.shape}")
  print(f"Effective rank: {compute_ranks(A)}")  # Fixed: use A directly
  print(f"Condition number: {actual_cond:.2f}")
  print(f"Sparsity: {nnz/(dim*dim):.4f}")
  
  return A_sparse

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

  plt.savefig(f"img/train-{name}.png")
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

  plt.savefig(f"img/test-{name}.png")
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
      
  # for i in range(3):
  #     ax.plot(ctrl['top_svs'][:, i], 
  #             color=small_lr_colors[i],
  #             linestyle=line_styles[i],
  #             linewidth=lw,
  #             alpha=1.0,
  #             label=f'small lr σ_{i+1}')
      
  # for i in range(20):
  #    print('top svs', i, exp['top_svs'][:40, i])
  
  ax.set_title('Largest Singular Values of W1', fontsize=16)
  ax.set_ylabel('Singular Values (W1)', fontsize=14)
  ax.set_xlabel('Training Iterations', fontsize=14)
  # ax.legend(loc='upper right')
  ax.grid(True, alpha=0.3) 

  plt.savefig(f"img/svs-{name}.png")
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

  plt.savefig(f"img/norm-{name}.png")
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

  print("Finished Generating Visaulizations...")


def add_annotation_4(ax, exp_val, ctrl_val, ctrl1_val, ctrl2_val, exp_color, ctrl_color, fmt):
    ax.text(1.05, 0.30, f'Larger Lr: {exp_val:{fmt}}', color=exp_color, ha='left', va='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1.05, 0.23, f'Small Lr + WD (0.001): {ctrl_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1.05, 0.16, f'Small Lr + WD (0.01): {ctrl1_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1.05, 0.09, f'Small Lr + WD (0.1): {ctrl2_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    

def generate_viz_4(exp, ctrl, ctrl1, ctrl2, name='ls'):
  colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
  line_styles = ['-', '--', ':']
  fig, ax1 = plt.subplots(figsize=(10, 6))

  ax1.plot(exp['train_loss'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['train_loss'], color=colors['ctrl'], label='Small Lr + WD (0.001)', linestyle=line_styles[0])
  ax1.plot(ctrl1['train_loss'], color=colors['ctrl'], label='Small Lr + WD (0.01)', linestyle=line_styles[1])
  ax1.plot(ctrl2['train_loss'], color=colors['ctrl'], label='Small Lr + WD (0.1)', linestyle=line_styles[2])

  add_annotation_4(ax1, exp['train_loss'][-1], ctrl['train_loss'][-1], ctrl1['train_loss'][-1], ctrl2['train_loss'][-1], colors['exp'], colors['ctrl'], '.2e')


  ax1.set_title('Training Loss')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('MSE Loss')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig(f"img/train-{name}-wd.png")
  plt.show()


  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['test_loss'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['test_loss'], color=colors['ctrl'], label='Small Lr + WD (0.001)', linestyle=line_styles[0])
  ax1.plot(ctrl1['test_loss'], color=colors['ctrl'], label='Small Lr + WD (0.01)', linestyle=line_styles[1])
  ax1.plot(ctrl2['test_loss'], color=colors['ctrl'], label='Small Lr + WD (0.1)', linestyle=line_styles[2])

  add_annotation_4(ax1, exp['test_loss'][-1], ctrl['test_loss'][-1], ctrl1['test_loss'][-1], ctrl2['test_loss'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.set_yscale('log') 
  ax1.set_title('Test Loss (log scaled)')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('MSE Loss')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(f"img/test-{name}-wd.png")
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['frobenius_norms'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['frobenius_norms'], color=colors['ctrl'], label='Small Lr + WD (0.001)', linestyle=line_styles[0])
  ax1.plot(ctrl1['frobenius_norms'], color=colors['ctrl'], label='Small Lr + WD (0.01)', linestyle=line_styles[1])
  ax1.plot(ctrl2['frobenius_norms'], color=colors['ctrl'], label='Small Lr + WD (0.1)', linestyle=line_styles[2])
  add_annotation_4(ax1, exp['frobenius_norms'][-1], ctrl['frobenius_norms'][-1], ctrl1['frobenius_norms'][-1], ctrl2['frobenius_norms'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Frobenius Norm of Weight 1 Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Frobenius Norm Value')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig(f"img/norm-{name}-wd.png")
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['W1WT_rank'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['W1WT_rank'], color=colors['ctrl'], label='Small Lr + WD (0.001)', linestyle=line_styles[0])
  ax1.plot(ctrl1['W1WT_rank'], color=colors['ctrl'], label='Small Lr + WD (0.01)', linestyle=line_styles[1])
  ax1.plot(ctrl2['W1WT_rank'], color=colors['ctrl'], label='Small Lr + WD (0.1)', linestyle=line_styles[2])
  add_annotation_4(ax1, exp['W1WT_rank'][-1], ctrl['W1WT_rank'][-1], ctrl1['W1WT_rank'][-1], ctrl2['W1WT_rank'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Effective Rank of Weight 1 Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Effective Rank')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig(f"img/rank-{name}-wd.png")
  plt.show()

  print("Finished Generating Visaulizations...")
