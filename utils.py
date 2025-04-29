import torch
import numpy as np
from scipy.linalg import svdvals
from numpy.linalg import svd, cond, matrix_rank
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(23)
torch.manual_seed(23)

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

def generate_lowrank_matrix(dim, target_rank, sparsity):
  """Generate a low rank matrix with controlled rank and sparsity."""
  # Create orthogonal basis for column space
  U, _ = np.linalg.qr(np.random.randn(dim, target_rank))
  
  # Create sparse mask
  sparse_mask = sparse_random(
      dim, dim, density=sparsity, 
      random_state=42
  ).astype(bool).toarray()
  
  # Construct matrix
  A = U @ U.T  # Rank target_rank orthogonal projection
  A = A * sparse_mask  # Apply sparsity

  A = csr_matrix(A)
  actual_cond = cond(A.toarray())
  nnz = A.count_nonzero()
  
  print(f"Dimension: {A.shape}")
  print(f"Effective rank: {compute_ranks(A.toarray())}")
  print(f"Condition number: {actual_cond:.2f}")
  print(f"Sparsity: {nnz/(dim*dim):.4f}")
  
  return csr_matrix(A)

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
  for i in range(10):
      ax.plot(exp['top_svs'][:, i], 
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
  """Add annotation to the plot."""
  ax.text(1.05, 0.30, f'Larger Lr (b=16): {exp_val:{fmt}}', color=exp_color, ha='left', va='top', 
          transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
  ax.text(1.05, 0.23, f'Larger Lr (b=8): {ctrl_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
          transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
  ax.text(1.05, 0.16, f'Larger Lr (b=4): {ctrl1_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
          transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
  ax.text(1.05, 0.09, f'Larger Lr (b=2): {ctrl2_val:{fmt}}', color=ctrl_color, ha='left', va='top', 
          transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))


