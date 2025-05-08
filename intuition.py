import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data import compute_relu_quadratic, CustomDataset, generate_data
from train import NN_TOY
from utils import compute_ranks, add_annotation_2, add_annotation_4

np.random.seed(24)
torch.manual_seed(24)

INPUT_DIM = 20
NUM_SAMPLES = 500
TYPE = 'ls'

def generate_data_1dir(num_samples, input_dim):
    A = np.zeros((input_dim, input_dim), dtype=int)
    A[0, 0] = 1
    
    X_data = np.random.randn(num_samples, input_dim)
    y_data = compute_relu_quadratic(X_data, A)

    train_size = int(num_samples * 0.8)

    X_train, X_test = X_data[:train_size], X_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    print("Finished Generating Dataset...")

    return train_dataset, test_dataset

def train_model_intu(train_dataset, test_dataset, lr, epochs, weight_decay=0.00):

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

  # Metrics storage
  metrics = {
      'train_loss': [],
      'test_loss': [],
      'W1WT_rank':[],
      'frobenius_norms': [],
      'similarity_e1_v1':[],
      'norm_ratios': [],
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
              W1WT = W1 @ W1.T
              metrics['W1WT_rank'].append(compute_ranks(W1WT))

              W2 = model.fc2.weight.detach().cpu()
              w1_norm = torch.norm(W1).item()
              w2_norm = torch.norm(W2).item()
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

def generate_viz_2(exp, ctrl, name='ls'):
  colors = {'exp': '#1f77b4', 'ctrl': '#ff7f0e'}
  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['train_loss'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['train_loss'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['train_loss'][-1], ctrl['train_loss'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Training Loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('MSE Loss')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"final_img/train-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['test_loss'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['test_loss'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['test_loss'][-1], ctrl['test_loss'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.set_yscale('log') 
  ax1.set_title('Test Loss (log scaled)')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('MSE Loss')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"final_img/test-{name}.png")
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

  plt.savefig(f"final_img/norm-{name}.png")
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

  plt.savefig(f"final_img/rank-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['norm_ratios'], color=colors['exp'], label='Large LR')
  ax1.plot(ctrl['norm_ratios'], color=colors['ctrl'], label='Small LR')
  ax1.set_title('Weight Norm Ratio (||W1||/||W2||)')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Norm Ratio')
  add_annotation_2(ax1, exp['norm_ratios'][-1], ctrl['norm_ratios'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)
  plt.savefig(f"final_img/norm_ratio-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['similarity_e1_v1'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['similarity_e1_v1'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['similarity_e1_v1'][-1], ctrl['similarity_e1_v1'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Cosine Similarity of Train Data & Weight Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Cosine Similarity')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"final_img/cos-{name}.png")
  plt.tight_layout()
  plt.show()

  print("Finished Generating Visaulizations...")

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
  plt.savefig(f"final_img/train-{name}.png")
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['test_loss'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['test_loss'], color=colors['ctrl'], label='Small Lr + WD (0.001)', linestyle=line_styles[0])
  ax1.plot(ctrl1['test_loss'], color=colors['ctrl'], label='Small Lr + WD (0.01)', linestyle=line_styles[1])
  ax1.plot(ctrl2['test_loss'], color=colors['ctrl'], label='Small Lr + WD (0.1)', linestyle=line_styles[2])
  add_annotation_4(ax1, exp['test_loss'][-1], ctrl['test_loss'][-1], ctrl1['test_loss'][-1], ctrl2['test_loss'][-1], colors['exp'], colors['ctrl'], '.2e')
  ax1.set_yscale('log') 
  ax1.set_title('Test Loss (log scaled)')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('MSE Loss')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig(f"final_img/test-{name}.png")
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
  plt.savefig(f"final_img/norm-{name}.png")
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
  plt.savefig(f"final_img/rank-{name}.png")
  plt.show()

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
  plt.savefig(f"final_img/norm_ratio-{name}.png")
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['similarity_e1_v1'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['similarity_e1_v1'], color=colors['ctrl'], label='Small Lr + WD (0.001)', linestyle=line_styles[0])
  ax1.plot(ctrl1['similarity_e1_v1'], color=colors['ctrl'], label='Small Lr + WD (0.01)', linestyle=line_styles[1])
  ax1.plot(ctrl2['similarity_e1_v1'], color=colors['ctrl'], label='Small Lr + WD (0.1)', linestyle=line_styles[2])
  add_annotation_4(ax1, exp['similarity_e1_v1'][-1], ctrl['similarity_e1_v1'][-1], ctrl1['similarity_e1_v1'][-1], ctrl2['similarity_e1_v1'][-1], colors['exp'], colors['ctrl'], '.2e') 
  ax1.set_title('Cosine Similarity of Train Data & Weight Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Cosine Similarity')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(f"final_img/cos-{name}.png")
  plt.show()

  print("Finished Generating Visaulizations...")

def experiment1():
  print("Generating Dataset...")
  train_dataset, test_dataset = generate_data_1dir(NUM_SAMPLES, INPUT_DIM)

  print("Training Catapult Model...")
  # Final Test Loss: 0.222584, R²: 0.733668
  exp = train_model_intu(train_dataset, test_dataset, 0.1, 1700, weight_decay=0.00)

  print("Training Control Model...")
  # Past 1700 epochs we get an error 
  # "eigenvalues, eigenvectors = torch.linalg.eigh(W1WT): torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: 289)."
  # Our weight matrix becomes Ill conditioned and has too many repeated eigenvalues where eigendecoposition struggles to converge.
  # Final Test Loss: 0.237700, R²: 0.715581
  ctrl = train_model_intu(train_dataset, test_dataset, 0.001, 1700, weight_decay=0.00)

  print("Generating Visaulizations...")
  generate_viz_2(exp, ctrl, name=TYPE)

def experiment2():
  print("Generating Dataset...")
  train_dataset, test_dataset = generate_data_1dir(NUM_SAMPLES, INPUT_DIM)

  print("Training Catapult Model...")
  exp = train_model_intu(train_dataset, test_dataset, 0.1, 1500, weight_decay=0.00)

  print("Training WD 0.001 Model...")
  ctrl1 = train_model_intu(train_dataset, test_dataset, 0.001, 1500, weight_decay=0.001)

  print("Training WD 0.01 Model...")
  ctrl2 = train_model_intu(train_dataset, test_dataset, 0.001, 1500, weight_decay=0.01)

  print("Training WD 0.1 Model...")
  ctrl3 = train_model_intu(train_dataset, test_dataset, 0.001, 1500, weight_decay=0.1)

  print("Generating Visaulizations...")
  generate_viz_4(exp, ctrl1, ctrl2, ctrl3, name=f'{TYPE}-wd')

def experiment3():
  for i in ['ls', 'ld', 'hd', 'hs']:
    print("Generating Dataset...")
    train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=i)

    print("Training Catapult Model...")
    exp = train_model_intu(train_dataset, test_dataset, 0.1, 1500, weight_decay=0.00)

    print("Training Control Model...")
    ctrl = train_model_intu(train_dataset, test_dataset, 0.001, 1500, weight_decay=0.00)

    print("Generating Visaulizations...")
    generate_viz_2(exp, ctrl, name=i)

if __name__ == "__main__":
  experiment3()