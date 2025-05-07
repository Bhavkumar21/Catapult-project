class CustomDatasetClassifier(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def generate_data_mnist(num_samples):
    mnist_train = datasets.MNIST(root='./mnist', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='./mnist', train=False, download=True, transform=None)
    
    train_size = int(num_samples * 0.8)
    test_size = num_samples - train_size
    
    # Convert to numpy arrays and float type
    X_train = np.array(mnist_train.data[:train_size], dtype=np.float32)
    y_train = np.array(mnist_train.targets[:train_size])
    X_test = np.array(mnist_test.data[:test_size], dtype=np.float32)
    y_test = np.array(mnist_test.targets[:test_size])
    
    # Scale pixel values to [0, 1]
    X_train /= 255.0
    X_test /= 255.0
    
    # Calculate mean and std from training data
    train_pixels = X_train.reshape(-1)
    mean = np.mean(train_pixels)
    std = np.std(train_pixels)
    
    print(f"MNIST Mean: {mean}")
    print(f"MNIST Std: {std}")
    
    # Reshape to (N, C, H, W) format, MNIST is grayscale so C=1
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    # Apply normalization
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    train_dataset = CustomDatasetClassifier(X_train, y_train)
    test_dataset = CustomDatasetClassifier(X_test, y_test)
    
    print("Finished Generating Normalized MNIST Dataset...")
    return train_dataset, test_dataset


class NN_MNIST(nn.Module):
    """
    3-hidden layer fully-connected network with ReLU non-linearity for MNIST
    classification, as used in Lewkowycz et al. 2020.
    """
    def __init__(self, input_dim=784, hidden_dim=1200, num_classes=10):
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

def generate_viz_mnist(exp, ctrl, name='mnist'):
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

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['frobenius_norm1'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['frobenius_norm1'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['frobenius_norm1'][-1], ctrl['frobenius_norm1'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Frobenius Norm of Weight 1 Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Frobenius Norm Value')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"img/norm1-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['frobenius_norm2'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['frobenius_norm2'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['frobenius_norm2'][-1], ctrl['frobenius_norm2'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Frobenius Norm of Weight 2 Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Frobenius Norm Value')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"img/norm2-{name}.png")
  plt.tight_layout()
  plt.show()

  # fig, ax1 = plt.subplots(figsize=(10, 6))
  # ax1.plot(exp['frobenius_norm3'], color=colors['exp'], label='Larger Lr')
  # ax1.plot(ctrl['frobenius_norm3'], color=colors['ctrl'], label='Small Lr')
  # add_annotation_2(ax1, exp['frobenius_norm3'][-1], ctrl['frobenius_norm3'][-1],colors['exp'], colors['ctrl'], '.2e')
  # ax1.set_title('Frobenius Norm of Weight 3 Matrix')
  # ax1.set_xlabel('Training Iterations')
  # ax1.set_ylabel('Frobenius Norm Value')
  # ax1.legend(loc='upper right')
  # ax1.grid(True, alpha=0.3)

  # plt.savefig(f"img/norm3-{name}.png")
  # plt.tight_layout()
  # plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['W1WT_rank'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['W1WT_rank'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['W1WT_rank'][-1], ctrl['W1WT_rank'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Effective Rank of Weight 1 Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Effective Rank')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"img/rank1-{name}.png")
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

  # fig, ax1 = plt.subplots(figsize=(10, 6))
  # ax1.plot(exp['W3WT_rank'], color=colors['exp'], label='Larger Lr')
  # ax1.plot(ctrl['W3WT_rank'], color=colors['ctrl'], label='Small Lr')
  # add_annotation_2(ax1, exp['W3WT_rank'][-1], ctrl['W3WT_rank'][-1],colors['exp'], colors['ctrl'], '.2e')
  # ax1.set_title('Effective Rank of Weight 3 Matrix')
  # ax1.set_xlabel('Training Iterations')
  # ax1.set_ylabel('Effective Rank')
  # ax1.legend(loc='upper right')
  # ax1.grid(True, alpha=0.3)

  # plt.savefig(f"img/rank3-{name}.png")
  # plt.tight_layout()
  # plt.show()
  print("Finished Generating Visaulizations...")

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

def experiment_mnist():
  print("Generating MNIST Dataset...")
  train_dataset, test_dataset = generate_data_mnist(NUM_SAMPLES)

  print("Training Catapult Model...")
  exp = train_model_mnist(train_dataset, test_dataset, 0.2, 10, weight_decay=0.00)
  # Epoch 50/50 - Train Loss: 0.006292, Test Loss: 0.035797, Accuracy: 0.8300

  print("Training Control Model...")
  ctrl = train_model_mnist(train_dataset, test_dataset, 0.001, 10, weight_decay=0.00)
  # Epoch 50/50 - Train Loss: 0.064149, Test Loss: 0.071199, Accuracy: 0.5700

  print("Generating Visaulizations...")
  generate_viz_mnist(exp, ctrl, name='mnist')