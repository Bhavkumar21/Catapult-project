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

def generate_data_cifar(num_samples):
    cifar_train = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=None)
    cifar_test = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=None)
    
    train_size = int(num_samples * 0.8)
    test_size = num_samples - train_size
    
    # Convert to numpy arrays and float type
    X_train = np.array(cifar_train.data[:train_size], dtype=np.float32)
    y_train = np.array(cifar_train.targets[:train_size])
    X_test = np.array(cifar_test.data[:test_size], dtype=np.float32)
    y_test = np.array(cifar_test.targets[:test_size])
    
    # Scale pixel values to [0, 1]
    X_train /= 255.0
    X_test /= 255.0
    
    # Calculate mean and std per channel from training data
    # Reshape to get all pixels for each channel
    train_pixels = X_train.reshape(-1, 3)
    mean = np.mean(train_pixels, axis=0)
    std = np.std(train_pixels, axis=0)
    
    print(f"CIFAR-10 Channel Mean: {mean}")
    print(f"CIFAR-10 Channel Std: {std}")
    
    # Transpose to (N, C, H, W) format
    X_train = X_train.transpose((0, 3, 1, 2))
    X_test = X_test.transpose((0, 3, 1, 2))
    
    # Apply normalization to each channel
    for i in range(3):
        X_train[:, i] = (X_train[:, i] - mean[i]) / std[i]
        X_test[:, i] = (X_test[:, i] - mean[i]) / std[i]
    
    train_dataset = CustomDatasetClassifier(X_train, y_train)
    test_dataset = CustomDatasetClassifier(X_test, y_test)
    
    print("Finished Generating Normalized CIFAR-10 Dataset...")
    return train_dataset, test_dataset

class CustomDatasetClassifier(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_viz_cifar(exp, ctrl, name='cifar10'):
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
  ax1.plot(exp['frobenius_conv_norm'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['frobenius_conv_norm'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['frobenius_conv_norm'][-1], ctrl['frobenius_conv_norm'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Frobenius Norm of Convolution Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Frobenius Norm Value')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"img/norm-conv-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['frobenius_fc_norm'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['frobenius_fc_norm'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['frobenius_fc_norm'][-1], ctrl['frobenius_fc_norm'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Frobenius Norm of FC Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Frobenius Norm Value')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"img/norm-fc-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['WWT_fc_rank'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['WWT_fc_rank'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['WWT_fc_rank'][-1], ctrl['WWT_fc_rank'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Effective Rank of Weight FC Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Effective Rank')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"img/rank-fc-{name}.png")
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax1.plot(exp['WWT_conv_rank'], color=colors['exp'], label='Larger Lr')
  ax1.plot(ctrl['WWT_conv_rank'], color=colors['ctrl'], label='Small Lr')
  add_annotation_2(ax1, exp['WWT_conv_rank'][-1], ctrl['WWT_conv_rank'][-1],colors['exp'], colors['ctrl'], '.2e')
  ax1.set_title('Effective Rank of Weight Conv Matrix')
  ax1.set_xlabel('Training Iterations')
  ax1.set_ylabel('Effective Rank')
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)

  plt.savefig(f"img/rank-conv-{name}.png")
  plt.tight_layout()
  plt.show()

  print("Finished Generating Visaulizations...")

def experiment_cifar_1():
  print("Generating CIFAR-10 Dataset...")
  train_dataset, test_dataset = generate_data_cifar(NUM_SAMPLES)

  print("Training Catapult Model...")
  exp = train_model_cifar(train_dataset, test_dataset, 0.002, 500, weight_decay=0.00)
  # Epoch 500/500 - Train Loss: 0.008400, Test Loss: 0.113012, Accuracy: 0.3000


  print("Training Control Model...")
  ctrl = train_model_cifar(train_dataset, test_dataset, 1e-6, 500, weight_decay=0.00)
  # Epoch 500/500 - Train Loss: 0.109722, Test Loss: 0.121507, Accuracy: 0.0800


  print("Generating Visaulizations...")
  generate_viz_cifar(exp, ctrl, name='cifar')