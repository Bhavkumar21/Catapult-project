import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# GPU setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

np.random.seed(42)
torch.manual_seed(42)

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

class ThreeLayerNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

class FourLayerNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x

def compute_hessian_largest_eigenvalue(model, data_loader, criterion):
    """Compute the largest eigenvalue of the Hessian matrix (sharpness)"""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Compute gradients
    total_loss = 0
    model.zero_grad()
    
    for data, target in data_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        data = data.view(data.size(0), -1)
        target = target.view(-1, 1)
        
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss
    
    # Compute gradients
    grads = torch.autograd.grad(total_loss, params, create_graph=True)
    
    # Use power iteration to find largest eigenvalue
    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum([torch.sum(vi**2) for vi in v]))
    v = [vi / v_norm for vi in v]
    
    for _ in range(10):  # Power iteration steps
        grad_grad = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        v_norm = torch.sqrt(sum([torch.sum(ggi**2) for ggi in grad_grad]))
        if v_norm > 1e-8:
            v = [ggi / v_norm for ggi in grad_grad]
        else:
            break
    
    # Compute eigenvalue
    grad_grad = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=False)
    eigenvalue = sum([torch.sum(vi * ggi) for vi, ggi in zip(v, grad_grad)])
    
    return eigenvalue.item()

def train_model(train_dataset, test_dataset, lr, epochs, weight_decay=0.0, num_layers=2):
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    # Select model based on number of layers
    if num_layers == 2:
        model = TwoLayerNN().to(DEVICE)
    elif num_layers == 3:
        model = ThreeLayerNN().to(DEVICE)
    elif num_layers == 4:
        model = FourLayerNN().to(DEVICE)
    else:
        raise ValueError("Number of layers must be 2, 3, or 4")
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'frobenius_norms': [],
        'sharpness': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.view(data.size(0), -1)
            target = target.view(-1, 1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            train_samples += data.size(0)
        
        metrics['train_loss'].append(train_loss / train_samples)
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                data = data.view(data.size(0), -1)
                target = target.view(-1, 1)
                
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                test_samples += data.size(0)
        
        metrics['test_loss'].append(test_loss / test_samples)
        
        # Compute Frobenius norm of first layer
        W1 = model.fc1.weight.detach().cpu()
        frobenius_norm = torch.norm(W1, 'fro').item()
        metrics['frobenius_norms'].append(frobenius_norm)
        
        # Compute sharpness (largest eigenvalue of Hessian)
        try:
            sharpness = compute_hessian_largest_eigenvalue(model, train_loader, criterion)
            metrics['sharpness'].append(sharpness)
        except:
            metrics['sharpness'].append(0.0)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {metrics['train_loss'][-1]:.6f}, Test Loss: {metrics['test_loss'][-1]:.6f}")
    
    return metrics

def train_model_exp(train_dataset, test_dataset, lr, epochs, weight_decay=0.0, num_layers=2):
    """Training with weight decay that gets turned off after 6 iterations"""
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    # Select model based on number of layers
    if num_layers == 2:
        model = TwoLayerNN().to(DEVICE)
    elif num_layers == 3:
        model = ThreeLayerNN().to(DEVICE)
    elif num_layers == 4:
        model = FourLayerNN().to(DEVICE)
    else:
        raise ValueError("Number of layers must be 2, 3, or 4")
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'frobenius_norms': [],
        'sharpness': []
    }
    
    global_iteration = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for data, target in train_loader:
            # Turn off weight decay after 6 iterations
            if global_iteration == 6:
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = 0.0
                print(f"Turned off weight decay at iteration {global_iteration}")
            
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.view(data.size(0), -1)
            target = target.view(-1, 1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            train_samples += data.size(0)
            global_iteration += 1
        
        metrics['train_loss'].append(train_loss / train_samples)
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                data = data.view(data.size(0), -1)
                target = target.view(-1, 1)
                
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                test_samples += data.size(0)
        
        metrics['test_loss'].append(test_loss / test_samples)
        
        # Compute Frobenius norm of first layer
        W1 = model.fc1.weight.detach().cpu()
        frobenius_norm = torch.norm(W1, 'fro').item()
        metrics['frobenius_norms'].append(frobenius_norm)
        
        # Compute sharpness (largest eigenvalue of Hessian)
        try:
            sharpness = compute_hessian_largest_eigenvalue(model, train_loader, criterion)
            metrics['sharpness'].append(sharpness)
        except:
            metrics['sharpness'].append(0.0)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {metrics['train_loss'][-1]:.6f}, Test Loss: {metrics['test_loss'][-1]:.6f}")
    
    return metrics
