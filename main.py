import torch
import numpy as np
from data import generate_data
from train import train_model, train_model_exp
from utils import plot_metrics, create_output_dir

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters
INPUT_DIM = 20
NUM_SAMPLES = 500
EPOCHS = 200

def run_experiment(num_layers, rank_type):
    """Run experiment for given number of layers and rank type"""
    print(f"\n=== Experiment: {num_layers}-layer NN with {rank_type} rank data ===")
    
    # Generate data
    print("Generating dataset...")
    train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, rank_type)
    
    # Train large learning rate model
    print("Training Large LR model...")
    large_lr = 0.1 if num_layers == 2 else (0.05 if num_layers == 3 else 0.02)
    exp_metrics = train_model(
        train_dataset, test_dataset, 
        lr=large_lr, 
        epochs=EPOCHS, 
        weight_decay=0.0,
        num_layers=num_layers
    )
    
    # Train small learning rate model with weight decay experiment
    print("Training Small LR + Weight Decay Experiment model...")
    small_lr = 0.001 if num_layers == 2 else (0.0005 if num_layers == 3 else 0.0002)
    ctrl_metrics = train_model_exp(
        train_dataset, test_dataset, 
        lr=small_lr, 
        epochs=EPOCHS, 
        weight_decay=0.01,
        num_layers=num_layers
    )
    
    # Generate plots
    print("Generating plots...")
    plot_metrics(exp_metrics, ctrl_metrics, rank_type, num_layers)
    
    print(f"Completed {num_layers}-layer NN experiment with {rank_type} rank data")

def experiment1():
    """2-layer neural network experiments"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: 2-Layer Neural Networks")
    print("="*60)
    
    for rank_type in ['low', 'medium', 'high']:
        run_experiment(2, rank_type)

def experiment2():
    """3-layer neural network experiments"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: 3-Layer Neural Networks")
    print("="*60)
    
    for rank_type in ['low', 'medium', 'high']:
        run_experiment(3, rank_type)

def experiment3():
    """4-layer neural network experiments"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: 4-Layer Neural Networks")
    print("="*60)
    
    for rank_type in ['low', 'medium', 'high']:
        run_experiment(4, rank_type)

if __name__ == "__main__":
    # Create output directory
    create_output_dir()
    
    print("Starting Neural Network Experiments")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Input dimension: {INPUT_DIM}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Training epochs: {EPOCHS}")
    
    # Run all experiments
    experiment1()  # 2-layer networks
    experiment2()  # 3-layer networks  
    experiment3()  # 4-layer networks
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("Check 'results/' directory for generated plots")
    print("="*60)
