import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data import generate_data, generate_data_1dir
from train import train_model, train_model_exp
from utils import generate_viz_2, generate_viz_4

np.random.seed(24)
torch.manual_seed(24)

INPUT_DIM = 20
NUM_SAMPLES = 500
TYPE = 'ls'

def experiment1():
  print("Generating Dataset...")
  train_dataset, test_dataset = generate_data_1dir(NUM_SAMPLES, INPUT_DIM)

  print("Training Catapult Model...")
  # Final Test Loss: 0.222584, R²: 0.733668
  exp = train_model(train_dataset, test_dataset, 0.1, 100, weight_decay=0.00)

  print("Training Control Model...")
  # Past 1700 epochs we get an error 
  # "eigenvalues, eigenvectors = torch.linalg.eigh(W1WT): torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: 289)."
  # Our weight matrix becomes Ill conditioned and has too many repeated eigenvalues where eigendecoposition struggles to converge.
  # Final Test Loss: 0.237700, R²: 0.715581
  ctrl = train_model(train_dataset, test_dataset, 0.001, 100, weight_decay=0.00)

  print("Generating Visaulizations...")
  generate_viz_2(exp, ctrl, name=TYPE)

def experiment2():
  print("Generating Dataset...")
  train_dataset, test_dataset = generate_data_1dir(NUM_SAMPLES, INPUT_DIM)

  print("Training Catapult Model...")
  exp = train_model(train_dataset, test_dataset, 0.1, 1500, weight_decay=0.00)

  print("Training WD 0.001 Model...")
  ctrl1 = train_model(train_dataset, test_dataset, 0.001, 1500, weight_decay=0.001)

  print("Training WD 0.01 Model...")
  ctrl2 = train_model(train_dataset, test_dataset, 0.001, 1500, weight_decay=0.01)

  print("Training WD 0.1 Model...")
  ctrl3 = train_model(train_dataset, test_dataset, 0.001, 1500, weight_decay=0.1)

  print("Generating Visaulizations...")
  generate_viz_4(exp, ctrl1, ctrl2, ctrl3, name=f'{TYPE}-wd')

def experiment3():
  for i in ['ls', 'ld', 'hd', 'hs']:
    print(f"Generating Dataset {i}...")
    train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=i)

    print("Training Catapult Model...")
    exp = train_model(train_dataset, test_dataset, 0.11, 1500, weight_decay=0.00)

    print("Training Control Model...")
    ctrl = train_model(train_dataset, test_dataset, 0.001, 1500, weight_decay=0.00)

    print("Generating Visaulizations...")
    generate_viz_2(exp, ctrl, name=i)

def experiment4():
  # for i in ['ls', 'ld', 'hd', 'hs']:
  for i in ['hd']:
    print(f"Generating Dataset {i}...")
    train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=i)

    print("Training Catapult Model...")
    exp = train_model(train_dataset, test_dataset, 0.1, 2000, weight_decay=0.00)

    print("Training WD 0.001 Model...")
    ctrl1 = train_model(train_dataset, test_dataset, 0.001, 2000, weight_decay=0.001)

    print("Training WD 0.01 Model...")
    ctrl2 = train_model(train_dataset, test_dataset, 0.001, 2000, weight_decay=0.01)

    print("Training WD 0.1 Model...")
    ctrl3 = train_model(train_dataset, test_dataset, 0.001, 2000, weight_decay=0.1)

    print("Generating Visaulizations...")
    generate_viz_4(exp, ctrl1, ctrl2, ctrl3, name=f'{i}-wd')

def experiment5():
  for i in ['ls', 'ld', 'hd', 'hs']:
    print(f"Generating Dataset {i}...")
    train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=i)

    print("Training Catapult Model...")
    exp = train_model(train_dataset, test_dataset, 0.002, 1500, weight_decay=0.00, imb=True)

    print("Training Control Model...")
    ctrl = train_model(train_dataset, test_dataset, 0.0001, 1500, weight_decay=0.00, imb=True)

    print("Generating Visaulizations...")
    generate_viz_2(exp, ctrl, name=f"imb-{i}")

def experiment6():
  for i in ['hs']:
    print(f"Generating Dataset {i}...")
    train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=i)

    print("Training Catapult Model...")
    exp = train_model(train_dataset, test_dataset, 0.002, 1500, weight_decay=0.00, imb=True)

    print("Training WD 0.001 Model...")
    ctrl1 = train_model(train_dataset, test_dataset, 0.0001, 1500, weight_decay=0.001, imb=True)

    print("Training WD 0.01 Model...")
    ctrl2 = train_model(train_dataset, test_dataset, 0.0001, 1500, weight_decay=0.01, imb=True)

    print("Training WD 0.1 Model...")
    ctrl3 = train_model(train_dataset, test_dataset, 0.0001, 1500, weight_decay=0.1, imb=True)

    print("Generating Visaulizations...")
    generate_viz_4(exp, ctrl1, ctrl2, ctrl3, name=f'imb-{i}-wd')

def experiment7():
  print("Generating Dataset...")
  train_dataset, test_dataset = generate_data_1dir(NUM_SAMPLES, INPUT_DIM)

  print("Training Catapult Model...")
  exp = train_model(train_dataset, test_dataset, 0.1, 2500, weight_decay=0.00)

  candidates = [
    # (0.01, 1.00),
    # (0.01, 2.00),
    (0.01, 60.00),
    # (0.001, 4.00),
  ]
  
  for lr, wd in candidates:
    print(f"Testing LR={lr}, WD={wd}")
    ctrl = train_model_exp(train_dataset, test_dataset, lr, 2500, weight_decay=wd)
    generate_viz_2(exp, ctrl, name=f'FINAL-ls-{lr}-{wd}')


if __name__ == "__main__":
  experiment7()