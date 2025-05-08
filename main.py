import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data import generate_data
from train import train_model
from utils import generate_viz_2, generate_viz_4

np.random.seed(24)
torch.manual_seed(24)

INPUT_DIM = 20
NUM_SAMPLES = 500
TYPE = 'ls'

def experiment1():
  print("Generating Dataset...")
  train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=TYPE)

  print("Training Catapult Model...")
  exp = train_model(train_dataset, test_dataset, 0.1, 500, weight_decay=0.00)

  print("Training Control Model...")
  ctrl = train_model(train_dataset, test_dataset, 0.001, 500, weight_decay=0.00)

  print("Generating Visaulizations...")
  generate_viz_2(exp, ctrl, name=TYPE)

def experiment2():
  print("Generating Dataset...")
  train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=TYPE)

  print("Training Catapult Model...")
  exp = train_model(train_dataset, test_dataset, 0.1, 2000, weight_decay=0.00)

  print("Training WD 0.001 Model...")
  ctrl1 = train_model(train_dataset, test_dataset, 0.001, 2000, weight_decay=0.001)

  print("Training WD 0.01 Model...")
  ctrl2 = train_model(train_dataset, test_dataset, 0.001, 2000, weight_decay=0.01)

  print("Training WD 0.1 Model...")
  ctrl3 = train_model(train_dataset, test_dataset, 0.001, 2000, weight_decay=0.1)

  print("Generating Visaulizations...")
  generate_viz_4(exp, ctrl1, ctrl2, ctrl3, name=TYPE)



if __name__ == "__main__":
  experiment1()
  # experiment2()