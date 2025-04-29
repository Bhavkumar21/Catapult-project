import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data import generate_data
from train import train_model
from utils import generate_viz_2

np.random.seed(23)
torch.manual_seed(23)

INPUT_DIM = 20
NUM_SAMPLES = 500
TYPE = 'hs'

def experiment1():
  print("Generating Dataset...")
  train_dataset, test_dataset = generate_data(NUM_SAMPLES, INPUT_DIM, type=TYPE)

  print("Training Catapult Model...")
  exp = train_model(train_dataset, test_dataset, 0.1, 500, weight_decay=0.00)

  print("Training Control Model...")
  ctrl = train_model(train_dataset, test_dataset, 0.001, 500, weight_decay=0.00)

  print("Generating Visaulizations...")
  generate_viz_2(exp, ctrl, name=TYPE)

experiment1()