import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data import generate_data, generate_data_cifar, generate_data_mnist
from train import train_model, train_model_mnist, train_model_cifar
from utils import generate_viz_2, generate_viz_4, generate_viz_mnist, generate_viz_cifar

np.random.seed(23)
torch.manual_seed(23)

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
  exp = train_model(train_dataset, test_dataset, 0.1, 500, weight_decay=0.00)

  print("Training WD 0.001 Model...")
  ctrl1 = train_model(train_dataset, test_dataset, 0.001, 500, weight_decay=0.001)

  print("Training WD 0.01 Model...")
  ctrl2 = train_model(train_dataset, test_dataset, 0.001, 500, weight_decay=0.01)

  print("Training WD 0.1 Model...")
  ctrl3 = train_model(train_dataset, test_dataset, 0.001, 500, weight_decay=0.1)

  print("Generating Visaulizations...")
  generate_viz_4(exp, ctrl1, ctrl2, ctrl3, name=TYPE)

def experiment_mnist():
  print("Generating MNIST Dataset...")
  train_dataset, test_dataset = generate_data_mnist(NUM_SAMPLES)

  print("Training Catapult Model...")
  exp = train_model_mnist(train_dataset, test_dataset, 0.2, 50, weight_decay=0.00)
  # Epoch 50/50 - Train Loss: 0.006292, Test Loss: 0.035797, Accuracy: 0.8300

  print("Training Control Model...")
  ctrl = train_model_mnist(train_dataset, test_dataset, 0.001, 50, weight_decay=0.00)
  # Epoch 50/50 - Train Loss: 0.064149, Test Loss: 0.071199, Accuracy: 0.5700

  print("Generating Visaulizations...")
  generate_viz_mnist(exp, ctrl, name='mnist')


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


if __name__ == "__main__":
  # experiment1()
  # experiment2()
  # experiment_mnist()
  experiment_cifar_1()