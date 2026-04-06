import os
import torch
import pickle
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from rl import TouchEnv
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv

def imitate(lr=0.001, epochs=500):
    with open('demos2.pkl', 'rb') as file:
        X, y = pickle.load(file)

    # train a small neural network to predict actions from observations using PyTorch
    # Create the same network architecture that PPO used for a policy,
    # namely 2 hidden layers of 64 nodes with ReLU activations after each
    # layer. No activation after the linear output layer.



    # Use PyTorch to save your policy as imitation_policy.pt


if __name__ == '__main__':
    imitate()