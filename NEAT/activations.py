import numpy as np

def sigmoid(nodes):
  sig = 1/(1 + np.exp(-nodes)) 
  return sig

def relu(x):
  return max([0, x])

def tanh(x):
  return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
