import numpy as np
from definition import *
import torch
def bit_gen(number):
  prob = (number.item() + 1) / 2
  return np.random.choice([bool(0), bool(1)], size=BITLEN, p=[1 - prob, prob])
def binarize(tensor):
  return  2 * np.sum(tensor) / BITLEN - 1
def to_binary(tensor):
  tensor_shape = tensor.shape
  tmp_tensor = tensor.reshape(-1,BITLEN)
  binary_tensor = np.zeros((tmp_tensor.shape[0:-1]))
  for i in range(tmp_tensor.shape[0]):
    binary_tensor[i] = binarize(tmp_tensor[i])
  binary_tensor = binary_tensor.reshape((tensor_shape[0:-1]))
  return binary_tensor

def to_sc(tensor):
  tensor_shape = tensor.shape
  tmp_tensor = tensor.reshape(-1,)
  sc_tensor = np.zeros((tmp_tensor.shape[0], BITLEN))
  
  for i in range(tmp_tensor.shape[0]):
    sc_tensor[i] = bit_gen(tmp_tensor[i])
  sc_tensor = sc_tensor.reshape(tensor_shape + (BITLEN,))
  return torch.from_numpy(sc_tensor)

def xnor(tensor1, tensor2):
  return np.logical_not((tensor1 ^ tensor2))

def mux(tensors):
  trans = np.transpose(tensors)
  out = np.zeros((BITLEN))
  for i in range(trans.shape[0]):
    out[i] = np.random.choice(trans[i])
  return out