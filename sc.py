import numpy as np
from definition import *
import torch

def bit_gen(number):
  prob = (number.item() + 1) / 2
  prob = np.clip(prob, -1, 1)
  assert(prob >= 0)
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

def linear(tensors1, tensors2):
  tensors = np.zeros((tensors1.shape[0], BITLEN))
  tmp_tensor = np.zeros((tensors2.shape[0], BITLEN))
  for i in range(tensors1.shape[0]):
    for j in range(tensors2.shape[0]):
      tmp_tensor[j] = xnor(tensors1[i][j], tensors2[j])
    tensors[i] = mux(tmp_tensor)
  # print(tensors.shape)
  return tensors


def mux(tensors):
  trans = np.transpose(tensors)
  out = np.zeros((BITLEN))
  for i in range(trans.shape[0]):
    out[i] = np.random.choice(trans[i])
  return out

def cmp(tensor1, tensor2):
  if (binarize(tensor1) > binarize(tensor2)):
    return True
  else:
    return False

def sc_argmax(tensors):
  max_idx = 0
  for i in range(tensors.shape[0]):
    if (cmp(tensors[i], tensors[max_idx])) :
      max_idx = i
  return tensors[max_idx]