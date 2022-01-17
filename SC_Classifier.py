import os
from definition import *
from torch import save, no_grad
import shutil
from tqdm import tqdm
import torch
import time
from sc import *

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=30, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.lam = 1e-5
        self.beta_1 = []
        self.beta_2 = []
        self.w = []
        self.m_t = []
        self.v_t = []
        for size_in, size_out in zip(sizes[:0:-1], sizes[-2::-1]):
            self.beta_1.append(np.full((size_in), 0.9).reshape(-1, 1))
            self.beta_2.append(np.full((size_in), 0.99).reshape(-1, 1))
            # self.w.append(np.full(size, 0.1).reshape(-1, 1))
            self.m_t.append(np.full((size_in), 0).reshape(-1, 1))
            self.v_t.append(np.full((size_in), 0).reshape(-1, 1))
        # m_t_b = 0.0
        # v_t_b = 0.0
        self.t = 0
        self.epsilon = 1e-8

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        print("A0 shape = ", params['A0'].shape, "to sc ", to_sc(params["W1"]).shape)
        params['Z1'] = linear(to_sc(params["W1"]).type(torch.BoolTensor), params['A0'].type(torch.BoolTensor))
        # params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = linear(to_sc(params["W2"]).type(torch.BoolTensor), params['Z1'].astype('bool'))
        # params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = linear(to_sc(params["W3"]).type(torch.BoolTensor), params['Z2'].astype('bool'))
        params['A3'] = self.softmax(to_binary(params['Z3']))

        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}
        # print("y train = ", y_train, "output = ", output, "y - o = ", output - y_train)
        # Calculate W3 update
        # print("o - y = ", output - y_train)
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True) + 2 * self.lam * np.sum(params["W3"])
        # print("error = ", error)
        # change_w['W3'] = np.outer(error, params['A2'])
        # print("shape = ", self.beta_1[0].shape, self.m_t[0].shape, self.beta_1[0].shape, error.shape)
        self.m_t[0] = self.beta_1[0] * self.m_t[0] + (1 - self.beta_1[0]) * error
        self.v_t[0] = self.beta_2[0] * self.v_t[0] + (1 - self.beta_2[0]) * np.multiply(error, error)
        m_cap = self.m_t[0] / (1 - (self.beta_1[0] ** self.t))
        v_cap = self.v_t[0] / (1 - (self.beta_2[0] ** self.t))
        change_w['W3'] = np.outer(((m_cap)/(np.sqrt(v_cap)+self.epsilon)), params['A2'])#.reshape(-1, 1)
        # print("change w3 shape = ", change_w["W3"].shape)

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) + 2 * self.lam * np.sum(params["W2"])
        # change_w['W2'] = np.outer(error, params['A1'])
        # error = 2 * np.dot((output - y_train), x_train.transpose()),  / output.shape[0] * self.softmax(params['Z2'], derivative=True)
        self.m_t[1] = self.beta_1[1] * self.m_t[1] + (1 - self.beta_1[1]) * error
        self.v_t[1] = self.beta_2[1] * self.v_t[1] + (1 - self.beta_2[1]) * np.multiply(error, error)
        m_cap = self.m_t[1] / (1 - (self.beta_1[1] ** self.t))
        v_cap = self.v_t[1] / (1 - (self.beta_2[1] ** self.t))
        change_w['W2'] = np.outer(((m_cap)/(np.sqrt(v_cap)+self.epsilon)), params['A1'])#.reshape(-1, 1)
        # print("change w2 shape = ", change_w["W2"].shape)


        # Calculate W1 update
        error = np.dot(params['W2'].T, error) + 2 * self.lam * np.sum(params["W1"])
        # change_w['W1'] = np.outer(error, params['A0'])
        # print("w1 error shape = ", error.shape)
        self.m_t[2] = self.beta_1[2] * self.m_t[2] + (1 - self.beta_1[2]) * error
        self.v_t[2] = self.beta_2[2] * self.v_t[2] + (1 - self.beta_2[2]) * np.multiply(error, error)
        m_cap = self.m_t[2] / (1 - (self.beta_1[2] ** self.t))
        v_cap = self.v_t[2] / (1 - (self.beta_2[2] ** self.t))
        change_w['W1'] = np.outer(((m_cap)/(np.sqrt(v_cap)+self.epsilon)), params['A0'])#.reshape(-1, 1)
        # print("change w1 shape = ", change_w["W1"].shape)

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''
        
        for key, value in changes_to_w.items():
            # print("shapes = ", self.params[key].shape, self.l_rate, value.shape)
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, test_loader):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in tqdm(test_loader):
            x, y = x.detach().numpy(), y.detach().numpy()
            for i in range(x.shape[0]):
                x_ = x[i].reshape(784, -1)
                sc_x_ = to_sc(x_)
                output = self.forward_pass(sc_x_)
                # print("output = ", output)
                pred = binarize(sc_argmax(output))
                # print("pred = ", pred)
                predictions.append(pred == y[i])
        
        return np.mean(predictions)

    def train(self, train_loader, test_loader):
        start_time = time.time()
        count = 0
        for iteration in range(self.epochs):
            for x, y_ in tqdm(train_loader):
                # if count % 1 == 0:
                self.t += 1 #for adam optimizer
                x, y_ = x.detach().numpy(), y_.detach().numpy()
                # count += 1
                x = x.reshape(784,)
                sc_x = to_sc(x)
                # print("sc_x = ", sc_x);
                output = self.forward_pass(sc_x)
                y = np.zeros((10,1))
                y[y_[0]] = 1
                # print("y_ = ", y_)
                changes_to_w = self.backward_pass(y, output)
                # print("changes to w shape = ", changes_to_w['W1'].shape, changes_to_w['W2'].shape, changes_to_w['W3'].shape)
                self.update_network_parameters(changes_to_w)
            
            accuracy = self.compute_accuracy(test_loader)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
            
