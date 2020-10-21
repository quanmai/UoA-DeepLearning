#!/usr/bin/env python

import numpy as np

class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, W_init=None, b_init=None):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = 0

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW, self.db = np.zeros(self.W.shape), np.zeros(self.b.shape)
        #Store values for backprop
        self.cache = None

        if W_init: self.W = W_init
        if b_init: self.b = b_init
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        (m, n_C_prev, n_W_prev) = x.shape
        W = self.W
        b = self.b
        #Dimensions of CONV output
        n_W = int((n_W_prev - self.kernel_size + 2 * self.pad) / self.stride) + 1
        n_C = self.out_channel

        #Initialize the output volume y with zeros
        y = np.zeros((m, n_C, n_W))

        for i in range(m):
            x_i = x[i,:,:]
            for c in range (n_C):
                for w in range(n_W):
                    start = w * self.stride
                    end = start + self.kernel_size
                    y[i,c,w] = np.sum(np.multiply(x_i[:,start:end], W[c,:]) +  b[c])

        #Check output shape
        assert(y.shape == (m, n_C, n_W))

        #Save information for backprop
        self.cache = x.copy()

        return y
    def backward(self, delta):
        """
        Implement the backward propagation for a convolution function

        Arguments:
        delta -- gradient of the cost with respect to the output of the conv layer, numpy array of shape (m, n_C, n_W)

        Returns:
        dx -- gradient of the cost with respect to the input of the conv layer (x),
               numpy array of shape (m, n_C_prev, n_W_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (n_C_prev, n_C, f)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (n_C)
        """
        # Retrieve information from "cache"
        x = self.cache
        (m, n_C_prev, n_W_prev) = x.shape
        # Initialize dA_prev, dW, db with the correct shapes

        dx = np.zeros(x.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)

        (m, n_C, n_W) = delta.shape
        for i in range (m):
            x_i = x[i]
            dx_i = dx[i]
            for c in range (self.out_channel):
                for w in range (n_W):
                    start = w * self.stride
                    end = start + self.kernel_size
                    segment = x_i[:,start:end]
                    #print(self.W[c,:,:].shape)
                    #print(delta[i,c,w].shape)
                    #print(dx_i[:,start:end].shape)
                    dx_i[:, start:end] += self.W[c,:,:] * delta[i, c, w]
                    dW[c,:,:] += segment * delta[i, c, w]
                    db[c] += delta[i, c, w]

        self.dW = dW
        self.db = db

        return dx
