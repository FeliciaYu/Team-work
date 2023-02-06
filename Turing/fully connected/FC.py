import numpy as np
#全连接层
class Linear(object):
    def __init__(self, inChannel, outChannel):
        scale = np.sqrt(inChannel/2)
        self.W = np.random.standard_normal((inChannel, outChannel)) / scale
        self.b = np.random.standard_normal(outChannel) / scale
        self.W_gradient = np.zeros((inChannel, outChannel))
        self.b_gradient = np.zeros(outChannel)
#前向传播
    def forward(self, x):
        self.x = x
        x_forward = np.dot(self.x, self.W) + self.b
        return x_forward
#反向传播
    def backward(self, delta, learning_rate):
        #梯度计算
        batch_size = self.x.shape[0]
        self.W_gradient = np.dot(self.x.T, delta) / batch_size
        self.b_gradient = np.sum(delta, axis=0) / batch_size
        delta_backward = np.dot(delta, self.W.T)
        #反向传播
        self.W -= self.W_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward
