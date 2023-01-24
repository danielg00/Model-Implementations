import torch as pt
import numpy as np
import matplotlib.pyplot as plt

X = pt.load("/home/daniel/DATASETS/MNIST/processed/processed_images.pt")
X = np.reshape(X, (60000, 784))
# X = np.floor((100+X)/255)


def data_sample(D, n):
    idx = np.random.randint(0, D.shape[0], n)
    return D[idx]


def sample_hidden(h):
    probs = np.random.uniform(0, 1, (1, h.shape[1]))
    return np.ceil(probs - h)

        
def sigmoid(x):
    return 1/(1 + np.exp(-1*x))

diffs = []
class RBM:
    def __init__(self, v_size, h_size):
        self.v_size, self.h_size = v_size, h_size
        self.W = np.random.randn(v_size, h_size)
        self.a = np.random.randn(1, v_size)
        self.b = np.random.randn(1, h_size)

    def train(self, data, bs, steps, lr):
        for i in range(steps):
            m1 = self.W
            v = data_sample(data, 1)
            h = sigmoid(self.b + (v @ self.W))

            pos_grad = np.outer(v, h)
            
            h_s = sample_hidden(h)
            
            v_ = (h_s @ self.W.T) + self.a
            h_ = sigmoid(self.b + (v_ @ self.W))

            neg_grad = np.outer(v_, h_)

            total_grad_W = pos_grad-neg_grad
            self.W = self.W + lr*total_grad_W

            self.a = self.a + lr*(v - v_)
            self.b = self.b + lr*(h - h_)

            m2 = self.W
            mu = np.mean(np.abs(m2 - m1))
            diffs.append(mu)
            
            if i % 10 == 0:
                print("%d : %f" % (i, mu))

        plt.plot(range(len(diffs)), diffs)
        plt.show()

    def visualise_node(self, idx):
        assert idx <= self.h_size and idx > 0, "Out of bounds with %d" % idx
        pass


rbm = RBM(784, 512)
rbm.train(X, 1, 1000, 0.0005)
