import numpy as np
import math
from scipy.special import ndtr


class L2LSH:
    def __init__(self, N, d, r):
        # N = number of hashes
        # d = dimensionality
        # r = "bandwidth"
        self.N = N
        self.d = d
        self.r = r

        # set up the gaussian random projection vectors
        self.W = np.random.normal(size=(N, d))
        self.b = np.random.uniform(low=0, high=r, size=N)

    def hash(self, x):
        return np.floor((np.squeeze(np.dot(self.W, x)) + self.b) / self.r)


def P_L2(c, w):
    try:
        p = 1 - 2 * ndtr(-w / c) - 2.0 / (np.sqrt(2 * math.pi) * (w / c)) * (1 - np.exp(-0.5 * (w ** 2) / (c ** 2)))
        return p
    except:
        return 1


# to fix nans, p[np.where(c == 0)] = 1

def P_SRP(x, y):
    x_u = x / np.linalg.norm(x)
    y_u = y / np.linalg.norm(y)
    angle = np.arccos(np.clip(np.dot(x_u, y_u), -1.0, 1.0))
    return 1.0 - angle / np.pi


class SimHash:
    # multiple SRP hashes combined into a set of N hash codes
    def __init__(self, reps, d, p):
        # reps = number of hashes (reps)
        # d = dimensionality
        # p = "bandwidth" = number of hashes (projections) per hash code
        self.N = reps * p  # number of hash computations
        self.N_codes = reps  # number of output codes
        self.d = d
        self.p = p

        # set up the gaussian random projection vectors
        self.W = np.random.normal(size=(self.N, d))
        self.reverseW = np.matmul(np.linalg.inv(np.matmul(np.transpose(self.W), self.W)), np.transpose(self.W))
        # self.reverseW = self.W

        self.powersOfTwo = np.array([2 ** i for i in range(self.p)])

    def hash(self, x):
        # p is the number of concatenated hashes that go into each
        # of the final output hashes
        h = np.sign(np.dot(self.W, x))
        h = np.clip(h, 0, 1)
        h = np.reshape(h, (self.N_codes, self.p))
        return np.dot(h, self.powersOfTwo)

    def reverse_hash(self, h):
        xe = np.matmul(h, np.transpose(self.reverseW))
        return xe


class ORHash:
    def __init__(self, reps, d, p):
        self.rot_num = 3
        self.d = d
        self.p = p
        self.N = reps * p  # number of hash computations
        self.hadamard_order = int(np.ceil(np.log2(self.d)))
        self.piece_length = 2**self.hadamard_order
        self.piece_number = int(np.ceil(self.N/self.piece_length))
        self.real_N = self.piece_number*self.piece_length
        self.N_codes = reps  # number of output codes
        self.flip = np.sign(np.random.normal(size=(self.rot_num, self.real_N)))
        self.flip = np.where(self.flip > 0, 1, -1)
        self.powersOfTwo = np.array([2 ** i for i in range(self.p)])

    def hash(self, x):
        # print("x_shape", x.shape)
        expand_x = np.pad(x, (0, self.piece_length-x.shape[0]), 'constant')
        expand_x = np.tile(expand_x, self.piece_number)
        for rot_t in range(self.rot_num):
            expand_x = self.flip[rot_t, :]*expand_x.reshape(-1)
            s = 1
            while s < self.piece_length:
                expand_x = expand_x.reshape((self.piece_number, -1, 2, s))
                xp = np.sum(expand_x, axis=2)
                xm = expand_x[:, :, 0, :] - expand_x[:, :, 1, :]
                expand_x = np.stack((xp, xm), axis=2)
                s *= 2
        h = np.sign(expand_x).reshape(-1)[:self.N]
        h = np.clip(h, 0, 1)
        # print(h)
        h = np.reshape(h, (self.N_codes, self.p))
        return np.dot(h, self.powersOfTwo)

    def batch_hash(self, x):
        n_batch = x.shape[0]
        expand_x = np.pad(x, ((0, 0), (0, self.piece_length-x.shape[1])), 'constant')
        expand_x = np.tile(expand_x, self.piece_number)
        for rot_t in range(self.rot_num):
            expand_x = self.flip[rot_t, :]*expand_x.reshape([n_batch, -1])
            s = 1
            while s < self.piece_length:
                expand_x = expand_x.reshape([n_batch, self.piece_number, -1, 2, s])
                xp = np.sum(expand_x, axis=3)
                xm = expand_x[:, :, :, 0, :] - expand_x[:, :, :, 1, :]
                expand_x = np.stack((xp, xm), axis=3)
                s *= 2
        h = np.sign(expand_x).reshape([n_batch, -1])[:, :self.N]
        h = np.clip(h, 0, 1)
        # print(h)
        h = np.reshape(h, [n_batch, self.N_codes, self.p])
        return np.dot(h, self.powersOfTwo)

    def reverse_hash(self, h):
        expand_h = np.pad(h, (0, self.real_N - h.shape[0]), 'constant')
        for rot_t in range(2, -1, -1):
            s = 1
            while s < self.piece_length:
                expand_h = expand_h.reshape((self.piece_number, -1, 2, s))
                hp = np.sum(expand_h, axis=2)
                hm = expand_h[:, :, 0, :] - expand_h[:, :, 1, :]
                expand_h = np.stack((hp, hm), axis=2)
                s *= 2
            expand_h = self.flip[rot_t, :] * expand_h.reshape(-1)
        x = expand_h.reshape((-1, self.piece_length))
        x_mean = np.mean(x, axis=0)
        return x.reshape(-1)[:self.d], x_mean[:self.d]
