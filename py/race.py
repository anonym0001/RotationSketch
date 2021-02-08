import numpy as np


def int2bin(int_hash, p=4):
    bin_hash_list = list()
    for i in range(p):
        bin_hash_list.append(np.bitwise_and(int_hash, 1 << i))
    bin_hash = np.array(bin_hash_list, dtype=np.int32)
    bin_hash = bin_hash.flatten('F')
    bin_hash = np.where(bin_hash == 0, -1, 1)
    return bin_hash


class RACE:
    def __init__(self, repetitions, hash_range, dtype=np.int32):
        self.dtype = dtype
        self.R = repetitions  # number of ACEs (rows) in the array
        self.W = hash_range  # range of each ACE (width of each row)
        self.counts = np.zeros((self.R, self.W), dtype=self.dtype)

    def add(self, hashvalues):
        for idx, hashvalue in enumerate(hashvalues):
            rehash = int(hashvalue)
            rehash = rehash % self.W
            self.counts[idx, rehash] += 1

    def batch_add(self, hashvalues):
        for idx, hashvalue in enumerate(np.transpose(hashvalues)):
            rehash = np.remainder(hashvalue, self.W).astype(int)
            bin_cnt = np.bincount(rehash)
            if bin_cnt.shape[0] < self.W:
                bin_cnt = np.pad(bin_cnt, [0, self.W - bin_cnt.shape[0]], 'constant')
            self.counts[idx, :] += bin_cnt

    def remove(self, hashvalues):
        for idx, hashvalue in enumerate(hashvalues):
            rehash = np.remainder(hashvalue, self.W)
            self.counts[idx, rehash] += -1

    def clear(self):
        self.counts = np.zeros((self.R, self.W), dtype=self.dtype)

    def query(self, hashvalues):
        mean = 0
        N = np.sum(self.counts) / self.R
        if N == 0:
            return 0
        for idx, hashvalue in enumerate(hashvalues):
            rehash = int(hashvalue)
            rehash = rehash % self.W
            mean = mean + self.counts[idx, rehash]
        return mean / (self.R * N)

    def batch_query(self, hashvalues):
        n_batch = hashvalues.shape[0]
        mean = np.zeros(n_batch)
        N = np.sum(self.counts) / self.R
        if N == 0:
            return 0
        for idx, hashvalue in enumerate(np.transpose(hashvalues)):
            rehash = np.remainder(hashvalue, self.W).astype(int)
            mean += self.counts[idx, rehash]
        return mean / (self.R * N)

    def find_min(self):
        raw_hash = np.argmin(self.counts, axis=1)
        return int2bin(raw_hash)

    def find_random_min(self, num):
        raw_hash = np.argmin(self.counts, axis=1)
        he = int2bin(raw_hash)
        hr = np.abs(np.random.normal(size=(num, he.shape[0])))
        hr = np.multiply(hr, he)
        return hr

    def print(self):
        for i, row in enumerate(self.counts):
            print(i, '| \t', end='')
            for thing in row:
                print(str(int(thing)).rjust(2), end='|')
            print('\n', end='')

    def counts(self):
        return self.counts

    def counts_lap(self, eps=0.0):
        if eps == 0:
            return self.counts
        else:
            noise = np.random.laplace(scale=1/eps, size=self.counts.shape)
            return np.round(self.counts + noise).astype(np.int32)

