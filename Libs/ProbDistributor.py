import numpy as np

class Distributor():
    def __init__(self, distribution):
        tmp = distribution.flatten()
        self.distribution = tmp / np.sum(tmp)
        self.accDistribution = self.distribution
        for i in range(1, len(self.distribution)):
            self.accDistribution[i] += self.accDistribution[i - 1]
        self.N = len(self.accDistribution)

    def Generate(self):
        num = np.random.uniform()
        for i in range(0, self.N - 1):
            if num >= self.accDistribution[i] and num < self.accDistribution[i + 1]:
                return i + 1
        return 0

    def Prob(self, i):
        if i < self.N: return self.distribution[i]
        return 0

def GenerateDistributionMatrix(row, col):
    ret = np.zeros(row * col).reshape(row, col)
    for i in range(row):
        ret[i, :] = GenerateDistribution(col)
    return ret

def GenerateDistribution(N):
    acc = 0.0
    ret = np.zeros(N)
    for i in range(N - 1):
        gen = np.random.uniform(0, 1.0 - acc)
        acc += gen
        ret[i] = gen
    ret[-1] = 1.0 - acc
    return ret