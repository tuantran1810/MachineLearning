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
