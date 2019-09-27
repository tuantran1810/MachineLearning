import numpy as np

class CircleGenerator():
    def __init__(self, center, R, std, seed = 0):
        self.R = R
        self.std = std
        self.center = center
        self.seed = seed

    def __randomNoise(self, N):
        np.random.seed(self.seed)
        return np.random.normal(0, self.std, (N, 2))

    def generate(self, N):
        np.random.seed(self.seed)
        R = self.R * np.ones((N, 2))
        phi = np.random.uniform(0, 2*np.pi, (N, 1))
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        cossin = np.hstack((cosPhi, sinPhi))
        noise = self.__randomNoise(N)
        return R * cossin + noise
