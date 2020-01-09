import numpy as np
from numpy import linalg as LA

class PCA():
    def __init__(self, X):
        self.X = X
        self.D = len(X[0])
        self.N = len(X)
        self.Xmean = None
        self.eigenVectors = None
        self.eigenValues = None

    def fit(self):
        Xmean = np.mean(self.X, axis = 0).reshape(1, -1)
        one = np.ones(self.N).reshape(-1, 1)
        Xmean = one.dot(Xmean)
        Xvar = self.X - Xmean
        S = (Xvar.T.dot(Xvar))/self.N
        self.Xmean = Xmean[0]
        self.eigenValues, self.eigenVectors = LA.eig(S)
        return self

    def getw(self):
        pairs = [(np.abs(self.eigenValues[i]), self.eigenVectors[i]) for i in range(len(self.eigenValues))]
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        wMatrix = pairs[0][1].reshape(-1, 1)
        for i in range(1, len(pairs)):
            wMatrix = np.hstack((wMatrix, pairs[i][1].reshape(-1, 1)))
        return wMatrix
