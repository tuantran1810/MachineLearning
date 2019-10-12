import numpy as np
from numpy import linalg as LA
from cvxopt import matrix, solvers


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
        print(self.eigenValues)
        print(self.eigenVectors)
        return self
