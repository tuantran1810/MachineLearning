import numpy as np
from SVM import SVM
from cvxopt import matrix, solvers

class PrimalSoftMarginSVM(SVM):
    def __init__(self, X, t, C):
        self.Corig = C
        self.b = None
        self.xi = None
        super().__init__(X, t)

    def fit(self):
        dim = self.N + self.Korig + 1
        K = np.identity(dim)
        for i in range(self.Korig, dim):
            K[i][i] = 0
        K = matrix(K)

        z = np.zeros(self.Korig + 1).reshape(-1, 1)
        c = self.Corig * np.ones(self.N).reshape(-1, 1)
        p = np.vstack((z, c))
        p = matrix(p)

        G1iden = np.identity(self.N)
        G1xtmp = np.hstack((self.Xorig, np.ones(self.N).reshape(-1, 1)))
        G1ttmp = self.torig[:, :]
        for _ in range(0, self.Korig):
            G1ttmp = np.hstack((G1ttmp, self.torig))
        G1 = G1xtmp * G1ttmp
        G1 = -np.hstack((G1, G1iden))

        G2zeros = np.zeros((self.Korig + 1) * self.N).reshape(self.N, -1)
        G2ones = np.identity(self.N)
        G2 = -np.hstack((G2zeros, G2ones))

        G = matrix(np.vstack((G1, G2)))
        h1 = -np.ones(self.N).reshape(-1, 1)
        h2 = np.zeros(self.N).reshape(-1, 1)
        h = matrix(np.vstack((h1, h2)))

        solvers.options['show_progress'] = False
        solultion = solvers.qp(K, p, G, h)
        final = np.array(solultion['x']).flatten()
        self.w = final[0 : self.Korig].reshape(-1, 1)
        self.b = final[self.Korig].item()
        self.xi = final[self.Korig + 1:].reshape(-1, 1)
        print("done fitting, w = \n{}\nb = {}\nxi = \n{}".format(self.w, self.b, self.xi))
        return self

    def __predict(self, Xpredict, raw = False):
        tpredict = super().predict(Xpredict, raw = True)
        tpredict += self.b * np.ones(len(Xpredict)).reshape(-1, 1)
        if raw: return tpredict
        return np.array([i for i in self._classify(tpredict)], dtype = int).reshape(-1, 1)

    def __findSupportVectorPoints(self):
        ypredict = self.__predict(self.Xorig, True)
        for i in range(len(ypredict)):
            num = ypredict[i].item()
            if (num > -1.001 and num < -0.999) or (num < 1.001 and num > 0.999):
                yield self.Xorig[i]

    def predict(self, Xpredict):
        return self.__predict(Xpredict)

    def supportVectorPoints(self):
        return np.array([point for point in self.__findSupportVectorPoints()])
