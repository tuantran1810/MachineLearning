import numpy as np
from cvxopt import matrix, solvers

class SVM():
    def __init__(self, Xorig, torig):
        self.Xorig = Xorig
        self.torig = torig
        self.N = len(torig)
        self.Korig = len(Xorig[0])
        self.w = None

    def fit(self):
        raise Exception("fit() haven't been implemented!")

    def _classify(self, tpredict):
        for t in tpredict.ravel():
            if t > 0.0:
                yield 1
            else:
                yield -1

    def predict(self, Xpredict, raw = False):
        if self.w is None:
            raise Exception("w haven't been calculated, do fit() first")
        tpredict = Xpredict.dot(self.w)
        if raw: return tpredict
        return np.array([i for i in self._classify(tpredict)], dtype = int).reshape(-1, 1)

class PrimalSVM(SVM):
    def __init__(self, X, t):
        N = len(X)
        one = np.ones(N, dtype = float).reshape(-1, 1)
        self.X = np.hstack((X, one))

        K = len(self.X[0])
        tfinal = t[:, :]
        for _ in range(K - 1):
            tfinal = np.hstack((tfinal, t))
        self.t = tfinal
        self.K = K
        super().__init__(X, t)

    def fit(self):
        K = np.identity(self.K, dtype = float)
        K[self.K - 1][self.K - 1] = 0.0
        K = matrix(K)
        p = matrix(np.zeros(self.K, dtype = float).reshape(-1, 1))
        G = matrix(-(self.X * self.t))
        h = -np.ones(self.N, dtype = float).reshape(-1, 1)
        h = matrix(h)

        solvers.options['show_progress'] = False
        solultion = solvers.qp(K ,p ,G , h)
        self.w = np.array(solultion['x']).reshape(-1, 1)
        print("done fitting, w = \n{}".format(self.w))
        return self

    def raw_predict(self, Xpredict):
        N = len(Xpredict)
        Xpredict = np.hstack((Xpredict, np.ones(N, dtype = float).reshape(-1, 1)))
        return super().predict(Xpredict, True)

    def predict(self, Xpredict):
        N = len(Xpredict)
        Xpredict = np.hstack((Xpredict, np.ones(N, dtype = float).reshape(-1, 1)))
        return super().predict(Xpredict)

    def __findSupportVectorPoints(self):
        ypredict = super().predict(self.X, True)
        for i in range(len(ypredict)):
            num = np.asscalar(ypredict[i])
            if (num > -1.001 and num < -0.999) or (num < 1.001 and num > 0.999):
                yield self.X[i][0:-1]

    def supportVectorPoints(self):
        return np.array([point for point in self.__findSupportVectorPoints()])

class DualitySVM(SVM):
    def __init__(self, X, t):
        super().__init__(X, t)
        self.b = None

    def _calculate_w(self, alpha):
        w = np.zeros(self.Korig).reshape(-1, 1)
        atn = alpha*self.torig
        at = alpha*self.torig
        for _ in range(1, self.Korig):
            at = np.hstack((at, atn))

        tmp = at*self.Xorig
        return np.sum(tmp, axis = 0).reshape(-1, 1)

    def _calculate_b(self, alpha, w, t):
        for i in range(self.N):
            anum = np.asscalar(alpha[i])
            if anum > 0.01:
                return np.asscalar(t[i] - w.T.dot(self.Xorig[i].T))
        return None

    def fit(self):
        Kgram = self.Xorig.dot(self.Xorig.T)
        Y = self.torig.dot(self.torig.T)
        K = matrix(Kgram * Y)
        p = matrix(-np.ones(self.N).reshape(-1, 1))
        G = matrix(-np.identity(self.N))
        h = matrix(np.zeros(self.N).reshape(-1, 1))
        A = matrix(self.torig.reshape(1, -1))
        b = matrix(np.zeros((1, 1)))

        solvers.options['show_progress'] = False
        solultion = solvers.qp(K, p, G, h, A, b)
        alpha = np.array(solultion['x']).reshape(-1, 1)
        self.w = self._calculate_w(alpha)
        self.b = self._calculate_b(alpha, self.w, self.torig)
        if self.b is not None:
            print("done fitting, w = \n{}".format(self.w))
            print("b = {}\n".format(self.b))
        return self

    def predict(self, Xpredict, raw = False):
        Npredict = len(Xpredict)
        tmp = super().predict(Xpredict, True)
        tmp = tmp + self.b*(np.ones(Npredict).reshape(-1, 1))
        if raw:
            return tmp
        return np.array([i for i in super()._classify(tmp)], dtype = int).reshape(-1, 1)

    def __findSupportVectorPoints(self):
        ypredict = self.predict(self.Xorig, True)
        for i in range(len(ypredict)):
            num = np.asscalar(ypredict[i])
            if (num > -1.001 and num < -0.999) or (num < 1.001 and num > 0.999):
                yield self.Xorig[i]

    def supportVectorPoints(self):
        return np.array([point for point in self.__findSupportVectorPoints()])