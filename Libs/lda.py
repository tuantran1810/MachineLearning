import numpy as np

class LDA():
    def __init__(self):
        self.X = None
        self.t = None
        self.D = 0
        self.N = 0
        self.totalCentralPoint = None
        self.central = {}
        self.cov = {}
        self.classMatrix = {}
        self.count = None
        self.classes = None
        self.w = None

    def __getCentralPoint(self, classNum):
        classPointMask = (self.t == classNum).reshape(-1, 1)
        point = np.sum(self.X * classPointMask, axis = 0)
        point = point / self.count[classNum]
        return point.reshape(1, -1)

    def __splitClasses(self, X, t):
        tmp_t = t.ravel()
        for i in range(len(t)):
            classNum = tmp_t[i]
            if classNum in self.classMatrix:
                self.classMatrix[classNum] = np.vstack((self.classMatrix[classNum], X[i, :])).reshape(-1, self.D)
            else:
                self.classMatrix[classNum] = X[i, :].reshape(-1, self.D)

    def __calculateClassCovariance(self):
        for c in self.classes:
            classMatrix = self.classMatrix[c]
            centralPoint = self.central[c]
            tmp = (classMatrix - centralPoint).reshape(-1, self.D)
            self.cov[c] = tmp.T.dot(tmp)

    def fit(self, X, t):
        self.X = X
        self.t = t
        self.D = len(X[0])
        self.N = len(t)
        self.classes, self.count = np.unique(t, return_counts=True)
        self.__splitClasses(X, t)

        for c in self.classes:
            self.central[c] = self.__getCentralPoint(c)
        self.totalCentralPoint = (np.sum(X, axis = 0) / self.N).reshape(1, -1)
        self.__calculateClassCovariance()
        return self

    def getw(self):
        Sw = np.zeros(self.D * self.D).reshape(self.D, self.D)
        Sb = np.zeros(self.D * self.D).reshape(self.D, self.D)
        for c in self.classes:
            Sw += self.cov[c].reshape(self.D, self.D)
            tmp = self.central[c] - self.totalCentralPoint
            cnt = self.count[c]
            Sb += tmp.T.dot(tmp)*cnt
        eigVal, eigVec = np.linalg.eigh(np.linalg.inv(Sw).dot(Sb))
        pairs = [(np.abs(eigVal[i]), eigVec[i]) for i in range(len(eigVal))]
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        wMatrix = pairs[0][1].reshape(-1, 1)
        for i in range(1, len(pairs)):
            wMatrix = np.hstack((wMatrix, pairs[i][1].reshape(-1, 1)))
        return wMatrix

class TwoClassesLDA(LDA):
    def __init__(self):
        super().__init__()

    def fit(self, X, t):
        super().fit(X, t)
        if len(self.classes) != 2:
            print(f"invalid input, number of classes is not 2: {len(self.classes)}")
        return self

    def getw(self):
        if len(self.classes) != 2:
            print(f"invalid input, number of classes is not 2: {len(self.classes)}")
            return None
        c1 = self.classes[0]
        c2 = self.classes[1]
        Sw = (self.cov[c1] + self.cov[c2]).reshape(self.D, self.D)
        m12 = (self.central[c2] - self.central[c1]).reshape(self.D, 1)
        tmpw = np.linalg.inv(Sw).dot(m12).flatten()
        return tmpw / np.linalg.norm(tmpw)
