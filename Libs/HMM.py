import numpy as np
import ProbDistributor

class HMMProbCalculator():
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi.flatten()
        if len(A) != len(B) or len(A) != len(pi) or len(B) != len(pi):
            raise Exception("A, B, pi are not identical")
        self.NStates = len(pi)
        self.ObservationRange = len(B[0])
        self.NObservation = 0
        self.alpha = None

    def __str__(self):
        return str(self.alpha)

    def _fitInit(self, observation):
        self.NObservation = len(observation)
        if self.NObservation < 1: return False
        firstObs = observation[0]
        self.alpha = np.zeros(shape = (self.NStates, self.NObservation))
        self.alpha[:, 0] = self.pi*self.B[:, firstObs]
        return True

    def _fitInduction(self, observation):
        for i in range(1, self.NObservation):
            preProb = self.alpha[:, i - 1].reshape(1, -1)
            postProb = preProb.dot(A).reshape(-1, 1)
            obsProb = self.B[:, observation[i]].reshape(-1, 1)
            self.alpha[:, i] = (postProb * obsProb).flatten()

    def fit(self, observation):
        for obs in observation:
            if obs >= self.NStates: raise Exception("observation state invalid")
        if not self._fitInit(observation):
            raise Exception("Can't initialize HMM Probability Calculator for the observation")
        self._fitInduction(observation)
        return self

    def getFittingProb(self):
        if self.alpha is None: raise Exception("model haven't been fitted")
        return np.sum(self.alpha[:, self.NObservation - 1])

class HMMViterbiStatePredictor(HMMProbCalculator):
    def __init__(self, A, B, pi):
        super().__init__(A, B, pi)
        self.theta = None

    def __str__(self):
        alphaStr = str(self.alpha)
        thetaStr = str(self.theta)
        return "alpha:\n" + alphaStr + "\ntheta:\n" + thetaStr

    def _fitInit(self, observation):
        if not super()._fitInit(observation): return False
        self.theta = np.zeros(shape = (self.NStates, self.NObservation))
        return True

    def _fitInduction(self, observation):
        for i in range(1, self.NObservation):
            preProb = self.alpha[:, i - 1].reshape(-1, 1)
            tmpProb = preProb * self.A
            postProb = np.max(tmpProb, axis = 0).reshape(-1, 1)
            obsProb = self.B[:, observation[i]].reshape(-1, 1)
            self.alpha[:, i] = (postProb * obsProb).flatten()
            self.theta[:, i] = np.argmax(tmpProb, axis = 0).flatten()

    def getFittingProb(self):
        if self.alpha is None: raise Exception("model haven't been fitted")
        return np.max((self.alpha[:, self.NObservation - 1]).ravel())

    def getPath(self):
        if self.alpha is None or self.theta is None: raise Exception("model haven't been fitted")
        endPointIndex = np.argmax((self.alpha[:, self.NObservation - 1]).ravel())
        lst = [int(self.theta[endPointIndex, self.NObservation - 1])]
        for i in range(self.NObservation - 2, -1, -1):
            endPointIndex = lst[-1]
            lst.append(int(self.theta[endPointIndex, i]))
        reversed(lst)
        return lst


A = np.array(
    [
        [0.2, 0.4, 0.1, 0.3],
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.5, 0.2, 0.2]
    ])

B = np.array(
    [
        [0.3, 0.3, 0.4],
        [0.2, 0.1, 0.7],
        [0.4, 0.1, 0.5],
        [0.1, 0.6, 0.3]
    ])

pi = np.array([0.25, 0.25, 0.25, 0.25])

cal = HMMProbCalculator(A, B, pi).fit([0, 2, 1, 1, 1, 0, 0, 2, 2])
print(cal)
print(cal.getFittingProb())

viterbi = HMMViterbiStatePredictor(A, B, pi).fit([0, 2, 1, 1, 1, 0, 0, 2, 2])
print(viterbi)
print(viterbi.getFittingProb())
print(viterbi.getPath())