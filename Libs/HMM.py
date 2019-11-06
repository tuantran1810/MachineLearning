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
        # self.AGenerator = [x for ai in A: x = ProbDistributor.Distributor(ai)]
        # self.BGenerator = [x for bi in B: x = ProbDistributor.Distributor(bi)]
        # self.piGenerator = ProbDistributor.Distributor(pi)
        self.NObservation = 0
        self.alpha = None

    def _fitInit(self, observation):
        self.NObservation = len(observation)
        if self.NObservation < 1: return False
        firstObs = observation[0]
        self.alpha = np.zeros(shape = (self.NStates, self.NObservation))
        self.alpha[:, 0] = self.pi*self.B[:, firstObs]

    def _fitInduction(self, observation):
        for i in range(1, self.NObservation):
            preProb = self.alpha[:, i - 1].reshape(1, -1)
            postProb = preProb.dot(A).reshape(-1, 1)
            obsProb = self.B[:, observation[i]].reshape(-1, 1)
            self.alpha[:, i] = (postProb * obsProb).flatten()

    def fit(self, observation):
        for obs in observation:
            if obs >= self.NStates: raise Exception("observation state invalid")
        self._fitInit(observation)
        self._fitInduction(observation)
        return np.sum(self.alpha[:, self.NObservation - 1])

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

cal = HMMProbCalculator(A, B, pi).fit([0, 2, 1, 1])
print(cal)
