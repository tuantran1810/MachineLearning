import numpy as np

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
        return "alpha matrix:\n" + str(self.alpha)

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
            postProb = preProb.dot(self.A).reshape(-1, 1)
            obsProb = self.B[:, observation[i]].reshape(-1, 1)
            self.alpha[:, i] = (postProb * obsProb).flatten()

    def fit(self, observation):
        observation = np.array(observation)
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

class HMMOptimizer():
    def __init__(self, A = None, B = None, pi = None, NStates = None, NObsStates = None):
        self.A = A
        self.B = B
        self.pi = pi
        self.NStates = NStates
        self.NObsStates = NObsStates
        if A is None and B is None and pi is None:
            self.A, self.B, self.pi = self.__initFromScratch(NStates, NObsStates)
        elif A is None or B is None or pi is None:
            raise Exception("cannot init the model!")
        else:
            self.NStates = len(self.A)
            self.NObsStates = len(self.B[0])
        self.A.reshape(NStates, -1)
        self.B.reshape(NStates, -1)
        self.pi.reshape(1, -1)
        self.NObservation = 0
        self.alpha = None
        self.beta = None
        self.theta = None
        self.eta = None

    def __str__(self):
        return ("A:\n" + str(self.A) + "\nB:\n" + str(self.B) + "\npi:\n" + str(self.pi))

    def __initFromScratch(self, NStates, NObsStates):
        if NStates is None or NObsStates is None:
            raise Exception("cannot init the model!")
        A = (np.ones(NStates * NStates) * (1.0 / NStates)).reshape(NStates, -1)
        B = (np.ones(NStates * NObsStates) * (1.0 / NObsStates)).reshape(NStates, -1)
        pi = (np.ones(NStates) * (1.0 / NStates)).reshape(1, -1)
        return A, B, pi

    def __fitInit(self, observation):
        obs = observation[0]
        self.alpha = np.zeros(shape = (self.NStates, self.NObservation))
        self.beta = np.zeros(shape = (self.NStates, self.NObservation))
        self.theta = np.zeros(shape = (self.NStates, self.NObservation))
        self.eta = np.zeros(shape = (self.NStates, self.NStates, self.NObservation - 1))
        self.alpha[:, 0] = self.pi * self.B[:, obs].ravel()
        self.beta[:, -1] = np.ones(self.NStates)

    def __fitInduction(self, observation):
        for t in range(1, self.NObservation):
            preProb = self.alpha[:, t - 1].reshape(1, -1)
            postProb = preProb.dot(A).reshape(-1, 1)
            obsProb = self.B[:, observation[t]].reshape(-1, 1)
            self.alpha[:, t] = (postProb * obsProb).flatten()
        for t in range(self.NObservation - 2, -1, -1):
            BmulBeta = (self.B[:, observation[t + 1]].ravel() * self.beta[:, t + 1].ravel()).reshape(-1, 1)
            self.beta[:, t] = self.A.dot(BmulBeta).flatten()

    def __fitThetaEta(self, observation):
        for t in range(self.NObservation - 1):
            alpha_t = self.alpha[:, t].reshape(-1, 1)
            beta_t1 = self.beta[:, t + 1].reshape(1, -1)
            tmp = alpha_t.dot(beta_t1) * self.A
            obs_t1 = observation[t + 1]
            bj = self.B[:, obs_t1].reshape(1, -1)
            eta_t = tmp * bj
            sumEta = np.sum(eta_t)
            eta_t = eta_t / sumEta
            self.eta[:, :, t] = eta_t
            self.theta[:, t] = np.sum(eta_t, axis = 1).ravel()
        self.theta[:, -1] = self.alpha[:, -1]

    def __fitUpdate(self, observation):
        self.pi = self.theta[:, 0].ravel()
        sum_eta = np.sum(self.eta, axis = 2)
        sum_theta = np.sum(self.theta, axis = 1)
        sum_theta_T_1 = sum_theta - self.theta[:, -1]
        self.A = (sum_eta.T / sum_theta_T_1).T
        for i in range(self.NObsStates):
            compare = (observation == i).reshape(1, -1)
            theta_obs_i = self.theta * compare
            sum_theta_obs_i = np.sum(theta_obs_i, axis = 1)
            self.B[:, i] = sum_theta_obs_i / sum_theta

    def __fitEpoch(self, observation):
        self.NObservation = len(observation)
        if self.NObservation < 1: raise Exception("cannot do fitting for the observation")
        self.__fitInit(observation)
        self.__fitInduction(observation)
        self.__fitThetaEta(observation)
        self.__fitUpdate(observation)

    def fit(self, observation):
        observation = np.array(observation)
        for i in range(50): self.__fitEpoch(observation)
        return self


# A = np.array(
#     [
#         [0.2, 0.4, 0.1, 0.3],
#         [0.1, 0.2, 0.3, 0.4],
#         [0.4, 0.3, 0.2, 0.1],
#         [0.1, 0.5, 0.2, 0.2]
#     ])

# B = np.array(
#     [
#         [0.3, 0.3, 0.4],
#         [0.2, 0.1, 0.7],
#         [0.4, 0.1, 0.5],
#         [0.1, 0.6, 0.3]
#     ])

# pi = np.array([0.25, 0.25, 0.25, 0.25])

# cal = HMMProbCalculator(A, B, pi).fit([0, 2, 1, 1, 1, 0, 0, 2, 2])
# print(cal)
# print(cal.getFittingProb())

# print("================================")
# viterbi = HMMViterbiStatePredictor(A, B, pi).fit([0, 2, 1, 1, 1, 0, 0, 2, 2])
# print(viterbi)
# print(viterbi.getFittingProb())
# print(viterbi.getPath())

# print("================================")
# opt = HMMOptimizer(NStates = 4, NObsStates = 3).fit([0, 2, 1, 1, 1, 0, 0, 2, 2])
# print(opt)
