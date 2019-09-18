#!/usr/bin/env python
# coding: utf-8

import numpy as np
import BasisFunction as bf
import GradientDescent as gd

class LinearRegressor:
    def __init__(self, basisFuncs, Xorig, t):
        self.Xorig = Xorig
        self.t = t
        self.w = None
        self.yPred = None
        self.basisFuncs = basisFuncs
        
    def fit(self):
        X = bf.BaseBasicFunction(self.Xorig, self.basisFuncs).generate()
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.t)
        return self
    
    def predict(self, Xpred):
        X = bf.BaseBasicFunction(Xpred, self.basisFuncs).generate()
        self.yPred = X.dot(self.w)
        return self.yPred

    def meanSquareError(self, tPred):
        N = float(len(tPred.ravel()))
        e = np.subtract(tPred, self.yPred)
        return np.asscalar((e.T.dot(e)/N).ravel())
        
class RidgeLinearRegressor(LinearRegressor):
    def __init__(self, basisFuncs, Xorig, t, lamb = 0):
        self.lamb = lamb
        super().__init__(basisFuncs, Xorig, t)
        
    def fit(self):
        X = bf.BaseBasicFunction(self.Xorig, self.basisFuncs).generate()
        lamb = self.lamb * np.identity(len(X.T))
        tmp = np.linalg.inv(np.add(lamb, X.T.dot(X)))
        self.w = tmp.dot(X.T).dot(self.t)
        return self
    
class LassoLinearRegression(LinearRegressor):
    def __init__(self, basisFuncs, Xorig, t, lamb = 0.1):
        self.lamb = lamb
        self.grad_eta = 0.1
        self.grad_eps = 0.01
        self.grad_steps = 100000
        self.grad_ylim = 0.1
        super().__init__(basisFuncs, Xorig, t)
        self.X = bf.BaseBasicFunction(self.Xorig, self.basisFuncs).generate()
        self.grad = None
        self.w = None

    def _getGrad(self):
        k = len(self.X[0])
        # w0 = np.zeros(k).reshape(-1, 1)
        w0 = np.random.uniform(-1, 1, k).reshape(-1, 1)
        
        # difffunc = lambda w: np.add(np.subtract(self.X.T.dot(self.X).dot(w), self.X.T.dot(self.t)), 0.5*self.lamb*np.sign(w))
        difffunc = lambda w: -2.0*(self.X.T).dot(self.t) + 2.0*(self.X.T).dot(self.X).dot(w) + self.lamb*np.sign(w)

        func = lambda w: ((self.t - self.X.dot(w)).T).dot(self.t - self.X.dot(w)) + self.lamb*np.sum(np.absolute(w))

        return gd.GradientDescent(self.grad_eta, func, difffunc, k, x0 = w0, eps = self.grad_eps, Nsteps = self.grad_steps, ylim = self.grad_ylim)

    def setGradParam(self, eta = 0.1, eps = 0.01, steps = 100000, ylim = 0.1):
        self.grad_eta = eta
        self.grad_eps = eps
        self.grad_steps = steps
        self.grad_ylim = ylim

    def fit(self):
        self.grad = self._getGrad().fit()
        tmpw, _ = self.grad.output()
        self.w = tmpw.reshape(-1, 1)
        return self

    def reportGradStatus(self):
        print(self.grad.yRecord[self.grad.steps - 10: self.grad.steps])
        self.grad.report()
