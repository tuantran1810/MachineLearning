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
    def __init__(self, basisFuncs, Xorig, t, grad, lamb = 0):
        self.lamb = lamb
        self.grad_eta = 0.1
        self.grad_eps = 0.01
        super().__init__(basisFuncs, Xorig, t)
        self.X = bf.BaseBasicFunction(self.Xorig, self.basisFuncs).generate()
        self.grad = None
        self.w = None

    def _getGrad(self):
        k = len(self.X[0])
        w0 = np.zeros(k).reshape(-1, 1)
        
        difffunc = lambda w: 
            self.X.T.dot(self.X).dot(w) - self.X.T.dot(self.t) + 0.5*self.lamb*np.sign(w)

        func = lambda w:
            np.subtract(self.t, w.T.dot(self.X)).T.dot(np.subtract(self.t, w.T.dot(self.X))) + 0.5*self.lamb*np.absolute(w)

        return gd.GradientDescent(self.grad_eta, func, difffunc, k, x0 = w0, eps = self.grad_eps)

    def setGradParam(self, eta, eps):
        self.grad_eta = eta
        self.grad_eps = eps

    def fit(self):
        self.grad = self._getGrad().fit()
        tmpw, _ = self.grad.output()
        self.w = tmpw.reshape(-1, 1)
        return self

    def reportGradStatus(self):
        self.grad.report()
