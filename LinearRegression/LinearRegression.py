#!/usr/bin/env python
# coding: utf-8

import numpy as np
import BasisFunction as bf

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
    