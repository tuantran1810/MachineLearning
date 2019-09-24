#!/usr/bin/env python
# coding: utf-8

import numpy as np

class utils:
    
    @staticmethod
    def colPower(x, col, deg):
        N = len(x)
        return np.power(x[:, col].reshape(N, 1), deg)

class BaseBasicFunction:
    def __init__(self, x, featureFunc): #x: N rows, K features
        self.x = x
        self.totalDegree = len(featureFunc) + 1
        self.N = len(x)
        self.K = len(x[0])
        self.featureFunc = featureFunc

    def _evaluate(self, degree):
        if degree == 0:
            return np.ones((self.N ,1))
        elif degree - 1 < len(self.featureFunc):
            return self.featureFunc[degree - 1](self.x)
        
        raise Exception('no such basic function at degree {}'.format(degree))
        
    def generate(self, genDummy = True):
        start = 1
        if genDummy: start = 0
        stack = self._evaluate(start)
        for i in range(start + 1, self.totalDegree):
            stack = np.column_stack((stack, self._evaluate(i)))
        return stack
