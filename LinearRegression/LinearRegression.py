#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


class LinearRegressor:
    def __init__(self, X, t):
        self.X = X
        self.t = t
        self.w = None
        
    def fit(self):
        self.w = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.t)
        return self
    
    def predict(self, Xpred):
        return Xpred.dot(self.w)

