#!/usr/bin/env python
# coding: utf-8

import BasisFunction as bf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class ThirdOrderSurface:
    def __init__(self, xrange: tuple, yrange: tuple, zrange: tuple, zfunction):
        self.xrange = xrange[:2]
        self.yrange = yrange[:2]
        self.zrange = zrange[:2]
        x = np.linspace(xrange[0], xrange[1], xrange[2])
        y = np.linspace(yrange[0], yrange[1], yrange[2])
        self.x, self.y = np.meshgrid(x, y)
        self.x = self.x.flatten()
        self.y = self.y.flatten()
        self.zfunction = zfunction

        combineXY = np.column_stack((self.x.reshape(-1, 1), self.y.reshape(-1, 1)))
        zfunc = bf.BaseBasicFunction(combineXY, zfunction)
        self.z = zfunc.generate(False).flatten()

    def plot(self, newfig: bool):
        fig = plt.gcf()
        if newfig:
            fig = plt.figure()

        ax = fig.gca(projection = '3d')

        surf = ax.plot_trisurf(
                                self.x,
                                self.y,
                                self.z,
                                cmap=cm.coolwarm,
                                linewidth=0,
                                antialiased=False)

        ax.set_xlim(self.xrange[0], self.xrange[1])
        ax.set_ylim(self.yrange[0], self.yrange[1])
        ax.set_zlim(self.zrange[0], self.zrange[1])

        fig.colorbar(surf, shrink=0.5, aspect=5)

    def genGaussianNoisyPoints(self, N, variance, seed = 0):
        if self.zfunction is None:
            raise Exception('no such z function')
        np.random.seed(seed)
        x = np.random.uniform(self.xrange[0], self.xrange[1], N)
        y = np.random.uniform(self.yrange[0], self.yrange[1], N)
        combineXY = np.column_stack((x.reshape(-1, 1), y.reshape(-1, 1)))
        zfunc = bf.BaseBasicFunction(combineXY, self.zfunction)
        z = zfunc.generate(False).flatten()

        noise = np.random.normal(0, np.sqrt(variance))
        z = np.add(z, noise)
        return (combineXY, z.reshape(-1, 1))
