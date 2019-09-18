import numpy as np
import matplotlib.pyplot as plt
from ThirdOrderSurface import ThirdOrderSurface

class GradientDescent:
	def __init__(self, eta, function, diffFunction, xDegree, eps = 0.01, ylim = 0.1, Nsteps = 100000, x0 = None, seed = 0):
		if x0 is not None:
			self.x = x0
		else:
			np.random.seed(seed)
			self.x = np.random.uniform(-10.0, 10.0, xDegree).reshape(-1, 1)
		self.eta = eta
		self.function = function
		self.diffFunction = diffFunction
		self.Nsteps = Nsteps
		self.eps = eps
		self.steps = -1
		self.xRecord = [self.x]
		self.yRecord = [function(self.x)]
		self.xSmallest = self.x
		self.ySmallest = function(self.x)
		self.smalestAt = 0
		self.ylim = ylim

	def  _evaluate(self):
		# print("-----------------------------------")
		# print(self.x)
		# print(self.diffFunction(self.x))
		# print(self.eta)
		# print("-----------------------------------")
		self.x = self.x - self.eta*(self.diffFunction(self.x))
		tmpy = self.function(self.x)
		self.xRecord.append(self.x)
		self.yRecord.append(tmpy)
		if tmpy < self.ySmallest:
			self.ySmallest = tmpy
			self.xSmallest = self.x
			return (True, self.x)
		return (False, self.x)

	def fit(self):
		pre_x = self.x
		cnt = 0
		for i in range(self.Nsteps):
			cnt += 1
			smallest, new_x = self._evaluate()
			if smallest: self.smalestAt = i

			reportSteps = self.Nsteps/100
			if i%reportSteps == 0:
				self.report()

			if np.linalg.norm(pre_x - new_x) <= self.eps:
				break
			elif self.ylim >= self.yRecord[-1]:
				break
			pre_x = new_x

		self.steps = cnt
		return self

	def recordData(self):
		return self.steps, self.smalestAt, np.array(self.xRecord), np.array(self.yRecord)

	def output(self):
		if self.steps == -1:
			self.fit()
		return self.xSmallest, self.ySmallest

	def report(self):
		print("-----------")
		print("xSmallest = \n{}".format(self.xSmallest))
		print("ySmallest = {}".format(self.ySmallest))
		print("total running steps: {}".format(self.steps))
		print("found smallest y at step: {}".format(self.smalestAt))
		print("-----------")


#Test
"""
x0 = -5.0
func = lambda x: np.add(np.multiply(x, x), 5*np.sin(x))
diffFunc = lambda x: 2*x + 5*np.cos(x)
gd = GradientDescent(0.1, func, diffFunc, 1, x0 = x0).fit()

x = np.linspace(-6, 6, 100)
y = func(x)

steps, xmin, ymin , xRecord = gd.output()

print(steps)
print((xmin, ymin))
yRecord = func(xRecord)

plt.figure()
plt.plot(x, y)
plt.scatter(xRecord, yRecord, c = 'r')
plt.scatter(xmin, ymin, c = 'y')
plt.xlim(-6, 6)
plt.ylim(-6, 30)
plt.show()
"""

#Test 3D surface
"""
x0 = np.array([-10.0, -10.0])
func = lambda x: np.sqrt(x.dot(x.T))
diffFunc = lambda x: x/func(x)

gd = GradientDescent(0.2, func, diffFunc, 2, x0 = x0, eps = 0.001, Nsteps = 10000).fit()

x = np.linspace(-6, 6, 100)
y = func(x)

xmin, ymin = gd.output()
steps, smallestStep, xRecord ,yRecord= gd.recordData()
xRecord = xRecord.reshape(-1, 2)

print(steps, smallestStep)
print((xmin, ymin))

surf = ThirdOrderSurface(
		(-10, 10, 100), (-10, 10, 100), (-1, 10),
		[lambda x: np.sqrt( np.add( np.power(x[:, 0], 2), np.power(x[:, 1], 2) ) )]
	)
surf.plot(False)

fig = plt.gcf()
ax = fig.gca()
ax.scatter(xRecord[:, 0].ravel(), xRecord[:, 1].ravel(), yRecord.ravel())
ax.scatter(xmin, ymin, c = 'r')
plt.show()
"""
