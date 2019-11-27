import numpy as np

class SimpleNode():
    def __init__(self):
        self.isLeaf = False
        self.leafValue = None
        self.nodeName = None
        self.dataArg = None
        self.nodes = {}

    def __entropy(self, cnt):
        s = np.sum(cnt)
        cnt = cnt / s
        entropy = 0
        log2cnt = np.log2(cnt)
        return -np.sum(cnt*log2cnt)

    def __elementEntropy(self, data, target):
        entropy = 0
        data = data.flatten()
        target = target.flatten()
        N = len(data)
        for d in np.unique(data):
            tmp = target[data == d]
            _, cnt = np.unique(tmp, return_counts = True)
            entropy += self.__entropy(cnt) * len(tmp) / N
        return entropy

    def fit(self, data, target, attributes):
        data = data[:,:]
        attributes = attributes[:]
        target = target[:,:]

        t_unique, cnt = np.unique(target, return_counts = True)
        if len(t_unique) == 1:
            self.isLeaf = True
            self.leafValue = t_unique[0]
            self.nodeName = "Leaf " + str(t_unique[0])
            return self

        targetEntropy = self.__entropy(cnt)
        D = len(data[0])

        infoGain = []
        for i in range(D):
            infoGain.append(targetEntropy - self.__elementEntropy(data[:, i], target))
        arg = np.argmax(infoGain)
        self.dataArg = arg

        self.nodeName = attributes[arg]
        nodeData = data[:, arg].flatten()
        data = np.delete(data, arg, 1)
        attributes = np.delete(attributes, arg)
        for d in np.unique(nodeData):
            tmp = nodeData == d
            newData = data[tmp]
            newTarget = target[tmp]
            self.nodes[d] = SimpleNode().fit(newData, newTarget, attributes)
        return self

    def printTree(self, depth = 0, key = None):
        result = ""
        if key is None:
            result += depth*'----' + self.nodeName + '\n'
        else:
            result += depth*'----' + key + ": " + self.nodeName + '\n'

        for k in self.nodes:
            result +=  self.nodes[k].printTree(depth + 1, k)
        return result

    def predict(self, data):
        data = data.flatten()
        if (self.isLeaf): return self.leafValue
        if data[self.dataArg] in self.nodes:
            return self.nodes[data[self.dataArg]].predict(data)
        else:
            raise Exception(f"Unrecognized attribute: {data[self.dataArg]}")

class SimpleDecisionTree():
    def __init__(self):
        self.root = None
        self.N = 0
        self.D = 0

    def fit(self, data, target, attributes):
        self.N = len(data)
        self.D = len(data[0])
        self.root = SimpleNode().fit(data, target, attributes)
        return self

    def predict(self, data):
        return self.root.predict(data)

    def __str__(self):
        return self.root.printTree()


# attributes = np.array(["Outlook", "Temperature", "Humidity", "Wind"])
# data = np.array([
#         ["Sunny",       "Hot",      "High",     "Weak"      ],
#         ["Sunny",       "Hot",      "High",     "Strong"    ],
#         ["Overcast",    "Hot",      "High",     "Weak"      ],
#         ["Rain",        "Mild",     "High",     "Weak"      ],
#         ["Rain",        "Cool",     "Normal",   "Weak"      ],
#         ["Rain",        "Cool",     "Normal",   "Strong"    ],
#         ["Overcast",    "Cool",     "Normal",   "Strong"    ],
#         ["Sunny",       "Mild",     "High",     "Weak"      ],
#         ["Sunny",       "Cool",     "Normal",   "Weak"      ],
#         ["Rain",        "Mild",     "Normal",   "Weak"      ],
#         ["Sunny",       "Mild",     "Normal",   "Strong"    ],
#         ["Overcast",    "Mild",     "High",     "Strong"    ],
#         ["Overcast",    "Hot",      "Normal",   "Weak"      ],
#         ["Rain",        "Mild",     "High",     "Strong"    ],
#     ])

# target = np.array(["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]).reshape(-1, 1)

# dt = SimpleDecisionTree().fit(data, target, attributes)
# print(dt)

# print(dt.predict(np.array(["Overcast", "Mild", "Normal", "Weak"])))
