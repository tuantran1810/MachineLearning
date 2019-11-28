import numpy as np
from DecisionTree import SimpleDecisionTree

class SimpleRandomForest():
    def __init__(self, rowsRemove, colsRemove, NTrees, nonSplitPercent = 0):
        self.rowsRemove = rowsRemove
        self.colsRemove = colsRemove
        self.nonSplitPercent = nonSplitPercent
        self.NTrees = NTrees
        self.trees = []
        self.rCols = []

    def fit(self, data, target, attributes):
        for _ in range(self.NTrees):
            tmpData = data[:,:]
            tmpTarget = target[:,:]
            tmpAttributes = attributes[:]
            deletedRows = np.random.randint(len(tmpData), size = self.rowsRemove)
            deletedCols = np.random.randint(len(tmpData[0]), size = self.colsRemove)
            tmpData = np.delete(tmpData, deletedRows, axis = 0)
            tmpData = np.delete(tmpData, deletedCols, axis = 1)
            tmpTarget = np.delete(tmpTarget, deletedRows, axis = 0)
            tmpAttributes = np.delete(tmpAttributes, deletedCols)
            self.trees.append(SimpleDecisionTree(self.nonSplitPercent).fit(tmpData, tmpTarget, tmpAttributes))
            self.rCols.append(deletedCols)
        return self

    def predict(self, data, detail = False):
        predictions = []
        for i in range(len(self.trees)):
            tree = self.trees[i]
            mask = self.rCols[i]
            tmpData = np.delete(data, mask)
            predictions.append(tree.predict(tmpData))
        predictions = np.array(predictions)
        u, c = np.unique(predictions, return_counts = True)
        if detail:
            print(f"Prediction: {u}")
            print(f"Count: {c}")
        return u[np.argmax(c)]

    def __str__(self):
        result = ""
        for tree in self.trees:
            result += str(tree)
        return result
