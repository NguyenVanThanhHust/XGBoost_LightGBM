"""
Link https://www.kaggle.com/grroverpr/gradient-boosting-simplified
"""

%matplotlib inline

import pandas as pd
import numpy as np
from IPython.display import display
from fastai.imports import *
from sklearn import metrics

class DecisionTree():
    def __init__(self, x, y, idxs = None, min_leaf = 2):
        if idxs is None:
            idxs = np.arange(len(y))
        self.x, self.y, self.min_leaf = x, y, idxs, min_leaf
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        