'''
Created on 11.07.2015

@author: Gabriel Krummenacher
@author: Matthew Rocklin
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask
from sklearn.svm import SVC
from sklearn import cross_validation
from dask.compose import daskify, value

D = daskify

def train_test(reg_param, train_idx, test_idx, model_sel_idx, X, y):
    svm = SVC(C=reg_param)
    svm.fit(X[model_sel_idx[train_idx], :], y[model_sel_idx[train_idx]])
    return svm.score(X[model_sel_idx[test_idx], :], y[model_sel_idx[test_idx]])


def mean(*args):
    return np.mean(args)

def argmax(*args):
    return max(enumerate(args), key=lambda (i, x): x)[0]

def sum2(*args):
    return sum(args)

d = 10
n = 100
y = value(np.sign(np.random.randn(n)), name='u')
X = value(np.random.randn(n, d), name='X')
reg_params = np.logspace(-2, 2, 5)
n_folds = 4

kf_test = cross_validation.KFold(n, n_folds=n_folds)
score_params = list()
test_scores = []
for model_sel_idx, test_idx in kf_test:
    nn = len(model_sel_idx)
    model_sel_idx = value(model_sel_idx)
    test_idx = value(test_idx)
    for reg_param in reg_params:
        score = 0
        kf = cross_validation.KFold(nn, n_folds=n_folds)
        scores = [D(train_test)(reg_param, train_idx, val_idx, model_sel_idx, X, y)
                  for train_idx, val_idx in kf]
        score = D(sum2)(*scores) / n_folds
        score_params.append((score, reg_param))

    scores, params = list(zip(*score_params))
    best_index = D(argmax)(*scores)
    getitem = lambda a, b: a[b]
    best_param = D(getitem)(params, best_index)

    test_scores += [D(train_test)(best_param, model_sel_idx, test_idx,
                                  np.arange(n), X, y)]

test_score = D(sum2)(*test_scores) / n_folds

print test_score.compute()
