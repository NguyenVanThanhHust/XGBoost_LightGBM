"""
Link original post: https://www.kaggle.com/avanwyk/a-lightgbm-overview
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
import multiprocessing
import json

mpl.style.use('seaborn')
np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load and pre-process data
from sklearn.preprocessing import StandardScaler as StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/creditcard.csv')
data.head(10)
data['NormalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis = 1)

positive_percentage = data[data['Class'] == 1].shape[0]/data.shape[0] * 100
print("{:.2f}% of the data are positive examples (highly skewed).".format(positive_percentage))
# 0.17% of the data are positive examples (highly skewed).

X = data.drop('Class', axis=1)
y = data['Class']
# Get data and corresponding label
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

# Wrap our training and validataion sets in LightGBM Datasets
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data = False)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)

# For Basic usage
number_core = multiprocessing.cpu_count()

core_params = {
    'boosting_type' : 'gbdt', # GBM type: gradient boosted decision tree, rf (random forest), dart, goss
    'objective' : 'binary', # the optimization object : binary, regression, multiclass, xentropy
    'learning_rate' : 0.05, # the gradient descent learning / shrinkage rate which controns the step size.
    'num_leaves' : 31, # number of leaves in one tree
    'nthread' : number_core, # number of threads to use for LightGBM 
    'metric': 'auc' # an additional metric to calculate during validation: area under curve (auc)
}

"""
- train a Gradient Boosted Decision Tree using LightGBM. 
- wrap the training call in a function that trains the GBDT, plots the results of the training for us and returns the GBM and the validation results per iteration.
"""
    
    
def train_gbm(params, training_set, validation_set, init_gbm=None, boost_rounds=100, early_stopping_rounds=0, metric='auc'):
    evals_result = {} 

    gbm = lgb.train(params, # parameter dict to use
                    training_set,
                    init_model=init_gbm, # initial model to use, for continuous training.
                    num_boost_round=boost_rounds, # the boosting rounds or number of iterations.
                    early_stopping_rounds=early_stopping_rounds, # early stopping iterations.
                    # stop training if *no* metric improves on *any* validation data.
                    valid_sets=validation_set,
                    evals_result=evals_result, # dict to store evaluation results in.
                    verbose_eval=False) # print evaluations during training.
    
    y_true = validation_set.label
    y_pred = gbm.predict(validation_set.data)
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.title("ROC Curve. Area under Curve: {:.3f}".format(roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    img = plt.plot(fpr, tpr, 'r')
    plt.savefig('ROC Curve.png')
    return gbm, evals_result

advanced_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    
    'learning_rate': 0.01,
    'num_leaves': 41, # more leaves increases accuracy, but may lead to overfitting.
    
    'max_depth': 5, # the maximum tree depth. Shallower trees reduce overfitting.
    'min_split_gain': 0, # minimal loss gain to perform a split
    'min_child_samples': 21, # or min_data_in_leaf: specifies the minimum samples per leaf node.
    'min_child_weight': 5, # minimal sum hessian in one leaf. Controls overfitting.
    
    'lambda_l1': 0.5, # L1 regularization
    'lambda_l2': 0.5, # L2 regularization
    
    'feature_fraction': 0.5, # randomly select a fraction of the features before building each tree.
    # Speeds up training and controls overfitting.
    'bagging_fraction': 0.5, # allows for bagging or subsampling of data to speed up training.
    'bagging_freq': 0, # perform bagging on every Kth iteration, disabled if 0.
    
    'scale_pos_weight': 99, # add a weight to the positive class examples (compensates for imbalance).
    
    'subsample_for_bin': 200000, # amount of data to sample to determine histogram bins
    'max_bin': 1000, # the maximum number of bins to bucket feature values in.
    # LightGBM autocompresses memory based on this value. Larger bins improves accuracy.
    
    'nthread': number_core, # number of threads to use for LightGBM, best set to number of actual cores.
}

basic_model, evals = train_gbm(core_params, lgb_train, lgb_val)
basic_model.save_model('basic_model.txt')
basic_model_json = basic_model.dump_model()
with open('basic_model.json', 'w+') as f:
    json.dump(basic_model_json, f, indent=4)
    
advanced_model, evals = train_gbm(advanced_params, lgb_train, lgb_val, boost_rounds=500)
advanced_model.save_model('advanced_model.txt')
advanced_model_json = advanced_model.dump_model()
with open('advanced_model.json', 'w+') as f:
    json.dump(advanced_model_json, f, indent=4)







