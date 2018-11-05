# -*- coding: utf-8 -*-
#
# algorithms.py
#
# The module is part of model_comparison.
#

"""
Representations of algorithms.Cat
"""

__author__ = 'Hanna Svennevik', 'Paulina Tedesco'
__email__ = 'hanna.svennevik@fys.uio.no', 'paulinatedesco@gmail.com'


from scipy import linalg
import scipy as sp
from sklearn import linear_model
import numpy as np
import Costfunctions

class OLS:
    """The ordinary least squares algorithm"""
    #import numpy as np

    def __init__(self, lmd=0, random_state=None):

        self.random_state = random_state
        self.lmd = lmd

        # NOTE: Variable set with fit method.
        self.coef_ = None

    def fit(self, X, y):
        """Train the model"""
        #self.coef_ = sp.linalg.inv(X.T @ X)@ X.T @ y
        #self.coef_ = np.linalg.pinv(X.T @ X)@ X.T @ y
        #u,s,v = svd(X)
        #self.coef_ = v@np.linalg.pinv(s)@u.T@y

        u,s,vh = np.linalg.svd(X, full_matrices = False)
        #vh is v transpose
        s = np.identity(len(s))*s
        self.coef_= vh.T@np.linalg.pinv(s)@u.T@y

    def predict(self, X):
        """Aggregate model predictions """
        return X @ self.coef_

               #print(y)
        #term1 =  np.dot(X.T, y)
        #term2 = np.dot(self.lmd*np.identity(len(y)), np.dot(X.T, y))
        #print(term2.shape)
class Ridge:
    """The Ridge algorithm."""
    #import numpy as np

    def __init__(self, lmd, random_state=None):
        self.lmd = lmd
        self.random_state = random_state
        # NOTE: Variable set whith fit method.
        self.coef_ = None

    def fit(self, X, y):
        """Train the model."""
        #self.coef_ = linalg.inv(X.T @ X + self.lmd * np.identity(X.shape[1])) @ X.T @ y
        #u,s,v = svd(X)

        #self.coef_ = X.T@y@np.linalg.inv(X.T@X + self.lmd*np.identity(X.shape[1]))
        self.coef_ = linalg.pinv(X.T @ X + self.lmd * np.identity(X.shape[1])) @ X.T @ y
        #u,s,vh = np.linalg.svd(X, full_matrices = False)
        #s = np.identity(len(s))*s
        #self.coef_ = X.T@y@np.linalg.inv(X.T@X + self.lmd*np.identity(X.shape[1]))

    def predict(self, X):
        """Aggregate model predictions."""
        return X @ self.coef_


class Lasso:
    """The LASSO algorithm."""
    # when the first column is 1 the use fit_intercept = True.

    def __init__(self, lmd, random_state=None):
        self.lmd = lmd
        self.random_state = random_state
        self.model = None
        self.coef_ = None

    def fit(self, X, y):
        """Train the model."""
        self.model = linear_model.Lasso(self.lmd)
        self.model.fit(X, y)
        self.coef_ = self.model.coef_

    def predict(self, X):
        """Aggregate model predictions."""
        return self.model.predict(X)
