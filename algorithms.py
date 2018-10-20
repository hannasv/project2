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
import Costfunction
from gradientmethods import stochastic_gradient_descent, standard_gradient_descent, mini_batch_gradient_descent

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

class LogisticRegression(object):
    """Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Logistic cost function value in each epoch.
    """
    def __init__(self, eta, n_iter, random_state, key, lmd):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        #self.cost_ = cost # initialization of the cost_function
        self.key = key
        self.lmd = lmd

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the numb
          er of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """

        func = {
            'ols' = Costfunction.Cost_OLS,
            'ridge' = Costfunction.Cost_Ridge,
            'lasso' = Costfunction.Cost_Lasso,
        }

        #initialization weights to be random numbers from -0.7 to 0.7
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.7, size=1 + X.shape[1])
        #self.w_ = np.random.rand(-0.7, 0.7, )
        self.cost_ = []

        costfunc = func[key](self.eta, self.w, self.lmd)

        for i in range(self.n_iter):
            # Linar combination of weights and x'es
            input = np.dot(X, self.w_[1:]) + self.w_[0]
            costfunc.activiation(input, "sigmoid")

            output = self.activation(input)
            errors = (y - output)

            # standard gradient descent.
            self.w_[1:] += self.eta * X.T.dot(errors) # X.T@y is the gradient.
            self.w_[0] += self.eta * errors.sum() # bias

            cost_ = costfunc.calculate()
            self.cost_.append(cost_)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # if wwe update this one to take a activation function as input we can make it objectiorented.
    def activation(self, z):
        """Compute logistic sigmoid activation"""
        # np clip tvinger X@w to be in the range -250 to 250
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
