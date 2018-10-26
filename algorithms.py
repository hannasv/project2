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
    n_iter : int (defalt 50)
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
     lmd: penalty


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Logistic cost function value in each epoch.
    """
    def __init__(self, eta, random_state, key, n_iter = 100, batch_size = 10, lmd = 0, tolerance=1e-14):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.key = key
        self.lmd = lmd
        self.w_ = None
        self.tol = tolerance

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
            'ols' : Costfunctions.Cost_OLS,
            'ridge' : Costfunctions.Cost_Ridge,
            'lasso' : Costfunctions.Cost_Lasso,
        }

        #initialization weights to be random numbers from -0.7 to 0.7
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.7, size=1 + X.shape[1])
        #self.w_ = np.random.rand(-0.7, 0.7, )
        self.cost_ = []
        costfunc = func[self.key](self.eta, self.lmd)
        max_iter = self.n_iter
        i = 0
        #cost = 1
        while (i < max_iter):# and cost >= self.tol
            # Computing the linar combination of x'es and weights.
            net_input = np.dot(X, self.w_[1:]) + self.w_[0]
            output = costfunc.activation(net_input, "sigmoid")
            errors = costfunc.r(y) # calculating the residuals (y-p)
            # calculating the gradient of this particular costfunction
            gradient = costfunc.grad(X, self.w_, errors)
            self.descent_method(errors, gradient, "steepest")
            cost = costfunc.calculate(X, y, self.w_)
            self.cost_.append(cost)
            i=i+1
        return self

    def descent_method(self, errors, grad, key = "steepest"):
        # costfunc.grad(errors)
        if (key == "steepest"):
            self.w_[1:] += self.eta * grad
            self.w_[0] += self.eta * errors.sum() # bias

        elif(key == "sgd"):
            #self.beta = np.random.randn(2, 1)
            for epoch in range(self.n_epochs):
                for i in range(self.batch_size):
                    random_index = np.random.randint(self.batch_size)
                    xi = X[random_index:random_index + 1]
                    yi = y[random_index:random_index + 1]

                    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                    #eta = self.learning_schedule(epoch * self.m + i)
                    theta = theta - self.eta * gradients
            print("theta from own sdg" + str(theta))

        else:
            print("Unvalid keyword, use: steepest or sgd.")

    def predict(self, X):
        """Return class label after unit step"""
        net_input = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(net_input >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    def learning_schedule(t):
        t0, t1 = 5, 50
        return t0 / (t + t1)
