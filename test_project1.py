import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import stats
import scipy.stats as st
import algorithms
import model_selection0
from model_comparison0 import model_comparison0
from utils import generateDesignmatrix, franke_function, train_test_split, bootstrap,  ci
import unittest.mock as mock
import numpy
import utils

np.random.seed(1000)
m = 30
x = np.random.rand(m, )
y = np.random.rand(m, )
z = franke_function(x, y)

def convert(X):
    """ Converting the array og columns to a array of rows so its on the same form as the designmatrix. """
    m = len(X)
    n = len(X[0])
    new = np.zeros((n,m))
    for i in range(m): #3
        for j in range(n): # 30
            new[j][i] = X[i][j]
    return new

def test_design():
    """ Checking if the designmatrix works for all polynomals up to degree p=5  """
    for p in range(1,6):
        if (p==1):
            X = [np.ones(len(x)), x, y]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

        elif(p==2):
            X = [np.ones(len(x)), x, y, x**2, x*y, y**2]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

        elif(p==3):
            X = [np.ones(len(x)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

        elif(p==4):
            X = [np.ones(len(x)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

        elif(p==5):
            X = [np.ones(len(x)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4,
                x**5, y*x**4, x**3*y**2, x**2*y**3, x*y**4, y**5]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

def test_mse():
    assert abs(mean_squared_error(x,y) -utils.mean_squared_error(x, y)) < 1e-8

def test_r2():
    assert abs( r2_score(x, y) -  utils.r2_score(x, y) ) < 1e-8

p = 2
X = generateDesignmatrix(p,x,y)

def test_ols():
    """  Testing to see if our algorithms compute approximetly the same betas as scikit does.   """
    our_ols = algorithms.OLS()
    our_ols.fit(X,z)
    our_betas = our_ols.coef_

    # Create linear regression object
    scikit_ols = LinearRegression(fit_intercept=False)
    # Train the model using the training sets
    scikit_ols.fit(X, z)
    assert np.all(abs(our_betas - scikit_ols.coef_[:])<1e8)


def test_ridge():
    """  Testing to see if our algorithms compute approximetly the same betas as scikit does.   """
    our_ridge = algorithms.Ridge(lmd = 0.1)
    our_ridge.fit(X,z)
    our_betas = our_ridge.coef_
    scikit_ridge=linear_model.RidgeCV(alphas=[0.1], fit_intercept=False)
    scikit_ridge.fit(X, z)
    assert  np.all(abs( our_betas == scikit_ridge.coef_[:])<1e8)


def test_bootstrap():
    with mock.patch("numpy.random.randint", return_value=np.arange(len(x))):
        X_subset, z_subset = bootstrap(X, z, 1)
        assert (np.allclose(X_subset, X) and np.allclose(z, z_subset))

def test_split():
    n = int(len(x)*0.8)
    with mock.patch("numpy.random.choice", return_value=np.arange(n)):
        X_train, X_test, z_train, z_test = train_test_split(X, z, split_size=0.2, random_state=1)
        print(np.shape(X))
        print("--------------")
        print(np.shape(X_train.tolist()+X_test.tolist()))
        assert (np.allclose(X_train.tolist()+X_test.tolist(), X) and np.allclose(z_train.tolist()+ z_test.tolist(), z  ))

def test_variance():
    assert utils.variance(x) == (x.var(ddof=1))

def test_ci():
    n = len(x)
    mu = np.sum(x)/n
    assert utils.ci(x) == st.t.interval(0.95, n-1, loc=mu, scale=st.sem(x))
