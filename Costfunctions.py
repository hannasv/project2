# Would prefer to make this an abstract class.
import numpy as np

class Costfunctions:
    def __init__(self, eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,
        self.p = None

    def r(self, y): # resuduals.
        return y-self.p

    def update_weights(self, new):
        self.w = new

    def activation(self, Xw, key):
        if (key == "sigmoid"):
            self.p = 1. / (1. + np.exp(-np.clip(Xw, -250, 250)))
            return self.p
        elif(key == "elu"):
            return 0

class Cost_OLS(Costfunctions):

    def __init__(self, eta,lmd):
        #self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def calculate(self, X, y,  w, key = "sigmoid"):
        self.p = self.activation(X, key)
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p)))

    def grad(self, X, errors):
        return X.T.dot(errors)


class Cost_Ridge(Costfunctions):

    def __init__(self, eta, lmd):
        #self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def calculate(self, X, y, w, key = "sigmoid"):
        self.p = self.activation(X, key)
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p))) + self.lmd*w[1:] + w[0]

    def grad(self, X, errors):
        return X.T.dot(errors) + self.lmd*w[1:] + w[0]

class Cost_Lasso(Costfunctions):

    def __init__(self, eta, lmd):
        #self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def calculate(self, X, y, w, key = "sigmoid"):
        self.p = self.activation(X, "sigmoid")
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p))) + self.lmd

    def grad(self, X, errors):
        return X.T.dot(errors) + self.lmd
