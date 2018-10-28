# Would prefer to make this an abstract class.
import numpy as np

class Costfunctions:
    def __init__(self, eta, w, lmd):
        self.w = w
        self.eta = eta
        self.lmd = lmd
        self.p = None

    # Computes the residuals = error
    def r(self, y): # resuduals.
        return y-self.p

    # Currently not used anywhere.
    def update_weights(self, new):
        self.w = new

    def activation(self, Xw, key):
        if (key == "sigmoid"):
            self.p = 1. / (1. + np.exp(-np.clip(Xw, -250, 250)))
            return self.p
        elif(key == "ELU"):
            if (Xw >= 0):
                return Xw
            else:
                return alpha*(np.exp(Xw)-1)
        else:
            print("Unvalide keyword argument. Use siogmoid or ELU for activation.")

class Cost_OLS(Costfunctions):

    def __init__(self, eta, lmd = 0):
        #self.w = w,
        self.eta = eta,
        self.lmd = lmd

    """ Normal costfunction and gradient"""
    def calculate_cost(self, X, y, w):
        return (y-X.dot(w[1:])).T@(y-X.dot(w[1:]))

    def grad(self, X, y, w, errors):
        theta=0
        return 2 * X.T.dot(X.dot(y-w[1:]))

    """ Logistic costfunction and gradient"""
    def log_calculate(self, X, y,  w):
        #self.p = self.activation(X, key)
        return -y.dot(np.log(self.p+ 1e-12)) - ((1 - y).dot(np.log(1 - self.p + 1e-12)))

    # returns a vector
    def log_grad(self, X, w, errors):
        #print(X.T.dot(errors))
        return X.T.dot(errors)


class Cost_Ridge(Costfunctions):

    def __init__(self, eta, lmd = 0.1):
        #self.w = w,
        self.eta = eta
        self.lmd = lmd
    """ Normal costfunction and gradient"""
    def calculate_cost(self, X, y, w):
        return (y-X.dot(w[1:])).T@(y-X.dot(w[1:])) + self.lmd*(np.sum(w[1:])**2)

    def grad(self, X, w, errors):
        l2term =  self.lmd *np.sum(w[1:] ** 2)
        return 2 * X.T.dot(X.dot(y-w[1:])) + l2term

    """ Logistic costfunction and gradient"""
    def log_calculate(self, X, y, w):
        # penalty on bias to?
        l2term =  self.lmd *np.sum(w[1:] ** 2)# + np.sum(w[0] ** 2))
        return -y.dot(np.log(self.p+ 1e-12)) - ((1 - y).dot(np.log(1 - self.p+ 1e-12) )) + l2term

    # returns a array
    def log_grad(self, X, w, errors):
        return X.T.dot(errors) + 2*self.lmd*w[1:]

class Cost_Lasso(Costfunctions):

    def __init__(self, eta, lmd):
        #self.w = w,
        self.eta = eta
        self.lmd = lmd

    """ Normal costfunction and gradient"""
    def calculate_cost(self, X, y, w):
        l1term = self.lmd*np.sum(np.abs(w[1:]))
        return (y-X.dot(w[1:])).T@(y-X.dot(w[1:])) + l1term

    def grad(self, X, w, errors):
        return 2 * X.T.dot(X.dot(y-w[1:])) + self.lmd*np.sign(w[1:])

    """ Logistic costfunction and gradient"""
    def log_calculate(self, X, y, w):
        # returns a scalar.
        #self.p = self.activation(X, "sigmoid")
        l1term = self.lmd*np.sum(np.abs(w[1:]))
        return -y.dot(np.log(self.p+ 1e-12)) - ((1 - y).dot(np.log(1 - self.p+1e-12))) + l1term

    # The derivative of a absolute value is a signfunction
    def log_grad(self, X, w, errors):
        return X.T.dot(errors) + self.lmd*np.sign(w[1:])
