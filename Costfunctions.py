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
            if (z >= 0):
                return z
            else:
                return alpha*(np.exp(z)-1)
        else:
            print("Unvalide keyword argument. Use siogmoid or ELU for activation.")

class Cost_OLS(Costfunctions):

    def __init__(self, eta, lmd = 0):
        #self.w = w,
        self.eta = eta,
        self.lmd = lmd

    def calculate(self, X, y,  w):
        #self.p = self.activation(X, key)
        return -y.dot(np.log(self.p+ 1e-12)) - ((1 - y).dot(np.log(1 - self.p + 1e-12)))

    # returns a vector
    def grad(self, X, w, errors):
        #print(X.T.dot(errors))
        return X.T.dot(errors)


class Cost_Ridge(Costfunctions):

    def __init__(self, eta, lmd = 0.1):
        #self.w = w,
        self.eta = eta
        self.lmd = lmd

    def calculate(self, X, y, w):
        # Bruker ikke Xen
        #net_input = np.dot(X, self.w_[1:]) + self.w_[0]
        #self.p = self.activation(X, key)
        #print(np.sum(w[0]**2))
        #print(np.sum(w[1:]**2)) blir nesten like

        # penalty on bias to?
        l2term =  self.lmd *np.sum(w[1:] ** 2)# + np.sum(w[0] ** 2))
        return -y.dot(np.log(self.p+ 1e-12)) - ((1 - y).dot(np.log(1 - self.p+ 1e-12) )) + l2term
        #+ self.lmd*w[1:] + w[0]

    # returns a array
    def grad(self, X, w, errors):
        return X.T.dot(errors) + 2*self.lmd*w[1:]

class Cost_Lasso(Costfunctions):

    def __init__(self, eta, lmd):
        #self.w = w,
        self.eta = eta
        self.lmd = lmd

    def calculate(self, X, y, w):
        # returns a scalar.
        #self.p = self.activation(X, "sigmoid")
        l1term = self.lmd*np.sum(np.abs(w[1:]))
        return -y.dot(np.log(self.p+ 1e-12)) - ((1 - y).dot(np.log(1 - self.p+1e-12))) + l1term

    # The derivative of a absolute value is a signfunction
    def grad(self, X, w, errors):
        return X.T.dot(errors) + self.lmd*np.sign(w[1:])
