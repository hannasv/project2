# Would prefer to make this an abstract class.
class Costfunctions:
    def __init__(self, eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def error(self, y):
        return y-self.p

    def update_weights(self, new):
        self.w = new

    def activation(self, Xw, key):
        if (key == "sigmoid"):
            return 1. / (1. + np.exp(-np.clip(Xw, -250, 250)))
        elif(key == "elu"):
            return 0

class Cost_OLS(Costfunctions):

    def __init__(self, eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def compute(self, X, key = "sigmoid"):
        self.p = self.activation(X, key)
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p)))

    def grad(self, X):
        return X.T.dot(error())


class Cost_Ridge(Costfunctions):

    def __init__(self, eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def compute(self, X, key = "sigmoid"):
        self.p = self.activation(X, key)
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p))) + self.lmd*self.w[1:] + self.w[0]

    def grad(self, X):
        return X.T.dot(error()) + self.lmd*w[1:] + w[0]

class Cost_Lasso(Costfunctions):

    def __init__(self, eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def compute(self, X, y):
        self.p = activation(X)
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p))) + self.lmd

    def grad(self, X):
        return X.T.dot(error()) + self.lmd
