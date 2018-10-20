@abstract
class Costfunctions:
    def __init__(eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def error(y):
        return y-self.p

    def update_weights(new):
        self.w = new

    def sigmoid(Xw):
        return 1. / (1. + np.exp(-np.clip(Xw, -250, 250)))

    def ELU():
        pass

    # Two types  of optimizer:
    def stochastic_gradient_descent(eta, n_iter):
        pass

    def standard_gradient_descent(eta, n_epochs, t0, t1):
        pass

    def mini_batch_gradient_descent(eta, n_epochs, batch_size):
        pass


class Cost_OLS("Costfunctions"):

    def __init__(eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def compute(X):
        self.p = sigmoid(X)
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p)))

    def grad(X):
        return X.T.dot(error())


class Cost_Ridge("Costfunctions"):

    def __init__(eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def compute(X):
        self.p = sigmoid(X)
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p))) + self.lmd*self.w[1:] + self.w[0]

    def grad(X):
        return X.T.dot(error()) + self.lmd*w[1:] + w[0]

class Cost_Lasso("Costfunctions"):

    def __init__(eta, w, lmd):
        self.w = w,
        self.eta = eta,
        self.lmd = lmd,

    def compute(X, y):
        self.p = sigmoid(X)
        return -y.dot(np.log(self.p)) - ((1 - y).dot(np.log(1 - self.p))) + self.lmd

    def grad(X):
        return X.T.dot(error()) + self.lmd
