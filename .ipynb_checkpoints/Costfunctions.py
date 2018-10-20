@abstract
class Costfunction:

    def __init__(X, y, w, lmd, key):
        self.X = X,
        self.y = y,
        self.w = w,
        self.lmd = lmd
        self.p = None

    # To types of activation function:
    def sigmoid(X, w):
        return 1./1. + np.exp(-X.dot(w[1:]) - w[0]])

    def ELU(X,w):
        pass

    def error(y):
        self.p = self.sigmoid(np.dot(X, self.w[1:]) + self.w[0])
        return (y-self.p).sum()

    def update_weigts(new_w):
        self.w = new_w

class Cost_OLS(Costfunction):
    def __init__(X, y, w, lmd):
        self.X = X,
        self.y = y,
        self.w = w,
        self.lmd = lmd

    def calculate():
        self.p = self.sigmoid(np.dot(X, self.w[1:]) + self.w[0])
        return -self.y.dot(np.log(output)) - ((1 - self.y).dot(np.log(1 - output)))

    def grad():
        return X.T.dot(self.error(self.y))

class Cost_Ridge(Costfunction):
    def __init__(X, y, w, lmd):
        self.X = X,
        self.y = y,
        self.w = w,
        self.lmd = lmd

    def calculate():
        self.p = self.key(np.dot(X, self.w[1:]) + self.w[0])
        return -self.y.dot(np.log(self.p)) - ((1 - self.y).dot(np.log(1 - self.p))) + lmd*self.w

    def grad():
        return X.T.dot(self.error(self.y)) - self.lmd*self.w

class Cost_Lasso(Costfunction):
    def __init__(X, y, w, lmd):
        self.X = X,
        self.y = y,
        self.w = w,
        self.lmd = lmd

    def calculate():
        output = self.key(np.dot(X, self.w[1:]) + self.w[0])
        return -self.y.dot(np.log(output)) - ((1 - self.y).dot(np.log(1 - output)))

    def grad():
        return X.T.dot(self.error(self.y)) - self.lmd
