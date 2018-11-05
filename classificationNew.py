import numpy as np
import Costfunctions

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
    def __init__(self, eta = 0.01, random_state = 0, shuffle = True, batch_size = 10, epochs=100, penalty = "l1",lmd = 0, tolerance=1e-14, key = "sigmoid", alpha = 0.01):

        self.eta = eta
        #self.n_iter = n_iter
        self.random = np.random.RandomState(random_state)
        self.key = key
        self.penalty = penalty
        self.lmd = lmd
        self.tol = tolerance
        self.shuffle = shuffle
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_ = None
        self.cost_=None
        #self.p = None
        self.w_ = None
        self.b_ = None
        self.epochCost = None

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

        #initalizing the weights
        self.w_ = 0.1*np.random.randn(X.shape[1])
        self.b_ = 1

        # Usinf stochastic gradient descent_method

        self.epochCost = []

        for epoch in range(self.epochs):
            n_samples, n_features = np.shape(X)
            indices = np.arange(n_samples)
            cost_ = []
            scores_epochs = []

            if self.shuffle:
                self.random.shuffle(indices)

            for idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[idx:idx + self.batch_size]
                batchX = X[batch_idx,:]
                batchY = y[batch_idx]
                #print(batchY.shape)
                net_input = np.clip(np.dot(batchX, self.w_) + self.b_,-250,250)
                output = self.activation(net_input, self.key)
                #if np.isfinite(output).all():
                #    print("ouput contains nans or inf")
                errors = output - batchY
                # Using lasso pentalty
                # TODO. Make if function witch sets penalty term based on model = "ols", "ridge", "lasso"

                if (self.penalty == "l2"):
                    # l2 --> ridge
                    gterm = 2*self.lmd*self.w_
                    cterm = self.lmd*(np.sum(self.w_)**2)
                elif (self.penalty == None):
                    gterm = 0
                    cterm = 0
                else:
                    # l1term --> lasso
                    gterm = self.lmd*np.sign(self.w_)
                    cterm = self.lmd*self.w_

                cost = -batchY.dot(np.log(output + 1e-8)) - ((1 - batchY).dot(np.log(1 - output + 1e-8) )) + cterm

                gradient = 1/self.batch_size*batchX.T.dot(errors) + gterm
                #update weights
                self.w_ -=  self.eta * gradient
                self.b_ -=  self.eta * errors.sum()

                net_input = np.dot(batchX, self.w_) + self.b_
                test = 1. / (1. + np.exp(-net_input))
                score = np.sum(np.where(test >= 0.5, 1, 0) == batchY)/len(output)
                scores_epochs.append(score)
                cost_.append(cost)
                #print("score: " + str(np.average(scores_epochs)) + " for epoch:  " + str(epoch))\
            self.epochCost.append( np.average( cost_ ))
        return self

    def activation(self, Xw, key):
        if (key == "sigmoid"):
            return  1. / (1. + np.exp(-Xw))
        elif(key == "LReLu"):
            Z_out = np.copy(Xw)
            Z_out[np.where(Xw <= 0)] = self.alpha * Xw[np.where(Xw <= 0)]
            return Z_out
        elif(key == "ELU"):
            Z_out = np.copy(Xw)
            Z_out[np.where(Xw <= 0)] = self.alpha *(np.exp( Xw[np.where(Xw <= 0)]) - 1)
            return Z_out
        else:
            print("Unvalide keyword argument. Use siogmoid or ELU for activation.")


    def predict(self, X):
        net_input = np.dot(X, self.w_) + self.b_
        new =  1. / (1. + np.exp(-net_input))
        return np.where(new >= 0.5, 1, 0)
