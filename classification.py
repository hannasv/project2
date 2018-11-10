import numpy as np
import Costfunctions

class LogisticRegression(object):
    """Logistic Regression Classifier using stochastic gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0001 and 1.0)
    batch_size : int (defalt 10)
      The size of the partioning of the trainingset.
    epochs : int (defalt 100)
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    shuffle : boolean (default True)
      Determines if the indices are shuffled between each epoch
    key : string (default "sigmoid")
        Choosing the activationfunction.
    alpha : float (default 0.0001)
        The elu activationfunction converges to -alpha for negative values.
    penalty : string (default "l1")
        Choosing the type of penalization. l1 is lasso, l2 is ridge.
    lmd : float (default 0.0)
        Value of the penaly. lmd = 0, corresponds to no penalty. This is OLS.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : float
        bias after fitting.
    epochCost : list
      Logistic cost function value in each epoch
    eval_ : dict
        Contains training and test performance.
    """

    def __init__(self, eta = 0.01, penalty = "l1", lmd = 0, random_state = 0, shuffle = True, batch_size = 10, epochs=100,  key = "sigmoid", alpha = 0.0001):
        self.eta = eta
        self.random = np.random.RandomState(random_state)
        self.key = key
        self.penalty = penalty
        self.lmd = lmd
        self.shuffle = shuffle
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs

        # Attributes:
        self.eval_ = None
        self.w_ = None
        self.b_ = None
        self.epochCost = None

    def fit(self, X_train, y_train, X_test, y_test):
        """ Fit training data.

        Parameters
        ----------
        X_train : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number
          of samples and n_features is the number of features.

        y_train : array-like, shape = [n_samples]
          Target values.

        X_test : {array-like}, shape = [n_samples, n_features]
          Test vectors, used for validation.

        y_test : array-like, shape = [n_samples]
          Test vector used for validation.



        Returns
        -------
        self : object

        """

        #initalizing the weights
        self.w_ = 0.1*np.random.randn(X_train.shape[1])
        self.b_ = 1

        self.epochCost = []
        self.eval_ = {'cost': [], 'train_preform': [], 'valid_preform': []}

        for epoch in range(self.epochs):
            n_samples, n_features = np.shape(X_train)
            indices = np.arange(n_samples)
            cost_ = []
            scores_epochs = []

            if self.shuffle:
                self.random.shuffle(indices)

            for idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[idx:idx + self.batch_size]
                batchX = X_train[batch_idx,:]
                batchY = y_train[batch_idx]
                net_input = np.dot(batchX, self.w_) + self.b_
                output = self.activation(net_input, self.key)
                errors = output - batchY

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

                gradient = batchX.T.dot(errors) + gterm

                self.w_ -=  self.eta * gradient
                self.b_ -=  self.eta * errors.sum()

                net_input = np.dot(batchX, self.w_) + self.b_
                test = 1. / (1. + np.exp(-net_input))
                score = np.sum(np.where(test >= 0.5, 1, 0) == batchY)/len(output)

                scores_epochs.append(score)
                cost_.append(cost)
                #print("score: " + str(np.average(scores_epochs)) + " for epoch:  " + str(epoch))\

            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)

            self.epochCost.append( np.average( cost_ ))
            acc_test = np.sum(y_test == y_test_pred)/len(y_test)
            acc_train = np.sum(y_train == y_train_pred)/len(y_train)
            self.eval_['train_preform'].append(acc_train)
            self.eval_['valid_preform'].append(acc_test)
        return self

    def activation(self, Xw, key):
        """
        Applies activation function.

        The sigmoid activation function determines the probability
        of being in a class.

        Xw : (array-like), shape = print en shape og se (y)
            Xw is the dotproduct of training data and weights pluss bias.

        key : string (default "sigmoid")
          The choosen activation function.

        """

        if (key == "sigmoid"):
            return  1. / (1. + np.exp(-Xw))
        elif(key == "LReLu"):
            Z_out = np.copy(Xw)
            Z_out[np.where(Xw <= 0)] = self.alpha * Xw[np.where(Xw <= 0)]
            return Z_out
        elif(key == "elu"):
            Z_out = np.copy(Xw)
            Z_out[np.where(Xw <= 0)] = self.alpha *(np.exp( Xw[np.where(Xw <= 0)]) - 1)
            return Z_out
        else:
            raise ValueError('Invalid activation function {}'.format(key))


    def predict(self, X):
        """ Predicts the results of logistic regression.

        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number
          of samples and n_features is the number of features.

        """
        net_input = np.dot(X, self.w_) + self.b_
        new =  1. / (1. + np.exp(-net_input))
        return np.where(new >= 0.5, 1, 0)
