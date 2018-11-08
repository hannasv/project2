import numpy as np
from utils import r2_score, mean_squared_error

class NeuralNetMLP:
    """Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    batch_size : int (default: 1)
        Number of training samples per minibatch used in the stochastic gradient descent.
        (One gradient update per minibatch)
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """
    def __init__(self, n_hidden=30, l2=0.4, epochs=100, eta=0.001, shuffle=True,
                 batch_size=1, seed=None, alpha=0.0001, activation='sigmoid', tpe = "logistic"):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.alpha = alpha
        self.activation = activation

        self.W_h = None
        self.b_h = None
        self.W_out = None
        self.b_out = None
        self.tpe = tpe


    def activate(self, Z, kind='elu', deriv=False):

        if kind == 'sigmoid':
            a = 1. / (1. + np.exp(-np.clip(Z, -250, 250)))
            if deriv:
                return a * (1. - a)
            #return 1 / (1 + np.exp(Z))
            return a
        elif kind == 'elu':
            if deriv:
                Z_deriv = np.ones((np.shape(Z)), dtype=float)
                Z_deriv[np.where(Z < 0)] = self.alpha * np.exp(np.clip(Z[np.where(Z < 0)], -250, 250))
                return Z_deriv
            else:
                Z_out = np.copy(Z)
                Z_out[np.where(Z < 0)] = self.alpha * (np.exp(np.clip(Z[np.where(Z < 0)], -250, 250)) - 1)
                return Z_out
        elif kind == "linear":
            if deriv:
                return 1
            else:
                return Z
        else:
            raise ValueError('Invalid activation function {}'.format(kind))

        return None

    def initialize_weights_and_bias(self, X_train):

        n_output = 1
        n_samples, n_features = np.shape(X_train)
        # Using three hidden h_layers
        self.b_h =  np.ones((1, self.n_hidden))
        self.W_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        self.b_out = np.ones(n_output)
        self.W_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))


    def _forwardprop(self, X):
        """Compute forward propagation step"""
        Z_hidden = np.clip(np.dot(X, self.W_h) + self.b_h, -250,250)
        A_hidden = self.activate(Z_hidden, self.activation, deriv = False)
        Z_out = np.dot(A_hidden, self.W_out) + self.b_out

        if (self.tpe == "regression"):
            A_out = self.activate(Z_out, "linear", deriv = False)
        elif (self.tpe == "logistic"):
            A_out = self.activate(Z_out, "sigmoid", deriv = False)
        else:
            raise ValueError('Invalid activation function {}'.format(self.tpe))

        return Z_hidden, A_hidden, Z_out, A_out

    def _backprop(self, y, X, A_hidden, Z_hidden, A_out, Z_out, batch_idx):

        delta_a_out = A_out - y[batch_idx].reshape(self.batch_size, 1)

        """ For the regressioncase where the outpu func is linar"""
        if (self.tpe == "regression"):
            act_derivative_out = self.activate(Z_out, "linear", deriv = True)
        elif (self.tpe == "logistic"):
            act_derivative_out = self.activate(Z_out, "sigmoid", deriv = True)
        else:
            raise ValueError('Invalid activation function {}'.format(self.tpe))
        # Since we are in the regression case with a linear ouput funct.

        delta_out = delta_a_out*act_derivative_out
        grad_w_out = np.dot(A_hidden.T, delta_out)
        grad_b_out = np.sum(delta_out, axis=0)

        self.W_out = self.W_out - self.eta * grad_w_out
        self.b_out = self.b_out - self.eta * grad_b_out

        # oppdatere med eta
        act_derivative_h = self.activate(Z_hidden, self.activation, deriv=True)
        error_hidden = np.dot(delta_out, self.W_out.T) * act_derivative_h
        grad_w_h = np.dot(X[batch_idx].T, error_hidden)
        grad_b_h = np.sum(error_hidden, axis=0)

        #delta_b_h = grad_b_h
        #delta_w_h = (grad_w_h + self.l2 * self.W_h)

        self.W_h = self.W_h - self.eta * grad_w_h
        self.b_h = self.b_h - self.eta * grad_b_h

        return None

    def _cost(self, X, y):
        """Compute cost function.

        Parameters
        ----------
        UPDATE:
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """

        L2_term = self.l2 * (np.dot(self.W_h.T, self.W_h) + np.dot(self.W_out.T, self.W_out))
        #output_errors = np.average((y_true - y_pred) ** 2, axis=0)
        # Logistic cost
        cost_linear = (y-X.dot(self.W_out)).T.dot(y-X.dot(self.W_out))
        return cost_linear

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.

        """

        Z_hidden, A_hidden, Z_out, A_out = self._forwardprop(X)

        if (self.tpe == "logistic"):
            return np.where(A_out >= 0.5, 1, 0)
        elif (self.tpe == "regression"):
            return A_out

    def _minibatch_sgd(self, X_train, y_train):
        n_samples, n_features = np.shape(X_train)

        indices = np.arange(n_samples)

        if self.shuffle:
            self.random.shuffle(indices)

        for idx in range(0, n_samples, self.batch_size):

            batch_idx = indices[idx:idx + self.batch_size]

            # Forwardpropagation.
            Z_hidden, A_hidden, Z_out, A_out = self._forwardprop(
                X_train[batch_idx, :]
            )

            # Backpropagation.
            self._backprop(
                y_train, X_train, A_hidden, Z_hidden, A_out, Z_out, batch_idx
            )

        return self

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """ Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples]
            Target class labels.
        X_test : array, shape = [n_samples, n_features]
            Sample features for validation during training
        y_test : array, shape = [n_samples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """

        self.initialize_weights_and_bias(X_train)
        #print(self.W_out.shape)

        # for progress formatting
        epoch_strlen = len(str(self.epochs))
        self.eval_ = {'cost': [], 'train_preform': [], 'valid_preform': []}

        # iterate over training epochs
        for epoch in range(self.epochs):

            # Includes forward + backward prop.
            self._minibatch_sgd( X_train, y_train)

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forwardprop(X_train)

            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)

            y_test = y_test.reshape((len(y_test),1))
            y_train = y_train.reshape((len(y_train),1))

            if (self.tpe == "regression"):
                # Cost without penalty (y-X.dot(self.W_out)).T.dot(y-X.dot(self.W_out))
                train_preform = mean_squared_error(y_train, y_train_pred)
                valid_preform = mean_squared_error(y_test, y_test_pred)

                #self.eval_['cost'].append(self._cost(X_train, y_train))
                self.eval_['train_preform'].append(train_preform)
                self.eval_['valid_preform'].append(valid_preform)

            elif(self.tpe == "logistic"):
                #Calculate accuracy
                acc_test = np.sum(y_test == y_test_pred)/len(y_test)
                acc_train = np.sum(y_train == y_test_pred)/len(y_test)
                self.eval_['train_preform'].append(acc_train)
                self.eval_['valid_preform'].append(acc_test)
        return self
