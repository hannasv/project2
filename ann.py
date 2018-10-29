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
    def __init__(self, n_hidden=30, h_layers = None, l2=0.4, epochs=100, eta=0.001, shuffle=True,
                 batch_size=1, seed=None, alpha=0.01, activation='elu'):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.alpha = alpha
        self.activation = activation
        self.h_layers = h_layers

        # Set to coorect dimentions??
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_weights = None
        self.output_bias = None


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
        else:
            raise ValueError('Invalid activation function {}'.format(kind))

        return None

    def initialize_weights_and_bias(self, X_train):
        #self.hidden_weights = np.random.randn(self.n_features, self.n_hidden)
        #self.hidden_bias = np.zeros(self.n_hidden) + 0.01

        #self.output_weights = np.random.randn(self.n_hidden, 1)
        #self.output_bias = np.zeros(1) + 0.01

        n_output = 1
        n_samples, n_features = np.shape(X_train)
        # weights for input -> hidden
        if (self.h_layers != None):
            n_hidden_layers = len(self.h_layers)
            self.b_h = np.zeros((self.n_hidden, n_hidden_layers)) + 0.0001
            self.W_h = np.array([np.random.uniform(-0.7,0.7,(n_features, self.n_hidden)) for i in range(n_hidden_layers)])
            self.b_out = np.zeros(n_output) + 0.0001
            self.W_out = np.random.uniform(-0.7,0.7,(self.n_hidden, n_output))

        else:
            self.b_h = np.zeros(self.n_hidden) + 0.0001
            self.W_h = np.random.uniform(-0.7,0.7,(n_features, self.n_hidden))

            # weights for hidden -> output
            self.b_out = np.zeros(n_output) + 0.0001
            self.W_out = np.random.uniform(-0.7,0.7,(self.n_hidden, n_output))

    def _forwardprop(self, X):
        """Compute forward propagation step"""
        if (self.h_layers == None):
            #n_layers = len(self.h_layers)
            # step 1: net input of hidden layer
            # [n_samples, n_features] dot [n_features, n_hidden]
            # -> [n_samples, n_hidden]

            Z_hidden = np.dot(X, self.W_h) + self.b_h
            print("X: " + str(X.shape))
            print("W_h:" + str(self.W_h.shape))
            print("b:h " + str(self.b_h.shape))
            # step 2: activation of hidden layer
            A_hidden = self.activate(Z_hidden, self.activation)

            # step 3: net input of output layer
            # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
            # -> [n_samples, n_classlabels]
            Z_out = np.dot(A_hidden, self.W_out) + self.b_out

            # step 4: linear activation output layer
            A_out = Z_out


        else: # When we have more than one hidden layers
            Z_hidden = []#np.zeros((len(self.W_h), len(self.h_layers)))
            A_hidden = []#np.zeros((len(self.W_h), len(self.h_layers)))

            for i in range(len(self.h_layers)):
                Z_hidden.append(np.dot(X, self.W_h[i]) + self.b_h[:,i])
                A_hidden.append(self.activate(Z_hidden[i], self.activation))

            #print(self.W_out.shape)
            #print(np.dot(A_hidden[-1], self.W_out).shape)
            Z_out = np.dot(A_hidden[-1], self.W_out) + self.b_out
            A_out = Z_out # because it linar is this wrong

        return np.array(Z_hidden), np.array(A_hidden), Z_out, A_out

    def _backprop(self, y, X, A_hidden, Z_hidden, A_out, batch_idx):

        if (self.h_layers == None):
            # [n_samples, n_classlabels]

            if np.ndim(y) < 2:
                # in order to keep y as a column
                error_out = A_out - y[batch_idx, np.newaxis]
            else:
                error_out = A_out - y[batch_idx]
                # not in use

            # [n_hidden, n_samples] dot [n_samples, n_classlabels]
            # -> [n_hidden, n_classlabels]
            grad_w_out = np.dot(A_hidden.T, error_out)
            grad_b_out = np.sum(error_out, axis=0)

            # [n_samples, n_hidden]
            act_derivative = self.activate(Z_hidden, self.activation, deriv=True)

            # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
            # -> [n_samples, n_hidden]
            error_hidden = np.dot(error_out, self.W_out.T) * act_derivative

            # [n_features, n_samples] dot [n_samples, n_hidden]
            # -> [n_features, n_hidden]
            grad_w_h = np.dot(X[batch_idx].T, error_hidden)
            grad_b_h = np.sum(error_hidden, axis=0)


            # Regularization and weight updates
            delta_w_h = (grad_w_h + self.l2 * self.W_h)

            # NOTE: bias is not regularized.
            delta_b_h = grad_b_h
            delta_b_out = grad_b_out

            self.W_h = self.W_h - self.eta * delta_w_h
            self.b_h = self.b_h - self.eta * delta_b_h

            delta_w_out = (grad_w_out + self.l2 * self.W_out)
        else:

            # [n_samples, n_classlabels]
            if np.ndim(y) < 2:
                error_out = A_out - y[batch_idx, np.newaxis]
            else:
                error_out = A_out - y[batch_idx]
                # not in use

            # [n_hidden, n_samples] dot [n_samples, n_classlabels]
            # -> [n_hidden, n_classlabels]
            error_prev = error_out
            prev_weights = self.W_out

            for i in range(len(self.h_layers)):
                #print("A_H.T: dot  " + str(np.shape(A_hidden[-1-i].T)))
                print("first weigfhts: " + str(self.W_out.shape))
                print("error_out/prev " + str(error_prev.shape))
                grad_w_out = np.dot(A_hidden[-1-i].T, error_prev)
                grad_b_out = np.sum(error_prev)
                # [n_samples, n_hidden]
                act_derivative = self.activate(Z_hidden[-1-i], self.activation, deriv=True)
                #print(" Z_hidden[:,-1-i]:  " + str(Z_hidden[-1-i].shape))
                #print("act_derivative: " + str(act_derivative.shape))
                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                print("prev_weights.T: " + str(prev_weights.T.shape))
                error_hidden = np.dot(error_prev, prev_weights.T) * act_derivative
                error_prev = error_hidden
                prev_weights = self.W_h[-i-1]
                print("update prev weights: " + str(prev_weights.shape))
                                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X[batch_idx].T, error_prev)
                grad_b_h = np.sum(error_prev, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2 * self.W_h[-i-1])

                # NOTE: bias is not regularized.
                delta_b_h = grad_b_h
                delta_b_out = grad_b_out # returned

                print(delta_b_h.shape)
                print(self.b_h[i].shape)

                self.W_h[i] = self.W_h[i] - self.eta * delta_w_h
                self.b_h[i] = self.b_h[i] - self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2 * self.W_out) # returned

        return delta_w_out, delta_b_out

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
        output_errors = np.average((y_true - y_pred) ** 2, axis=0)
        cost_linar = (y-X.dot(self.W_h)).T.dot(y-X.dot(self.W_h))
        return cost_linar + L2_term

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

        return A_out

    def _minibatch_sgd(self, X_train, y_train):
        n_samples, n_features = np.shape(X_train)

        indices = np.arange(n_samples)

        if self.shuffle:
            self.random.shuffle(indices)

        for idx in range(0, n_samples, self.batch_size):

            batch_idx = indices[idx:idx + self.batch_size]

            # TODO: Add extra dim for layers.

            # Forwardpropagation.
            Z_hidden, A_hidden, Z_out, A_out = self._forwardprop(
                X_train[batch_idx, :]
            )

            # Backpropagation.
            delta_w_out, delta_b_out = self._backprop(
                y_train, X_train, A_hidden, Z_hidden, A_out, batch_idx
            )

            self.W_out = self.W_out - self.eta * delta_w_out
            self.b_out = self.b_out - self.eta * delta_b_out

        return self

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """ Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples]
            Target class labels.
        X_valid : array, shape = [n_samples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_samples]
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

            #cost = self._cost(y_test, y_test_pred)

            train_preform = mean_squared_error(y_train, y_train_pred)
            valid_preform = mean_squared_error(y_test, y_test_pred)

            #self.eval_['cost'].append(cost)
            self.eval_['train_preform'].append(train_preform)
            self.eval_['valid_preform'].append(valid_preform)

        return self


if __name__ == '__main__':

    import numpy as np
    import pandas as pd

    from sklearn import datasets
    from sklearn.neural_network import MLPRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    #digits = datasets.load_digits()
    #X, y = digits.data, digits.target
    X, y = make_regression(n_features=10, random_state=1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    ann = NeuralNetMLP(batch_size=5, n_hidden=30, h_layers = [30,30,30], eta = 0.001)
    ann.fit(X_train, y_train, X_valid, y_valid)
    print(ann.eval_['valid_preform']) #nans
    mlp = MLPRegressor(max_iter=10000)
    mlp.fit(X, y)
    print(mlp.loss_)
