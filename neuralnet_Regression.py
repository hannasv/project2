import numpy as np

class MLP_Regression:
    """ Feedforward neural network / Multi-layer perceptron classifier.
    Parameters
    ------------
    n_hidden : (array like)
        List containing the number of nodes in each layer.
    lmd : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.
    """
    def __init__(self, hidden_layers,lmd=0., epochs=100, eta=0.001, shuffle=True,
                  minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = hidden_layers

        """Need to make this number of hidden units in each layer
        IDEA:
        n_hidden_units = [nr in h1, nr in h2, ..., nr in hn]
        nr layers is the length of this one.
        """
        self.l2 = lmd
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size


    def activation(self, z, key = "sigmoid", alpha = 0.1):
        if (key == "sigmoid"):
            return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
        elif(key == "linear"):
            return z
        elif(key == "ELU"):
            if (z >= 0):
                return z
            else:
                return alpha*(np.exp(z)-1)
        else:
            print("Unvalide keyword argument. Use siogmoid or ELU for activation.")

    """ Copied from slides"""
    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def _forward(self, X):
        """Compute forward propagation step"""

        # net input of hidden layer
        z_h = X.dot(self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self.activation(z_h, "sigmoid")

        # step 3: net input of output layer
        z_out = a_h.dot(self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self.activation(z_out, "linear")

        return z_h, a_h, z_out, a_out

    """ Copied from slides."""
    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities


    def _compute_cost(self, y_enc, output):
        """Compute cost function.
        Parameters
        ----------
        y_enc : array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)
        Returns
        ---------
        cost : float
            Regularized cost
        """

        """ Ridge Regression --> Default in scikit learn """

        # ridge penalty term
        L2_term = self.l2 *(np.sum(self.w_h ** 2.) +  np.sum(self.w_out ** 2.))
        cost = (y-X.dot(w[1:])).T@(y-X.dot(w[1:])) + L2_term

        return cost

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
        z_h, a_h, z_out, a_out = self._forward(X)

        return z_out

    def fit(self, X_train, y_train, X_test, y_test):
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

        n_output = 1
        n_features = X_train.shape[1]

        """ Initialize weights """

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}


        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                """ Back propagation """

                # [n_samples, n_classlabels]

                """BP1A:  dC/da = aj^L - yj, where L denotes outpul layer"""
                sigma_out = a_out - y_train[batch_idx]

                # [n_samples, n_hidden]
                """ The deriavtive of the activation function """
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]

                """BP2: error in layer l delta_l = w_out.T dot sigma_out *derivative of activation function.   """
                sigma_h = sigma_out.dot(self.w_out.T) * sigmoid_derivative_h

                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = X_train[batch_idx].T.dot(sigma_h)
                grad_b_h = sigma_h.sum(axis=0)

                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                """ Eq. 32 nielsens book """
                grad_w_out = a_h.T.dot(sigma_out)
                """ BP3: """
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and weight updates
                """ BP4: """
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h  # bias shouldn't be penalized

                # Updating weights and biases in hidden layer
                self.w_h = self.w_h - self.eta * delta_w_h
                self.b_h = self.b_h - self.eta * delta_b_h

                """ 32: """
                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out

                # Updating weights and biases in output layer.
                self.w_out = self.w_out - self.eta * delta_w_out
                self.b_out = self.b_out - self.eta * delta_b_out

            """ Evaluation """
            # Better name since this is a regression case.
            y_train_enc = y_train
            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc = y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_test)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_test == y_valid_pred)).astype(np.float) /
                         X_test.shape[0])

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self


if __name__ == '__main__':

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y, = make_regression(n_samples=100, n_features=100, n_informative=10)
    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

    nnreg = MLP_Regression(2)
    nnreg.fit(X_train, y_train, X_test, y_test)
    print(nnreg.eval_)
