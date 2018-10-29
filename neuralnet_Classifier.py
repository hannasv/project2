import numpy as np

class MLP_Classifier:
    """ Feedforward neural network / Multi-layer perceptron 'classifier'.
    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    lmd : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
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
    def __init__(self, n_hidden=30,lmd=0., epochs=100, eta=0.001,
                 minibatch_size=1, random_state=None):

        self.random = np.random.RandomState(random_state)
        self.n_hidden = n_hidden

        """ When upgrading to a deep neural network
        n_hidden_units = [nr in h1, nr in h2, ..., nr in hn]
        nr layers is the length of this one.
        """
        self.lmd = lmd
        self.epochs = epochs
        self.eta = eta
        #self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.n_features = None

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


    def create_biases_and_weights(self):
        hidden_weights = np.random.randn(self.n_features, self.n_hidden)
        hidden_bias = np.zeros(self.n_hidden) + 0.01

        output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        output_bias = np.zeros(self.n_categories) + 0.01
        return hidden_weights, hidden_bias, utput_weights, output_bias


    """ Copied from slides."""
    """def feed_forward_out(self, X): --> When we increase dimentions
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)
._sigmoid
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities"""

    def feed_forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = X.dot(self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self.activation(z_h, "sigmoid")

        # step 3: net input of output layer
        z_out = a_h.dot(self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self.activation(z_out, "sigmoid")

        return z_h, a_h, z_out, a_out

    def compute_cost(self, y_test, output):
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

        """ Used for Logistic Regression"""

        L2_term = (self.l2 *(np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        # adds 1e12 to avoid dividing by zero.
        term1 = -y_enc * (np.log(output+1e12))
        term2 = (1. - y_enc) * np.log(1. - output + 1e12)
        cost = np.sum(term1 - term2) + L2_term
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
        z_h, a_h, z_out, a_out = self.forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

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

        n_output = np.unique(y_train).shape[0]  # number of class labels
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
                sigma_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                sigma_h = sigma_out.dot(self.w_out.T) * sigmoid_derivative_h

                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = X_train[batch_idx].T.dot(sigma_h)
                grad_b_h = sigma_h.sum(axis=0)

                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = a_h.T.dot(sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h  # bias shouldn't be penalized
                self.w_h = self.w_h - self.eta * delta_w_h
                self.b_h = self.b_h - self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            """ Evaluation """

            y_train_enc = y_train # since its already encoded/ labeled in this case.
            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self.forward(X_train)
            cost = self.compute_cost(y_test = y_train, output=a_out)

            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            test_acc = ((np.sum(y_test == y_test_pred)).astype(np.float) /
                         X_test.shape[0])

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
        return self
