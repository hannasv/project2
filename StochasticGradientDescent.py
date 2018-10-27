import numpy as np

class gradientDescent:
    """Steepest descent

    Search for the minimum in cost_history and find its corresponding weight in w_history

        PARAMETERS:
            lmd: float
                Regularization parameter. If lmd = 0, OLS method
            eta: float
                Learning rate (between 0.0 and 1.0)
            n_iter: int
                Number of iterations over the training set
            tolerance: float
                Tolerance for the error
            random_state: int
            n_epochs: int
                number of epochs
            m: int
                size of the minibatches

    ATTRIBUTES:

    """
    def __init__(self, lmd, n_epochs, m, eta=0.1, tolerance=1e-14, random_state=105):

        self.lmd = lmd
        self.n_epochs = n_epochs
        self.m = m
        self.eta = eta
        self.tolerance = tolerance
        self.random_state = random_state




    def fit(self, X, y):

        """
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors
        y: {array-like}, shape = [n_samples]
            Target values

        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.7, size = 1)
        self.w_ = np.random.randn(X.shape[1], 1)  # weight with the same size as x?


        self.cost_history = []
        self.w_history

        # Collect the cost values in a list to check whether the algorithm converged after training

        M = X.shape[0]/self.m

        for epoch in range(self.n_epochs):
            # initialize the total loss for the epoch
            epoch_cost = []
            epoch_w_ =[]

            for (batchX, batchY) in self.next_batch(X, y, M):
                net_input = np.dot(batchX, self.w_[1:]) + self.w_[0]
                output = self.activation(net_input)
                r = (y - output)

                cost = np.sum(r ** 2)  # skal vi beregne det sånn? det gjorde vi ikke med steepest.
                epoch_cost.append(cost)

                # Update gradient
                gradient = 2 * (batchX.T.dot(batchX.dot(self.w_) - batchY)
                                + self.lmd*self.w_)
                # eta = self.learning_schedule(epoch * self.m + i) #Skal vi ha det?
                self.w_[1:] = self.w_[1:] - self.eta * gradient
                self.w_[0] = self.w_[0] - self.eta * r.sum()

                epoch_w_.append(self.w_)

            self.cost_history.append(epoch_cost)
            self.w_history.append(epoch_w_)


        return self

    # TODO write a function for the gradient?

    def next_batch(X, y, M):
        # M is the size of the minibatches, M=n/m
        # loop over our dataset `X` in mini-batches of size `M`
        for i in np.arange(0, X.shape[0], M):
            # random?
            # for i in range(M):
            #k = np.random.randint(m)  # Pick the k-th minibatch at random
            # yield a tuple of the current batched data and labels
            yield (X[i:i + M], y[i:i + M])

    def learning_schedule(t):
        # skal vi ha det? Tror vi sa nei
        t0, t1 = 5, 50
        return t0 / (t + t1)

    def activation(self, net_input):
        """compute linear activation function"""
        # TODO: add other activation functions
        activated = net_input
        return activated
