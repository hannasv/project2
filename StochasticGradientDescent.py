import numpy as np

class gradientDescent:
    """Steepest descent

        PARAMETERS:
            gradient: float
                gradient of the cost function
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
    def __init__(self, gradient, n_epochs, m, eta=0.1, n_iter=50, tolerance=1e-14, random_state=105):

        self.gradient = gradient
        self.n_epochs = n_epochs
        self.m = m
        self.eta = eta
        self.n_iter = n_iter
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

        # TODO: CHECK IF WE HAVE A COLUMN WITH ONES IN X

        # self.cost_ = []
        #max_iter = self.n_iter



        # residuals
        cost = 1  # TODO: initialize this in a better way

        # Collect the cost values in a list to check whether the algorithm converged after training
        #self.cost_ = []

        for epoch in range(self.n_epochs):
            # initialize the total loss for the epoch
            epochLoss = []

            for (batchX, batchY) in self.next_batch(X, y, self.m):
                net_input = net_input = np.dot(X, self.w_[1:]) + self.w_[0]
                output = self.activation(net_input)

                # TODO: continue here
                random_index = np.random.randint(self.m)
                xi = X[random_index:random_index + 1]
                yi = y[random_index:random_index + 1]
                gradients = 2 * xi.T.dot(xi.dot(self.w_) - yi)
                eta = self.learning_schedule(epoch * self.m + i)
                self.w_ = self.w_- eta * gradients
        print("theta from own sdg" + str(self.w_))

        return self

    def next_batch(X, y, m):
        # loop over our dataset `X` in mini-batches of size `self.m`
        for i in np.arange(0, X.shape[0], m):
            # yield a tuple of the current batched data and labels
            yield (X[i:i + m], y[i:i + m])

    def learning_schedule(t):
        t0, t1 = 5, 50
        return t0 / (t + t1)

    def activation(self, net_input):
        """compute linear activation function"""
        # TODO: add other activation functions
        activated = net_input
        return activated
