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
        self.w_ = np.random.randn(2, 1)

        #max_iter = self.n_iter



        # residuals
        cost = 1  # TODO: initialize this in a better way

        # Collect the cost values in a list to check whether the algorithm converged after training
        #self.cost_ = []

        for epoch in range(self.n_epochs):
            for i in range(self.m):
                random_index = np.random.randint(self.m)
                xi = X[random_index:random_index + 1]
                yi = y[random_index:random_index + 1]
                gradients = 2 * xi.T.dot(xi.dot(self.w_) - yi)
                eta = self.learning_schedule(epoch * self.m + i)
                self.w_ = self.w_- eta * gradients
        print("theta from own sdg" + str(theta))

        return self

    def learning_schedule(t):
        t0, t1 = 5, 50
        return t0 / (t + t1)

    def activation(self, net_input):
        """compute linear activation function"""
        # TODO: add other activation functions
        activated = net_input
        return activated
