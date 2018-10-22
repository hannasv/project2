import numpy as np

class SteepestDescent:
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

    ATTRIBUTES:

    """
    def __init__(self, gradient, eta=0.1, n_iter=50, tolerance=1e-14, random_state=105):

        self.gradient = gradient
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

        max_iter = self.n_iter

        # residuals
        cost = 1  # TODO: initialize this in a better way

        # Collect the cost values in a list to check whether the algorithm converged after training
        self.cost_ = []

        i = 0
        while i < max_iter or cost >= self.tolerance:
            net_input = np.dot(X, self.w_[1:]) + self.w_[0]
            output = self.activation(net_input)
            r = (y - output)
            self.w_[1:] = self.w_[1:] + self.eta * X.T.dot(r)
            self.w_[0] = self.w_[0] +self.eta * r.sum()
            cost = (r**2).sum() / 0.2
            self.cost_.append(cost)

        return self


    def activation(self, net_input):
        """compute linear activation function"""
        # TODO: add other activation functions
        activated = net_input
        return activated
