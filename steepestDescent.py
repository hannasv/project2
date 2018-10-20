class steepestDescent(object):
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
    def __init__(self, gradient, eta=0.1, n_iter, tolerance=1e-14, random_state):

        max_iter = n_iter


