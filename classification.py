import numpy as np
import Costfunctions

class LogisticRegression(object):
    """Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int (defalt 50)
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
     lmd: penalty


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Logistic cost function value in each epoch.
    """
    def __init__(self, eta, random_state, key, dm ="steepest", shuffle = True, n_iter = 100, batch_size = 10, epochs=100,lmd = 0, tolerance=1e-14):
        self.eta = eta
        self.n_iter = n_iter
        self.random = np.random.RandomState(random_state)
        self.key = key
        self.lmd = lmd
        self.tol = tolerance
        self.shuffle = shuffle
        self.dm = dm # descent method
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_ = None
        self.cost_=None
        self.p = None
        self.w_ = None
        self.b_ = None

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the numb
          er of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """

        func = {
            'ols' : Costfunctions.Cost_OLS,
            'ridge' : Costfunctions.Cost_Ridge,
            'lasso' : Costfunctions.Cost_Lasso,
        }

        #initialization weights to be random numbers from -0.7 to 0.7
        #rgen = np.random.RandomState(self.random_state)
        self.w_ = 0.1*np.random.randn(X.shape[1])
        #np.random.uniform(-0.7,0.7, size = X.shape[1])
        self.b_ = 1
        #np.random.uniform(-0.7,0.7, size = 1)

        self.cost_ = []
        costfunc = func[self.key](self.eta, self.lmd)

        self.descent_method(costfunc,X,y)

        return self

    def descent_method(self, costfunc, X, y):
        key = self.dm
        # costfunc.grad(errors)
        if (key == "steepest"):
            #self.w_[1:] += self.eta * grad
            #self.w_[0] += self.eta * errors.sum() # bias
            max_iter = self.n_iter
            i = 0
            while (i < max_iter):# and cost >= self.tol
                # Computing the linar combination of x'es and weights.
                net_input = np.dot(X, self.w_) + self.b_
                output = costfunc.activation(net_input, "sigmoid")
                errors = costfunc.r(y) # calculating the residuals (y-p)
                # calculating the gradient of this particular costfunction
                gradient = costfunc.log_grad(X, self.w_, errors)

                self.w_ = self.w_ -  self.eta * gradient
                self.b_ = self.b_ - self.eta * errors.sum() # bias
                cost = costfunc.log_calculate(X, y, self.w_)
                self.cost_.append(cost)
                i=i+1

        elif(key == "sgd"):

            #self.eval_ = {"loss_history":[]}
            for epoch in range(self.epochs):
                n_samples, n_features = np.shape(X)
                indices = np.arange(n_samples)
                epochLoss = []
                scores_epochs = []

                if self.shuffle:
                    self.random.shuffle(indices)

                for idx in range(0, n_samples, self.batch_size):
                    batch_idx = indices[idx:idx + self.batch_size]
                    batchX = X[batch_idx,:]
                    batchY = y[batch_idx]
                    #print(batchY.shape)
                    net_input = np.dot(batchX, self.w_) + self.b_
                    output = costfunc.activation(net_input, "sigmoid")
                    errors = costfunc.r(batchY)
                    gradient = costfunc.log_grad(batchX, self.w_, errors)/batchY.shape[0]
                    #update weights
                    self.w_ = self.w_ - self.eta * gradient
                    self.b_ = self.b_ - self.eta * errors.sum()

                    score = np.sum(np.where(output >= 0.5, 1, 0) == batchY)/len(output)
                    scores_epochs.append(score)

                #print("score: " + str(np.average(scores_epochs)) + " for epoch:  " + str(epoch))

                epochLoss.append(errors.sum())
                #self.eval_["loss_history"].append(np.mean(epochLoss))
            #print(self.eval_["loss_history"])
            """
            self.cost_history = []
            self.w_history = []

            # Collect the cost values in a list to check whether the algorithm converged after training

            M = int(X.shape[0]/self.batch_size)
            n_samples, n_features = np.shape(X)




            for epoch in range(self.epochs):
                # initialize the total loss for the epoch
                #epoch_cost = []
                #epoch_w_ =[]
                #epoch_b_ =[]
                indices = np.arange(n_samples, dtype = int)
                scores_epochs = []
                #Shuffler mellom hver batch
                if self.shuffle:
                    self.random.shuffle(indices)
                print(self.w_)
                print("----------")
                print(self.b_)
                print("---------")
                #for (batchX, batchY) in self.next_batch(X, y, M):
                for idx in range(0, n_samples, self.batch_size):
                    batch_idx = indices[idx:idx + self.batch_size]
                    batchX = X[batch_idx,:]
                    batchY = y[batch_idx]

                    net_input = np.dot(batchX, self.w_) + self.b_
                    #print(net_input)
                    output = self.activation(net_input ,"sigmoid")

                    r = (batchY - output)
                    #print(r.shape, output.shape)
                    #cost = -y.dot(np.log(self.p+ 1e-12)) - ((1 - y).dot(np.log(1 - self.p + 1e-12)))
                    # skal vi beregne det sånn? det gjorde vi ikke med steepest.
                    #epoch_cost.append(cost)
                    gradient =  batchX.T.dot(r) # + 2*self.lmd*self.w_ --> Bruke penalty senere.
                    #print("eta*grad:    " + str(self.eta * gradient))
                    # eta = self.learning_schedule(epoch * self.m + i) #Skal vi ha det?


                    self.w_ = self.w_ + self.eta * gradient
                    self.b_ = self.b_ + self.eta * r.sum()
                    score = np.sum(np.where(output >= 0.5, 1, 0) == batchY)/len(output)
                    #epoch_w_.append(self.w_)
                    #epoch_b_.append(self.b_)
                    scores_epochs.append(score)

                print("score: " + str(np.average(scores_epochs)) + " for epoch:  " + str(epoch))
                """
                #self.cost_history.append(epoch_cost)
                #self.w_history.append(epoch_w_)
            #print(self.cost_history)
        else:
            print("Unvalid keyword, use: steepest or sgd.")

    #@staticmethod
    def next_batch(X, y, M):
        # M is the size of the minibatches, M=n/m
        # loop over our dataset `X` in mini-batches of size `M`

        """Kan bruke denne, men da må vi shuffle indexer."""
        for i in np.arange(0, X.shape[0], M):
            # random?
            # for i in range(M):
            #k = np.random.randint(m)  # Pick the k-th minibatch at random
            # yield a tuple of the current batched data and labels
            #if(i+M <= X.shape[1]):
            yield (X[i:i + M], y[i:i + M])
            #else:
            #    yield (X[i:], y[i:])

    def activation(self, Xw, key):
        if (key == "sigmoid"):
            self.p = 1. / (1. + np.exp(-Xw))
            return self.p
        elif(key == "ELU"):
            if (Xw >= 0):
                return Xw
            else:
                return alpha*(np.exp(Xw)-1)
        else:
            print("Unvalide keyword argument. Use siogmoid or ELU for activation.")


    def predict(self, X):
        """Uses a linear prediction"""
        net_input = np.dot(X, self.w_) + self.b_
        return np.where(net_input >= 0.5, 1, 0)
