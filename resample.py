# send in dictionaries with their best lmd values.
from utils import bootstrap, train_test_split, variance, mean_squared_error, r2_score, ci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resample(models, lmd, X, z, nboots, split_size=0.2):

    """ Dictionaires to keep track of the results  """
    z_test = {"ridge": [], "lasso": [], "ols": []}
    z_pred_test = {"ridge": [], "lasso": [], "ols": []}

    bias = {"ridge": [], "lasso": [], "ols": []}
    var = {"ridge": [], "lasso": [], "ols": []}
    beta = {"ridge": [], "lasso": [], "ols": []}

    mse_test = {"ridge": [], "lasso": [], "ols": []}
    #r2_test = {"ridge": [], "lasso": [], "ols": []}
    "       ----------------------"
    mse_train = {"ridge": [], "lasso": [], "ols": []}
    #r2_train = {"ridge": [], "lasso": [], "ols": []}

    np.random.seed(2018)

    # Spilt the data in tran and split
    X_train, X_test, z_train, z_test_ = train_test_split(
        X, z, split_size=split_size
    )

    # # extract data from design matrix
    # x = X[:, 1]
    # y = X[:, 2]
    # x_test = X_test[:, 1]
    # y_test = X_test[:, 2]

    for name, model in models.items():
        # creating a model with the previosly known best lmd
        estimator = model(lmd[name])
        # Train a model for this pair of lambda and random state
        """  Keeping information for test """
        estimator.fit(X_train, z_train)
        z_pred_test_ = np.empty((z_test_.shape[0], nboots))
        z_pred_train_ = np.empty((z_train.shape[0], nboots))
        beta_ = np.empty((X.shape[1], nboots))
        for i in range(nboots):
            X_, z_ = bootstrap(X_train, z_train, i)  # i is now also the random state for the bootstrap

            estimator.fit(X_, z_)
            # Evaluate the new model on the same test data each time.
            z_pred_test_[:, i] = np.squeeze(estimator.predict(X_test))
            z_pred_train_[:, i] = np.squeeze(estimator.predict(X_train))
            beta_[:, i] = np.squeeze(estimator.coef_)

        beta[name] = beta_
        z_pred_test[name] = z_pred_test_

        z_test_ = z_test_.reshape((z_test_.shape[0], 1))
        z_test[name] = z_test_
        mse_test[name] = (np.mean(np.mean((z_test_ - z_pred_test_) ** 2, axis=1, keepdims=True)))
        bias[name] = np.mean((z_test_ - np.mean(z_pred_test_, axis=1, keepdims=True)) ** 2)
        var[name] = np.mean(np.var(z_pred_test_, axis=1, keepdims=True))

        z_train = z_train.reshape((z_train.shape[0], 1))
        mse_train[name] = np.mean(np.mean((z_train - z_pred_train_) ** 2, axis=1, keepdims=True))

        # print('Error:', mse_test)
        # print('Bias^2:', bias)
        # print('Var:', var)
        # print('{} >= {} + {} = {}'.format(mse_test, bias, variance, bias + variance))

        # plt.figure(1, figsize=(11, 7))
        # plt.subplot(121)
        # plt.plot(x, z, label='f')
        # plt.scatter(x_test, z_test, label='Data points')
        # plt.scatter(x_test, np.mean(z_pred, axis=1), label='Pred')
        # plt.legend()
        # plt.xlabel('x')
        # plt.ylabel('z')
        #
        # plt.subplot(122)
        # plt.plot(y, z, label='f')
        # plt.scatter(y_test, z_test, label='Data points')
        # plt.scatter(y_test, np.mean(z_pred, axis=1), label='Pred')
        # plt.legend()
        # plt.xlabel('y')
        # plt.ylabel('z')
        # plt.show()


        # Confidence intervals
        ci_beta = np.empty((2, beta_.shape[0]))
        poly = []
        for p in range(beta_.shape[0]):
            ci_beta[:, p] = np.array(ci(beta_[p, :])).T
            poly.append(p)

        # plt.plot(poly, ci_beta[0, :], label='Upper CI (95%)')  # --> Vise i tabell
        # plt.plot(poly, np.mean(beta, axis=1), label='Beta')
        # plt.plot(poly, ci_beta[1, :], label='Lower CI (95%)')
        # plt.legend()
        # plt.show()

    return z_test, z_pred_test, bias, var, beta, mse_test, mse_train, ci_beta
