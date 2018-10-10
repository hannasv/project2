# -*- coding: utf-8 -*-
#
# model_selection.py
#
# The module is part of model_comparison.
#

"""
The hyperparameter grid search framework.
"""

__author__ = 'Hanna Svennevik', 'Paulina Tedesco'
__email__ = 'hanna.svennevik@fys.uio.no', 'paulinatedesco@gmail.com'


import numpy as np
from utils import bootstrap, mean_squared_error, train_test_split, r2_score

class GridSearchNew:

    """
    Determines optimal hyperparameter for given algorithm, without resampling.

    """

    def __init__(self, model, params, name):
        self.model = model
        self.params = params
        self.name = name
        self.best_mse = None
        self.best_r2 = None
        self.best_param_mse = None
        self.best_param_r2 = None
        self.best_avg_z_pred_mse = None
        self.avg_z_pred = None
        self.best_avg_z_pred_r2 = None
        self.mse_test = None
        self.mse_train = None
        self.r2_test = None
        self.r2_train = None

    def fit(self, X, z, split_size):
        """Searches for the optimal hyperparameter combination."""
        # model and params are now lists --> sende med navn istedenfor.
        # Setup
        self.results = {self.name: []}
        self.train_scores_mse, self.test_scores_mse= [], []
        self.train_scores_r2, self.test_scores_r2 = [], []

        # Splitting our original dataset into test and train.
        X_train, X_test, z_train, z_test = train_test_split(
            X, z, split_size=split_size, random_state=105
        )

        " Returning these dictionaries to plot mse vs model"
        self.mse_test = []
        self.mse_train = []
        self.r2_test = []
        self.r2_train = []
        self.z_pred = []
        # For en model tester vi alle parameterne og returnerer denne.
        for param in self.params:
            estimator = self.model(lmd=param)
            # Train a model for this pair of lambda and random state
            estimator.fit(X_train, z_train)
            temp = estimator.predict(X_test)
            temp2 = estimator.predict(X_train)
            self.mse_test.append(mean_squared_error(z_test, temp))
            self.mse_train.append(mean_squared_error(z_train, temp2))
            self.r2_test.append(r2_score(z_test, temp))
            self.r2_train.append(r2_score(z_train, temp2))
            self.z_pred.append(temp)

        return self
