# Project 2 in fys-stk4155 fall 2018
This cite contains all the material for the assignment project 2 in the course fys-stk4155 at UiO.

## Structure and implementations
The programs are written inn python scrips and the results are produced in the notebooks Project2*.ipynb. The ann.py file contains the code for a neural network both for classification and regression. The file classifier.py contains the program for doing a logistic regression. We have reused the code from project1. Each linear regression method implemented as a class in the file algorithms.py. These can be compared by running, model _comparison.py and for making the bias varince analysis we use the resample.py program. All the programs have their own class functions fit and predict. The folder with results contain all the figures created with the notebooks.

## Abstract


This study provides an assessment of various state of the art methods used in machine learning for solving regression and classification problems, applied to the widely used Ising model. First, we estimate the coupling constant of the one-dimensional Ising model by applying linear regression and neuronal networks. The LASSO is the only approach that reproduces the original pattern from which the data was generated, breaking the symmetry. The optimal regularisation parameter is \lamba = 0.001, resulting in an MSE = 3.07, much lower than the error of the other two schemes (the OLS and ridge). Neuronal networks improve slightly the results giving, in the best case, an MSE = 2.48 - without regularisation, using \eta =0.001, 50 epochs, and a batch size of 10 data points.   

In the second part, we determine the phase of the two-dimensional Ising model in a $40x40$ lattice. Logistic regression and neuronal networks are used for this purpose. For 50 epochs, ridge performs better than LASSO (with an accuracy of 0.70 versus 0.68). However, the highest accuracy achieved with logistic regression is 0.71 (LASSO, 10 epochs). On the other hand, MLPClassier gave an accuracy of $0.\overline{99}$ for the best subset of hyperparameters. This is nearly a perfect classifier. 

In both cases, we explore different ranges of the hyperparameters, such as the learning rate, the number of iterations, and the regularisation parameter, attempting to find the optimal values for each problem. A variety of activation functions were tested, but we present our main results for the sigmoid function. 

The research accomplished by Metha et al (2018), validates our results and, consequently,  the implementation of the algorithms, which is a central part of this work. We emphasise that a big part of the value of this study relies on the notebooks and the developed code, available on the GitHub repository for further contributions of the scientific community, as well as comparisons using different sets of hyperparameters and/or potentially new methods.

