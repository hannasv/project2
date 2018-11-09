import pickle
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as skms
import sklearn.linear_model as skl
import sklearn.metrics as skm
import tqdm
import copy
import time
from IPython.display import display
from annCopy import NeuralNetMLP

cwd = os.getcwd()
filenames = glob.glob(os.path.join(cwd, 'files*'))
label_filename = "/home/hanna/project2/files/Ising2DFM_reSample_L40_T=All_labels.pkl"

# Read in the labels
with open(label_filename, "rb") as f:
    labels = pickle.load(f)

dat_filename = "/home/hanna/project2/files/Ising2DFM_reSample_L40_T=All.pkl"

# Read in the corresponding configurations
with open(dat_filename, "rb") as f:
    data = np.unpackbits(pickle.load(f)).reshape(-1, 1600).astype("int")

# Set spin-down to -1
data[data == 0] = -1

# Set up slices of the dataset
ordered = slice(0, 70000)
critical = slice(70000, 100000)
disordered = slice(100000, 160000)

X = np.concatenate((data[ordered], data[disordered]))
Y = np.concatenate((labels[ordered], labels[disordered]))

from utils import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, split_size=0.5, random_state=0)

metric = []

ann = NeuralNetMLP(n_hidden=30,
                   epochs=50,
                   eta=0.1,
                   shuffle=True,
                   batch_size=10,
                   activation='sigmoid',
                   tpe = "logistic")

ann.fit(X_train, y_train, X_test, y_test)
ann.predict(X_test)
# returns a list of the mean mse score for different epochs or batches

metric.append(ann.eval_["valid_preform"])
print("Sigmoid for nr of epochs "+str(10) + " and eta: " + str(0.001) + "  batchsize = " + str(10) +"   performance is " + str(np.mean(ann.eval_["valid_preform"])))
print("---------------------------")
