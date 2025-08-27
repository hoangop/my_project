import math
import pickle
import gzip
import numpy as np
import matplotlib.pylab as plt
#%matplotlib inline

# importing all the required libraries

from math import exp
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# This cell sets up the MNIST dataset 

class MNIST_import:
    """
    sets up MNIST dataset from OpenML 
    """
    def __init__(self):
        
        df = pd.read_csv("tree_decisiom/data/mnist_784.csv")
        
        # Create arrays for the features and the response variable
        # store for use later 
        y = df['class'].values
        X = df.drop('class', axis=1).values
         
        # Convert the labels to numeric labels
        y = np.array(pd.to_numeric(y))
        
        # create training and validation sets 
        self.train_x, self.train_y = X[:5000,:], y[:5000]
        self.val_x, self.val_y = X[5000:6000,:], y[5000:6000]
        
data = MNIST_import()

def view_digit(x, label=None):
    fig = plt.figure(figsize=(3,3))
    plt.imshow(x.reshape(28,28), cmap='gray');
    plt.xticks([]); plt.yticks([]);
    if label: plt.xlabel("true: {}".format(label), fontsize=16)
    #plt.show()


training_index = 9
view_digit(x=data.train_x[training_index], label=data.train_y[training_index])

# Here are the numbers you need to provide here:
num_training_examples = 0
num_test_examples = 0
pixels_per_image = 0

# your code here
num_training_examples = data.train_x.shape[0]
num_test_examples = data.val_x.shape[0]
pixels_per_image = data.train_x.shape[1]

# print(num_training_examples)
# print(num_test_examples)
# print(pixels_per_image)

class KNN:
    """
    Class to store data for regression problems 
    """
    def __init__(self, x_train, y_train, K=5):
        """
        Creates a kNN instance
        :param x_train: numpy array with shape (n_rows,1)- e.g. [[1,2],[3,4]]
        :param y_train: numpy array with shape (n_rows,)- e.g. [1,-1]
        :param K: The number of nearest points to consider in classification
        """
        from sklearn.neighbors import BallTree
        self.balltree = BallTree(x_train)
        self.x_train = x_train
        self.y_train = y_train
        self.K = K 

    def majority(self, neighbor_indices, neighbor_distances=None):
        """
        Given indices of nearest neighbors in training set, return the majority label. 
        Break ties by considering 1 fewer neighbor until a clear winner is found. 
        :param neighbor_indices: The indices of the K nearest neighbors in self.X_train 
        :param neighbor_distances: Corresponding distances from query point to K nearest neighbors. 
        """
        k = len(neighbor_indices)
        while k > 0:
            labels = self.y_train[neighbor_indices[:k]]
            vals, counts = np.unique(labels, return_counts=True)
            max_count = np.max(counts)
            winners = vals[counts == max_count]
            if len(winners) == 1:
                return winners[0]
            k -= 1
        # fallback (should not happen)
        return self.y_train[neighbor_indices[0]]

    def classify(self, x):
        """
        Given a query point, return the predicted label 
        :param x: a query point stored as an ndarray  
        """
        dist, ind = self.balltree.query(x.reshape(1, -1), k=self.K)
        return self.majority(ind[0], dist[0])

    def predict(self, X):
        """
        Given an ndarray of query points, return yhat, an ndarray of predictions 
        :param X: an (m x p) dimension ndarray of points to predict labels for 
        """
        yhat = np.array([self.classify(x) for x in X])
        return yhat
    

    #test

    # Sample tests for KNN class

import pytest
# set-up
X_train = np.array([[1,6], [6,4], [2,5], [1,3], [2,2], [3,1], [1,5], [2,3], [4,6], [3,5], [6,5], [0,4]])
y_train = np.array([+1, -1, +1, +1, -1, +1, +1, -1, +1, -1, +1, -1])
x = np.array([0,5])

# test k=2,
k2nn = KNN(X_train, y_train, K=2)
assert -1 == pytest.approx(k2nn.classify(x)), "KNN class doesn't perform as expected with two neighbors"


# test k=3
k3nn = KNN(X_train, y_train, K=3) 
assert 1 == pytest.approx(k3nn.classify(x)), "KNN class doesn't perform as expected with three neighbors"


# test 3NN Predict 
X = np.array([[2,5], [5,1]])
k3p = KNN(X_train, y_train, K=3) 
yhat = k3p.predict(X)

# correct labels for the above two points(X).
ytrue = [1, -1]

for yh, yt in zip(yhat, ytrue):
    assert yh == yt, "Look at the predict function in the KNN class."

# use your KNN class to perform KNN on the validation data with K = 3
knn = KNN(data.train_x, data.train_y, K=3)
val_yhat = knn.predict(data.val_x)

# create a confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data.val_y, val_yhat)
print("Confusion Matrix:")
print(cm)

print(data.val_x.shape)

acc = []
allks = range(1, 30)

for k in allks:
    knn = KNN(data.train_x, data.train_y, K=k)
    val_yhat = knn.predict(data.val_x)
    accuracy = np.mean(val_yhat == data.val_y)
    acc.append(accuracy)

# you can use this code to create your plot    
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))
ax.plot(allks, acc, marker="o", color="steelblue", lw=3, label="unweighted")
ax.set_xlabel("number neighbors", fontsize=16)
ax.set_ylabel("accuracy", fontsize=16)
plt.xticks(range(1, 31, 2))
ax.grid(alpha=0.25)
ax.legend()
plt.title("KNN Accuracy on Validation Set vs. Number of Neighbors", fontsize=16)
plt.show()

