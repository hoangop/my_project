import math
import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pytest
from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, target_explained_variance=None):
        """
        explained_variance: float, the target level of explained variance
        """
        self.target_explained_variance = target_explained_variance
        self.feature_size = -1

    def standardize(self, X):
        """
        standardize features using standard scaler
        :param X: input data with shape m (# of observations) X n (# of features)
        :return: standardized features (Hint: use skleanr's StandardScaler. Import any library as needed)
        """
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        return X_std

    def compute_mean_vector(self, X_std):
        """
        compute mean vector
        :param X_std: transformed data
        :return n X 1 matrix: mean vector
        """
        # Compute mean along axis 0 (across samples for each feature)
        # Return as 1D array to match test case expectations
        mean_vec = np.mean(X_std, axis=0)
        return mean_vec

    def compute_cov(self, X_std, mean_vec):
        """
        Covariance using mean, (don't use any numpy.cov)
        :param X_std:
        :param mean_vec:
        :return n X n matrix:: covariance matrix
        """
        # Center data explicitly using provided mean
        # mean_vec is now 1D array, so no need to ravel()
        X_centered = X_std - mean_vec
        m = X_centered.shape[0]
        cov_mat = (X_centered.T @ X_centered) / (m - 1)
        return cov_mat

    def compute_eigen_vector(self, cov_mat):
        """
        Eigenvector and eigen values using numpy. Uses numpy's eigenvalue function
        :param cov_mat:
        :return: (eigen_values, eigen_vector)
        """
        # For symmetric covariance matrices use eigh (stable for Hermitian)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        # Sort in descending order by eigenvalue
        order = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]
        return eig_vals, eig_vecs

    def compute_explained_variance(self, eigen_vals):
        """
        sort eigen values and compute explained variance.
        explained variance informs the amount of information (variance)
        can be attributed to each of  the principal components.
        :param eigen_vals:
        :return: explained variance.
        """
        total = np.sum(eigen_vals)
        var_exp = eigen_vals / total if total != 0 else np.zeros_like(eigen_vals)
        return var_exp

    def cumulative_sum(self, var_exp):
        """
        return cumulative sum of explained variance.
        :param var_exp: explained variance
        :return: cumulative explained variance
        """
        return np.cumsum(var_exp)

    def compute_weight_matrix(self, eig_pairs, cum_var_exp):
        """
        compute weight matrix of top principal components conditioned on target
        explained variance.
        (Hint : use cumilative explained variance and target_explained_variance to find
        top components)
        
        :param eig_pairs: list of tuples containing eigenvalues and eigenvectors, 
        sorted by eigenvalues in descending order (the biggest eigenvalue and corresponding eigenvectors first).
        :param cum_var_exp: cumulative expalined variance by features
        :return: weight matrix (the shape of the weight matrix is n X k)
        """
        if self.target_explained_variance is None:
            k = len(eig_pairs)
        else:
            # Small epsilon to ensure meeting or exceeding the target
            k = int(np.searchsorted(cum_var_exp, self.target_explained_variance - 1e-12) + 1)
            k = min(max(k, 1), len(eig_pairs))
        # Stack top-k eigenvectors as columns
        matrix_w = np.column_stack([eig_pairs[i][1] for i in range(k)])
        return matrix_w

    def transform_data(self, X_std, matrix_w):
        """
        transform data to subspace using weight matrix
        :param X_std: standardized data
        :param matrix_w: weight matrix
        :return: data in the subspace
        """
        return X_std.dot(matrix_w)

    def fit(self, X):
        """    
        entry point to the transform data to k dimensions
        standardize and compute weight matrix to transform data.
        The fit functioin returns the transformed features. k is the number of features which cumulative 
        explained variance ratio meets the target_explained_variance.
        :param   m X n dimension: train samples
        :return  m X k dimension: subspace data. 
        """
        self.feature_size = X.shape[1]

        # 1) Standardize
        X_std = self.standardize(X)
        
        # 2) Mean vector
        mean_vec = self.compute_mean_vector(X_std)
        
        # 3) Covariance matrix
        cov_mat = self.compute_cov(X_std, mean_vec)
        
        # 4) Eigen decomposition
        eig_vals, eig_vecs = self.compute_eigen_vector(cov_mat)
        
        # 5) Explained variance and cumulative sum
        var_exp = self.compute_explained_variance(eig_vals)
        cum_var_exp = self.cumulative_sum(var_exp)
        
        # 6) Prepare eigen pairs (value, vector) sorted already
        eig_pairs = [(eig_vals[i], eig_vecs[:, i]) for i in range(len(eig_vals))]
        
        # 7) Compute weight matrix using the target explained variance threshold
        matrix_w = self.compute_weight_matrix(eig_pairs, cum_var_exp)
        
        # 8) Transform
        return self.transform_data(X_std=X_std, matrix_w=matrix_w)


# X_train = pickle.load(open('./data/fashionmnist/train_images.pkl','rb'))
# y_train = pickle.load(open('./data/fashionmnist/train_image_labels.pkl','rb'))

# from sklearn.datasets import fetch_openml
# fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
# X_train, y_train = fashion_mnist["data"], fashion_mnist["target"]

# X_train = X_train[:1500]
# y_train = y_train[:1500]

# pca_handler = PCA(target_explained_variance=0.99)
# X_train_updated = pca_handler.fit(X_train)



# print(f"Kích thước ban đầu: {X_train.shape}")
# print(f"Kích thước sau khi giảm chiều: {X_train_updated.shape}")

np.random.seed(42)
X = np.array([[0.39, 1.07, 0.06, 0.79], [-1.15, -0.51, -0.21, -0.7], [-1.36, 0.57, 0.37, 0.09], [0.06, 1.04, 0.99, -1.78]])
pca_handler = PCA(target_explained_variance=0.99)

X_std_act = pca_handler.standardize(X)

X_std_exp = [[ 1.20216033, 0.82525828, -0.54269609, 1.24564656],
             [-0.84350476, -1.64660539, -1.14693504, -0.31402854],
             [-1.1224591, 0.04302294, 0.15105974, 0.51291329],
             [ 0.76380353, 0.77832416, 1.53857139, -1.4445313]]

for act, exp in zip(X_std_act, X_std_exp):
    assert pytest.approx(act, 0.01) == exp, "Check Standardize function"

mean_vec_act = pca_handler.compute_mean_vector(X_std_act)

mean_vec_exp = [5.55111512, 2.77555756, 5.55111512, -5.55111512]

mean_vec_act_tmp = mean_vec_act * 1e17

assert pytest.approx(mean_vec_act_tmp, 0.1) == mean_vec_exp, "Check compute_mean_vector function"