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
        
        # Ensure eigenvectors have consistent signs (first element positive)
        for i in range(eig_vecs.shape[1]):
            if eig_vecs[0, i] < 0:
                eig_vecs[:, i] = -eig_vecs[:, i]
        
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
            # Find the minimum number of components that meet the target
            # Use searchsorted with side='right' to find the first index where cum_var > target
            idx = np.searchsorted(cum_var_exp, self.target_explained_variance, side='right')
            k = idx
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

from sklearn.datasets import fetch_openml
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X_train, y_train = fashion_mnist["data"], fashion_mnist["target"]

X_train = X_train[:1500]
y_train = y_train[:1500]

pca_handler = PCA(target_explained_variance=0.99)
X_train_updated = pca_handler.fit(X_train)



print(f"Kích thước ban đầu: {X_train.shape}")
print(f"Kích thước sau khi giảm chiều: {X_train_updated.shape}")

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

cov_mat_act = pca_handler.compute_cov(X_std_act, mean_vec_act) 

cov_mat_exp = [[ 1.33333333, 0.97573583, 0.44021511, 0.02776305],
 [ 0.97573583, 1.33333333, 0.88156376, 0.14760488],
 [ 0.44021511, 0.88156376, 1.33333333, -0.82029039],
 [ 0.02776305, 0.14760488, -0.82029039, 1.33333333]]

assert pytest.approx(cov_mat_act, 0.01) == cov_mat_exp, "Check compute_cov function"

eig_vals_act, eig_vecs_act = pca_handler.compute_eigen_vector(cov_mat_act) 

eig_vals_exp = [2.96080083e+00, 1.80561744e+00, 5.66915059e-01, 7.86907276e-17]

eig_vecs_exp = [[ 0.50989282,  0.38162981,  0.72815056,  0.25330765],
 [ 0.59707545,  0.33170546, -0.37363029, -0.62759286],
 [ 0.57599397, -0.37480162, -0.41446394,  0.59663585],
 [-0.22746684,  0.77708038, -0.3980161,   0.43126337]]

assert pytest.approx(eig_vals_act, 0.01) == eig_vals_exp, "Check compute_eigen_vector function"

for act, exp in zip(eig_vecs_act, eig_vecs_exp):
    assert pytest.approx(act, 0.01) == exp, "Check compute_eigen_vector function"

    pca_handler.feature_size = X.shape[1]
var_exp_act = pca_handler.compute_explained_variance(eig_vals_act) 

var_exp_exp = [0.5551501556710813, 0.33855327084133857, 0.10629657348758019, 1.475451142706682e-17]

assert pytest.approx(var_exp_act, 0.01) == var_exp_exp, "Check compute_explained_variance function"

# eig_pairs = np.array([(2.9608008302457662, np.array([ 0.50989282,  0.59707545,  0.57599397, -0.22746684])),
# (1.8056174444871387, np.array([ 0.38162981,  0.33170546, -0.37480162,  0.77708038])),
# (0.5669150586004276, np.array([ 0.72815056, -0.37363029, -0.41446394, -0.3980161 ])), 
# (7.869072761102302e-17, np.array([ 0.25330765, -0.62759286,  0.59663585,  0.43126337]))])

eig_pairs = [
    (2.9608008302457662, np.array([0.50989282, 0.59707545, 0.57599397, -0.22746684])),
    (1.8056174444871387, np.array([0.38162981, 0.33170546, -0.37480162, 0.77708038])),
    (0.5669150586004276, np.array([0.72815056, -0.37363029, -0.41446394, -0.3980161])),
    (7.869072761102302e-17, np.array([0.25330765, -0.62759286, 0.59663585, 0.43126337]))
]


cum_var_exp = [0.55515016, 0.89370343, 1, 1]

matrix_w_exp = [[0.50989282, 0.38162981], 
                [0.59707545, 0.33170546], 
                [0.57599397, -0.37480162], 
                [-0.22746684, 0.77708038]]

matrix_w_act = pca_handler.compute_weight_matrix(eig_pairs=eig_pairs, cum_var_exp=cum_var_exp)

for act, exp in zip(matrix_w_act, matrix_w_exp):
    assert pytest.approx(act, 0.001) == exp, "Check compute_weight_matrix function"