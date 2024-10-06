"""
Created on Fri Jul 12 2024 by Xinyu Liu

The implementation of PCA (Principal Component Analysis) algorithm.
PCA is a linear dimensionality reduction technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space.
The goal of PCA is to find the directions (principal components) that maximize the variance in the data or minimizing the loss of linear projection.
"""

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        """
        params:
        X: (n, d) array
        """
        # Mean centering
        self.mean_ = np.mean(X, axis=0) #(d,)
        X = X - self.mean_ #(n, d)

        # Perform the singular value decomposition
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[:self.n_components] #(n_components, d)
        self.singular_values_ = S[:self.n_components] #(n_components,)

        # Compute the explained variance (use n_samples - 1 degree of freedom)
        self.explained_variance_ = self.singular_values_ ** 2 / (X.shape[0] - 1) #(n_components,)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)


    def fit_transform(self, X):
        """
        params:
        X: (n, d) array
        return:
        X_new: (n, n_components) array
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        params:
        X: (n, d) array
        return:
        X_new: (n, n_components) array
        """
        X = X - self.mean_
        return np.dot(X, self.components_.T)
    
    def inverse_transform(self, X_new):
        """
        params:
        X_new: (n, n_components) array
        return:
        X: (n, d) array
        """
        return np.dot(X_new, self.components_) + self.mean_


if __name__ == "__main__":
    # Test PCA and compare with sklearn
    from sklearn.decomposition import PCA as PCA_sklearn
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca_sklearn = PCA_sklearn(n_components=2)

    pca.fit(X)
    pca_sklearn.fit(X)
    print(pca.explained_variance_ratio_)

    print(pca.transform(X))
    print(pca_sklearn.transform(X))