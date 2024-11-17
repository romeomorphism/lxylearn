import numpy as np
from kmeans import KMeans

class SpectralClustering:
    
    def __init__(self, n_clusters, n_neighbors=10, affinity='rbf', gamma=None):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors

        self.affinity = affinity
        if self.affinity == 'rbf' and gamma is None:
            self.gamma = 1.0
        else:
            self.gamma = gamma

    def fit(self, X, normalized=False):
        """
        params:
        X: (n, d) array
        """

        # Compute the distance matrix
        distance_matrix = np.linalg.norm(X[:, None] - X, axis=2)

        # Compute the similarity matrix
        if self.affinity == 'rbf':
            S = np.exp(-self.gamma * distance_matrix ** 2)
            
        elif self.affinity == 'nearest_neighbors':
            S = np.zeros((X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                nearest_neighbors = np.argsort(distance_matrix[i])[:self.n_neighbors]
                S[i, nearest_neighbors] = 1
                S[nearest_neighbors, i] = 1

        # Compute the degree matrix
        D = np.diag(np.sum(S, axis=1))
        # Compute the Laplacian matrix
        L = D - S
        if normalized:
            D_sqrt_inv = np.diag(1 / np.sqrt(np.sum(S, axis=1)))
            L = D_sqrt_inv @ L @ D_sqrt_inv
        # Compute the eigenvectors of the Laplacian matrix
        _, eigenvectors = np.linalg.eigh(L)
        # Select the eigenvectors corresponding to the smallest eigenvalues
        self.U_ = eigenvectors[:,:self.n_clusters]
        # Normalize the eigenvectors
        self.U_ = self.U_ / np.linalg.norm(self.U_, axis=1)[:, None]

        # Apply KMeans to the eigenvectors
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(self.U_)
        self.L_ = L
        self.labels_ = kmeans.labels_

if __name__ == "__main__":
    # Test SpectralClustering and compare with sklearn
    from sklearn.cluster import SpectralClustering as SpectralClustering_sklearn
    from sklearn import datasets

    n_samples = 500
    X = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05
    )
    spectral = SpectralClustering(n_clusters=2)
    spectral_sklearn = SpectralClustering_sklearn(n_clusters=2)

    spectral.fit(X)
    spectral_sklearn.fit(X)
    
    print(spectral.labels_)
    print(spectral_sklearn.labels_)

