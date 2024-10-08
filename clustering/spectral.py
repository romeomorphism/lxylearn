import numpy as np
from kmeans import KMeans

class SpectralClustering:
    
    def __init__(self, n_clusters, n_neighbors=10, gamma=1.0, affinity='rbf'):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.affinity = affinity

    def fit(self, X):
        """
        params:
        X: (n, d) array
        """

        # Compute the similarity matrix
        if self.affinity == 'rbf':
            S = np.exp(-self.gamma * np.linalg.norm(X[:, None] - X, axis=2) ** 2)

        # Compute the degree matrix
        D = np.diag(np.sum(S, axis=1))
        # Compute the Laplacian matrix
        L = D - S
        # Compute the eigenvectors of the Laplacian matrix
        _, eigenvectors = np.linalg.eigh(L)
        # Select the eigenvectors corresponding to the smallest eigenvalues
        U = eigenvectors[:,:self.n_clusters]
        # Normalize the eigenvectors
        U = U / np.linalg.norm(U, axis=1)[:, None]

        # Apply KMeans to the eigenvectors
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(U)
        self.labels_ = kmeans.labels_

if __name__ == "__main__":
    # Test SpectralClustering and compare with sklearn
    from sklearn.cluster import SpectralClustering as SpectralClustering_sklearn
    X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
    spectral = SpectralClustering(n_clusters=2)
    spectral_sklearn = SpectralClustering_sklearn(n_clusters=2)

    spectral.fit(X)
    spectral_sklearn.fit(X)
    
    print(spectral.labels_)
    print(spectral_sklearn.labels_)

