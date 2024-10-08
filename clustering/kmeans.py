import numpy as np

class KMeans:
    def __init__(self, n_clusters, init='random', max_iter=100, algorithm="lloyd"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        if type(init) is not str:
            self.centers_ = np.asarray(init)
    def fit(self, X, plot=False):
        """
        params:
        X: (n, d) array
        """
        if self.init == 'random':
            self.centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            # Assign each data point to the closest center
            self.labels_ = np.argmin(np.linalg.norm(X[:, None] - self.centers_, axis=2), axis=1)
            # Update the centers
            new_centers = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(new_centers == self.centers_):
                break
            self.centers_ = new_centers
        if plot:
            import matplotlib.pyplot as plt
            plt.scatter(X[:, 0], X[:, 1], c=self.labels_)
            plt.scatter(self.centers_[:, 0], self.centers_[:, 1], c='red')
            plt.show()
        
    def predict(self, X):
        """
        params:
        X: (n, d) array
        return:
        labels: (n,) array
        """
        return np.argmin(np.linalg.norm(X[:, None] - self.centers_, axis=2), axis=1)
    
if __name__ == "__main__":
    # Test KMeans and compare with sklearn
    from sklearn.cluster import KMeans as KMeans_sklearn
    X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2)
    kmeans_sklearn = KMeans_sklearn(n_clusters=2)

    kmeans.fit(X)
    kmeans_sklearn.fit(X)
    
    print(kmeans.labels_)
    print(kmeans_sklearn.labels_)