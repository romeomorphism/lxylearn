import numpy as np
class GMM():
    def __init__(self, k, ndim, centers=None, sigma=None, weights=None):
        self.k = k
        self.ndim = ndim

        if centers is not None:
            self.centers = np.asarray(centers)
            self.ndim = self.centers.shape[1]

        if sigma is not None:
            self.sigma = np.asarray(sigma)

        if weights is None:
            self.weights = np.ones(self.k) / self.k
        else:
            self.weights = np.asarray(weights)

    def sample(self, n_samples, latent=False):
        samples = np.zeros((n_samples, self.ndim))
        z = np.zeros((n_samples, self.k))
        z[(range(n_samples), np.random.choice(self.k, n_samples, p=self.weights))] = 1

        for i in range(self.k):
            # samples += np.random.multivariate_normal(self.centers[i], self.sigmas[i], n_samples) * z[:,i].reshape((-1,1))
            samples += np.random.multivariate_normal(self.centers[i], self.sigma, n_samples) * z[:,i].reshape((-1,1))
        if latent:
            return samples, z
        return samples

