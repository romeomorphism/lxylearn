import numpy as np
import pandas as pd
from abc import abstractmethod, ABCMeta

class MixtureModel(metaclass=ABCMeta):
    """
    The base class for mixture models with a discrete latent variable
    """
    def __init__(self, k, ndim, weights=None):
        self.k = k
        self.ndim = ndim

        if weights is None:
            self.weights = np.ones(self.k) / self.k
        else:
            self.weights = np.asarray(weights)
    
    def generate_labels(self, n_samples):
        """
        Generate the labels for the samples
        """
        z = np.zeros((n_samples, self.k))
        z[(range(n_samples), np.random.choice(self.k, n_samples, p=self.weights))] = 1
        return z
    
    @abstractmethod
    def sample(self, n_samples):
        """Generate samples from the model"""
        pass

    @abstractmethod
    def sample_df(self, n_samples):
        """Generate samples from the model and store them into a DataFrame"""
        pass

    @abstractmethod
    def plot_cluster(self, cluster=0, n_samples=1000):
        pass

class GMM(MixtureModel):
    """
    Gaussian Mixture Model
    """
    def __init__(self, k, ndim, centers=None, sigma=None, weights=None):
        super(GMM, self).__init__(k, ndim, weights=weights) # call the __init__ of the GMM parent class

        if centers is not None:
            self.centers = np.asarray(centers)
            self.ndim = self.centers.shape[1]

        if sigma is not None:
            self.sigma = np.asarray(sigma)

    def sample(self, n_samples, latent=False):
        """generate samples from the model and store them into a nArray"""
        samples = np.zeros((n_samples, self.ndim))
        z = self.generate_labels(n_samples)

        for i in range(self.k):
            # samples += np.random.multivariate_normal(self.centers[i], self.sigmas[i], n_samples) * z[:,i].reshape((-1,1))
            samples += np.random.multivariate_normal(self.centers[i], self.sigma, n_samples) * z[:,i].reshape((-1,1))
        if latent:
            return samples, z
        return samples
    
    def sample_df(self, n_samples):
        """generate samples from the model and store them into a DataFrame"""
        z = self.generate_labels(n_samples)
        samples = np.zeros((n_samples, self.ndim))
        for i in range(self.k):
            samples += np.random.multivariate_normal(self.centers[i], self.sigma, n_samples) * z[:,i].reshape((-1,1))
        
        df = pd.DataFrame(samples, columns=[f'x{i}' for i in range(self.ndim)])
        df['cluster'] = np.argmax(z, axis=1)
        return df
    
    def plot_cluster(self, cluster=0, n_samples=1000):

        if cluster >= self.k:
            raise ValueError(f'cluster must be less than {self.k}')
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        df = self.sample_df(n_samples)
        sns.scatterplot(x='x0', y='x1', data=df[df['cluster'] == cluster], color='blue')
        sns.scatterplot(x='x0', y='x1', data=df[df['cluster'] != cluster], color='gray')
        plt.show()



if __name__ == '__main__':
    # Example
    k = 3
    ndim = 2
    centers = [[0, 0], [1, 1], [2, 2]]
    sigma = np.eye(ndim) * 0.1
    gmm = GMM(k, ndim, centers=centers, sigma=sigma)

    samples = gmm.sample(1000)
    print(samples)
    print(samples.shape)

    df = gmm.sample_df(1000)
    print(df.head())
    print(df.shape)
    gmm.plot_cluster(cluster=1, n_samples=1000)
    gmm.fit()