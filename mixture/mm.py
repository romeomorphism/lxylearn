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
        z[(range(n_samples),
           np.random.choice(self.k, n_samples, p=self.weights))] = 1
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
        """Highlight one particular cluster in the plot"""
        pass


class GMM(MixtureModel):
    """
    Gaussian Mixture Model
    """
    def __init__(self, k, ndim, centers=None, sigma=None, weights=None):
        super(GMM, self).__init__(k, ndim, weights=weights)
        # call the __init__ of the GMM parent class

        if centers is not None:
            self.centers = np.asarray(centers)

        if sigma is not None:
            self.sigma = np.asarray(sigma)

    def plot_pdf(self, n_grid_points=500, *grid):
        """
        Plot the PDF of the model
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import norm, multivariate_normal

        if self.ndim == 1:
            if len(grid) == 0:
                x = np.linspace(-5, 5, n_grid_points)
            elif len(grid) == 1:
                x = grid[0]
            else:
                raise ValueError('grid must be a tuple with at most one element')
            y = np.zeros_like(x)
            for i in range(self.k):
                y += self.weights[i] * norm.pdf(x, loc=self.centers[i], scale=self.sigma)
            plt.plot(x, y)
            plt.show()

        elif self.ndim == 2:
            if len(grid) == 0:
                x = np.linspace(-5, 5, n_grid_points)
                y = np.linspace(-5, 5, n_grid_points)
            elif len(grid) == 2:
                x, y = grid
            else:
                raise ValueError('grid must be a tuple with at most two elements')
            X, Y = np.meshgrid(x, y)
            pos = np.dstack((X, Y))
            Z = np.zeros_like(X)
            for i in range(self.k):
                Z += self.weights[i] * multivariate_normal.pdf(pos, mean=self.centers[i], cov=self.sigma)
            sns.heatmap(Z, xticklabels=False, yticklabels=False)
            plt.show()
        
        else:
            if len(grid) == 0:
                x = np.linspace(-5, 5, n_grid_points)
                y = np.linspace(-5, 5, n_grid_points)
            elif len(grid) == 2:
                x, y = grid
            else:
                raise ValueError('grid must be a tuple with at most two elements')
            X, Y = np.meshgrid(x, y)
            pos = np.dstack((X, Y))
            Z = np.zeros_like(X)
            for i in range(self.k):
                Z += self.weights[i] * multivariate_normal.pdf(pos, mean=self.centers[i][:2], cov=self.sigma[:2,:2])
            sns.heatmap(Z, xticklabels=False, yticklabels=False)
            plt.show()
            

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
        """highlight one particular cluster in the plot and show the mean of the cluster"""
        if cluster >= self.k:
            raise ValueError(f'cluster must be less than {self.k}')
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        df = self.sample_df(n_samples)
        sns.scatterplot(x='x0', y='x1', data=df[df['cluster'] == cluster], color='blue')
        sns.scatterplot(x='x0', y='x1', data=df[df['cluster'] != cluster], color='gray')
        sns.scatterplot(x='x0', y='x1', data=df[df['cluster'] == cluster].mean().to_frame().T, color='red', marker='x', s=100)
        plt.show()


def cluster_means(samples_df):
    """
    Compute the means of each cluster
    """
    return samples_df.groupby('cluster').mean()
    

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
    print(df[df['cluster'] == 0].mean().to_frame().T)
    print(df.head())
    print(df.shape)
    gmm.plot_cluster(cluster=1, n_samples=1000)
    print(cluster_means(df))
    
    model_1d = GMM(3, 1, centers=[-2, 0, 1], sigma=0.5)
    model_1d.plot_pdf()
