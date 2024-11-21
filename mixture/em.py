import numpy as np
from .mm import GMM

class EM():
    def __init__(self, k, eps=1e-5, max_iter=5000):
        self.k = k
        self.eps = eps
        self.max_iter = max_iter

    def estimate(self, samples, centers=None, sigma=None, likelihood=False, fixed_sigma=False):
        
        num = samples.shape[0]

        if sigma is None:
            sigma = np.cov(samples.T)
        if centers is None:
            centers = samples[np.random.choice(num, size=self.k, replace=False)]

        model = GMM(self.k, samples.shape[1], centers=centers, sigma=sigma)

        l_mat = np.exp(ll_mat_hd(samples, model)) # shape (n,k)
        ll_cur = np.sum(np.log(np.dot(l_mat, model.weights)))

        num_iter = 0

        while True:
            ll_pre = ll_cur
            labels = l_mat * model.weights # shape (n,k)
            labels /= np.sum(labels, axis=1)[:,np.newaxis] # shape (n,k)

            sum_labels = np.sum(labels, axis=0) # shape (k,)

            model.weights = sum_labels / num
            model.centers = np.dot(labels.T, samples) / sum_labels[:,np.newaxis]

            if not fixed_sigma:
                cross = (samples[:,:,None]*samples[:,None])[:,None] \
                        -2*samples[:,None,:,None]*model.centers[:,None] \
                        +model.centers[:,:,None] * model.centers[:,None] # shape (n, k, d, d)
                
                model.sigma = np.sum(labels[:,:,None,None] * cross, axis=(0,1)) / num

            l_mat = np.exp(ll_mat_hd(samples, model))
            ll_cur = np.sum(np.log(np.dot(l_mat, model.weights)))

            num_iter += 1

            if ll_cur - ll_pre < self.eps or num_iter==self.max_iter:
                break
        print("Iteration Time: ", num_iter)
        print("Likelihood: ", ll_cur)

        if likelihood:
            return model, ll_cur
        else:
            return model
        
    def estimate_fixed_sigma(self, samples, centers=None, sigma=None, likelihood=False):
        
        num = samples.shape[0]

        if sigma is None:
            sigma = np.cov(samples.T)
        if centers is None:
            centers = samples[np.random.choice(num, size=self.k, replace=False)]

        model = GMM(self.k, samples.shape[1], centers=centers, sigma=sigma)

        l_mat = np.exp(ll_mat_hd(samples, model))

        ll_cur = np.sum(np.log(np.dot(l_mat, model.weights)))

        num_iter = 0

        while True:
            ll_pre = ll_cur
            labels = l_mat * model.weights # shape (n,k)
            labels /= np.sum(labels, axis=1)[:,np.newaxis] # shape (n,k)

            sum_labels = np.sum(labels, axis=0) # shape (k,)

            model.weights = sum_labels / num
            model.centers = np.dot(labels.T, samples) / sum_labels[:,np.newaxis]

            l_mat = np.exp(ll_mat_hd(samples, model))
            ll_cur = np.sum(np.log(np.dot(l_mat, model.weights)))

            num_iter += 1

            if ll_cur - ll_pre < self.eps or num_iter==self.max_iter:
                break
        print("Iteration Time: ", num_iter)
        print("Likelihood: ", ll_cur)

        if likelihood:
            return model, ll_cur
        else:
            return model
        
def ll_mat_hd(samples, model):
    samples = np.asarray(samples)
    precision = np.linalg.inv(model.sigma)

    ll0 = ((samples@precision*samples).sum(axis=1))[:,np.newaxis] \
          -2*np.inner(samples@precision, model.centers) \
          +(model.centers@precision*model.centers).sum(axis=1) # shape:(n,k)
    
    return -0.5*(model.ndim*np.log(2*np.pi) + ll0 + np.log(np.linalg.det(model.sigma)))