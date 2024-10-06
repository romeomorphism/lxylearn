import numpy as np


class LogisticRegression:
    def __init__(self) -> None:
        pass

    def fit(self, X, y, alpha=1, max_iter=10000):
        """
        Learn the model parameters from the training data.
        params:
        X: (n, d) array
        y: (n,) array
        alpha: float, learning rate
        max_iter: int, maximum number of iterations
        """
        
        X = np.hstack([np.ones((X.shape[0], 1)), X]) #(n, d+1)
        self.n_samples_, self.n_features_ = X.shape

        self.coef_ = np.zeros(self.n_features_)

        for _ in range(max_iter):
            grad = self._compute_gradient(X, y)
            self.coef_ -= alpha * grad

    def fit_newton(self, X, y, max_iter=100, eps=1e-5):
        """
        Learn the model parameters from the training data using Newton's method.
        params:
        X: (n, d) array
        y: (n,) array
        max_iter: int, maximum number of iterations
        """
        
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.n_samples_, self.n_features_ = X.shape

        self.coef_ = np.random.random(self.n_features_)

        for _ in range(max_iter):
            grad = self._compute_gradient(X, y)
            hess = self._compute_hessian(X)
            try:
                self.coef_ -= np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                print("Error: Singular matrix encountered. Exiting loop.")
                break
            if np.linalg.norm(self._compute_gradient(X, y)) < eps:
                break
                
    
    def _compute_hessian(self, X):
        """
        Compute the Hessian of the loss function with respect to the model parameters.
        params:
        X: (n, d) array
        return:
        hess: (d, d) array
        """
        z = np.dot(X, self.coef_)
        h = 1 / (1 + np.exp(-z))
        W = np.diag(h * (1 - h))
        hess = np.dot(X.T, np.dot(W, X))
        return hess
    
    def _compute_gradient(self, X, y):
        """
        Compute the gradient of the loss function with respect to the model parameters.
        params:
        X: (n, d) array
        y: (n,) array
        retu rn:
        grad: (d,) array
        """
        z = np.dot(X, self.coef_)
        h = 1 / (1 + np.exp(-z))
        grad = np.dot(X.T, h - y)
        return grad
    
    def predict_prob(self, X):
        """
        Make predictions using the learned model.
        params:
        X: (n, d) array
        return:
        y_pred: (n,) array
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        z = np.dot(X, self.coef_)
        h_pred = 1 / (1 + np.exp(-z))
        return h_pred
    
    def predict(self, X):
        """
        Make predictions using the learned model.
        params:
        X: (n, d) array
        return:
        y_pred: (n,) array
        """
        return (self.predict_prob(X) > 0.5).astype(int)
    
if __name__ == '__main__':
    import numpy as np
    from sklearn.linear_model import LogisticRegression as LogisticRegression_sklearn
    X = np.arange(1, 6).reshape(-1, 1)
    y = np.array([0] * 2 + [1] * 3)
    reg_sklearn = LogisticRegression_sklearn(solver='newton-cg', penalty=None).fit(X, y)
    print(reg_sklearn.intercept_, reg_sklearn.coef_)

    reg_sklearn = LogisticRegression_sklearn(solver='lbfgs', penalty=None).fit(X, y)
    print(reg_sklearn.intercept_, reg_sklearn.coef_)
    print(reg_sklearn.predict(X))

    reg = LogisticRegression()
    reg.fit_newton(X, y, 100, 1e-3)
    print(reg.coef_)

    reg = LogisticRegression()
    reg.fit(X, y)
    print(reg.coef_)



