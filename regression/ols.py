import numpy as np

class LinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y, residual=False):
        """
        Fit the linear regression model using Ordinary Least Squares (OLS) method.
        We assume the linear model
        $$y_i = w_0 + w_1 * x_i1 + w_2 * x_i2 + ... + w_d * x_id + epsilon_i$$
        Computational cost: O(d^2 * n)
        params:
        X: (n, d) array
        y: (n,) array

        """
        # Add a column of ones to X
        self.n_samples_, self.n_features_ = X.shape
        X = np.hstack([np.ones((self.n_samples_, 1)), X]) #(n, d+1)

        # Perform Singular Value Decomposition of X
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        X_pinv = np.dot(Vt.T / S, U.T)

        # Compute the weights
        self.coef_ = np.dot(X_pinv, y)

        # Compute the residuals
        if residual:
            self.residuals_ = np.sum((y - np.dot(X, self.coef_))**2)
    
    def predict(self, X):
        """
        Make predictions using the linear model
        params:
        X: (n, d) array
        return:
        y_pred: (n,) array
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.coef_)

if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import LinearRegression as LinearRegression_sklearn
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    reg_sklearn = LinearRegression_sklearn().fit(X, y)
    print(reg_sklearn.coef_, reg_sklearn.intercept_)

    reg = LinearRegression()
    reg.fit(X, y)
    print(reg.coef_, reg.residuals_)