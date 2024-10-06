from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import pandas as pd


def model_selection_gmm_aic(X, max_components=5, covariance_type=["tied",], tol=[1e-5,], max_iter=[5000,], detailed=False):
    """Perform model selection for a Gaussian Mixture Model using the AIC score."""

    def gmm_aic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the AIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.aic(X)

    param_grid = {
        "n_components": range(1, max_components + 1),
        "covariance_type": covariance_type,
        "tol": tol,
        "max_iter": max_iter,
    }
    grid_search_aic = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_aic_score
    )

    grid_search_aic.fit(X)

    df = pd.DataFrame(grid_search_aic.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "AIC score",
        }
    )
    if detailed:
        df.sort_values(by="AIC score").head()
    
    return df.sort_values(by="AIC score")['Number of components'].values[0]

