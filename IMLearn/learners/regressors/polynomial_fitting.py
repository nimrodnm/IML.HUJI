from __future__ import annotations
from typing import NoReturn
from IMLearn.learners.regressors import LinearRegression
from IMLearn.base import BaseEstimator
import numpy as np


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """

    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self._deg = k
        self._regressor = LinearRegression(include_intercept=False)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # TODO: should accept X with shape=(n,1) or shape=(n,) ? For now accepting the 2'nd option
        self._regressor.fit(self.__transform(X), y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self._regressor.predict(self.__transform(X))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return self._regressor.loss(self.__transform(X), y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        return np.vander(X, N=(self._deg + 1), increasing=True)


if __name__ == '__main__':
    response = lambda x: x ** 4 - 2 * x ** 3 - .5 * x ** 2 + 1
    # Take only even indices:
    x = np.linspace(-1.2, 2, 40)[0::2]
    y_ = response(x)
    from sklearn.preprocessing import PolynomialFeatures
    import sklearn.linear_model as lm
    from sklearn.pipeline import make_pipeline

    m, k, X = 5, 4, x.reshape(-1, 1)
    pol = PolynomialFitting(k)

    sklearn_pred = make_pipeline(PolynomialFeatures(k),
                                 lm.LinearRegression(fit_intercept=False)).fit(X, y_).predict(X)
    pol.fit(x, y_)
    my_pred = pol.predict(x)
    print(sklearn_pred)
    print("\n")
    print(my_pred)
