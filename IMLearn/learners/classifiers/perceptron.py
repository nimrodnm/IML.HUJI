from __future__ import annotations
from typing import Callable
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from IMLearn.metrics import misclassification_error, accuracy


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data.
            Callable function should receive as input a Perceptron instance, current sample and
            current response.
    """

    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data.
            Callable function should receive as input a Perceptron instance, current sample and
            current response.
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to the given samples. Iterate over given data as long as there exists a
        sample misclassified, and did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        X = self.__adjust_intercept(X)
        self.__partial_fit(X, y, first_fit=True)
        t = 1

        while t < self.max_iter_:
            if self.__partial_fit(X, y, first_fit=False):
                break
            t += 1

    def __partial_fit(self, X: np.ndarray, y: np.ndarray, first_fit: bool = False) -> bool:
        """
        Runs one iteration of the Perceptron algorithm:
        loops over the rows of the design matrix X, and if it finds a miss-classification (which
        is indicated by y[i] * inner_product(self.coefs ,X[i,:]) <=0) it adjusts self.coefs_ to
        give a "better" separating hyperplane.
        Runs self.callback_ at the end of the loop.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_features + 1)
            Input data to do partial fit on.
            Note: Assumes X includes an intercept column in the case include_intercept_ == True !!

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to.

        first_fit : if True, initialize self.coefs_ as the 0 vector.
                    if False, assume self.coefs_ is already initialized (not None)

        Returns
        -------
        True: the model is completely fitted (meaning y[i] * inner_product(self.coefs ,X[i,:]) > 0
        for all i=0,...,X.shape[1])
        False: model is partially fitted.
        """
        if first_fit:
            self.coefs_ = np.zeros(X.shape[1])
            self.fitted_ = True

        for i in range(X.shape[0]):
            if y[i] * np.inner(self.coefs_, X[i, :]) <= 0:
                self.coefs_ = self.coefs_ + y[i] * X[i, :]
                self.callback_(self, X[i, :], y[i])
                return False

        return True

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
        X = self.__adjust_intercept(X)

        y = np.sign(X @ self.coefs_)
        return np.where(y == 0, -1, y)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))

    def __adjust_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Adds a column of 1's as the 0'th column of X, to allow calculating the coefficients
        including an intercept.

        Parameters
        ----------
            X : ndarray of shape (n_samples, n_features)
            Test samples

        Returns
        ----------
            X with an extra column on 1's: a ndarray of shape (n_samples, n_features + 1)
        """
        return np.c_[np.ones(X.shape[0]), X] if self.include_intercept_ else X
