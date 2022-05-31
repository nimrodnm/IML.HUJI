from __future__ import annotations
from typing import NoReturn, Tuple
from IMLearn.base import BaseEstimator
from IMLearn.learners.regressors import LinearRegression
import numpy as np


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.

        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam
        self.regressor = LinearRegression(include_intercept=False)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

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
        # If an intercept is needed - add a column of ones to the design matrix:
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]
        self.regressor.fit(self.__transform(X), np.concatenate((y, np.zeros(X.shape[1]))))
        self.coefs_ = self.regressor.coefs_

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
        # If an intercept is needed - add a column of ones to the design matrix:
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]
        return self.regressor.predict(X)

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
        # If an intercept is needed - add a column of ones to the design matrix:
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]
        return self.regressor.loss(X, y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the given design matrix X with dimensions m*d, into an extended design matrix with dimensions
        (m+d)*d, where the d extra rows at the bottom are the identity matrix I multiplied by sqrt(self.lam_).

        Parameters:
            X : ndarray of shape (n_samples, n_features)
                The design matrix to be transformed.

        Returns:
            transformed_X: ndarray of shape (n_samples + n_features, n_features)
        """
        regularization_mat = np.identity(X.shape[1]) * np.sqrt(self.lam_)
        # Prevent weighting the intercept term in case it was added to the design matrix:
        if self.include_intercept_:
            regularization_mat[0, 0] = 0
        return np.concatenate((X, regularization_mat), axis=0)
