import numpy as np
from IMLearn.base import BaseEstimator
from IMLearn.metrics import misclassification_error
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Set initial distribution to be uniform over all samples:
        self.D_ = np.full(shape=y.shape, fill_value=(1 / y.size))

        # Initialize ndarrays:
        self.models_, self.weights_ = [], []

        # Create and fit weak learners:
        for t in range(self.iterations_):
            # Fitting a weak learner:
            print(f"adaboost fit iteration number {t+1}")
            self.models_.append(self.wl_().fit(X, self.D_ * y))
            y_pred = self.models_[t].predict(X)
            # Computing the weighted misclassification error of the weak learner:
            error = (self.D_ * (y_pred != y).astype(int)).sum()
            # Setting the weight of the weak learner:
            self.weights_.append(0.5 * np.log((1 / error) - 1))
            # Updating and normalizing the distribution:
            self.D_ = self.D_ * np.exp(-self.weights_[t] * y * y_pred)
            self.D_ /= self.D_.sum()

    def _predict(self, X):
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
        return self.partial_predict(X, self.iterations_)

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if not self.fitted_:
            raise ValueError("AdaBoost must first be fitted before calling ``partial_predict``")

        pred = np.zeros(X.shape[0])
        for t in range(T):
            print(f"adaboost predict iteration number {t + 1}")
            pred += self.weights_[t] * self.models_[t].predict(X)
        return np.sign(pred)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        if not self.fitted_:
            raise ValueError("AdaBoost must first be fitted before calling ``partial_loss``")

        return misclassification_error(y, self.partial_predict(X, T))