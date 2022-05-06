from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
from IMLearn.metrics import misclassification_error
import numpy as np
from itertools import product

EPSILON = 1 / 100
POSITIVE = 1


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_err = np.inf
        for j in range(X.shape[1]):
            values = X[:, j]
            for sign in [POSITIVE, -POSITIVE]:
                thresh, err = self._find_threshold(values, y, sign)
                if err < min_err:
                    self.threshold_, self.j_, self.sign_, min_err = thresh, j, sign, err

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # Sort values and labels simultaneously according to values:
        s_idx = values.argsort()
        s_values, s_labels = values[s_idx], labels[s_idx]

        # Check first possible prediction - sign for every sample:
        error = np.abs(s_labels[np.sign(s_labels) != sign]).sum()
        # Set the initial threshold and minimal error:
        best_thr, min_err = s_values[0], error
        # Iterate through the values and check if changing the prediction of the latest value (which is the same as
        # "moving" the threshold to the next value) results by a mistake. If so - update the error accordingly by
        # subtracting (-sign)*s_labels[i - 1].
        for i in range(1, s_values.size):  # start iterating from the second value
            error -= (-sign) * s_labels[i - 1]
            # Update min_err only if the current value is a legal threshold (different from the previous value):
            if error < min_err and s_values[i - 1] != s_values[i]:
                best_thr, min_err = s_values[i], error

        # Check last possible prediction, -sign for every sample:
        if np.abs(s_labels[np.sign(s_labels) != -sign]).sum() < min_err:
            best_thr, min_err = (s_values[-1] + abs(EPSILON * s_values[-1])), error

        return best_thr, min_err

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
            Performance under misclassification loss function
        """
        return misclassification_error(y, self._predict(X))
