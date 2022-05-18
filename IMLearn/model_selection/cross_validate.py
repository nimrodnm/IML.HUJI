from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # Split the data into k=cv manifolds:
    X_indices = np.arange(X.shape[0])
    # X_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    manifolds = np.array_split(X_indices, cv)

    # Train over every manifold and sum the errors:
    train_score, validation_score = 0, 0
    for i in range(cv):
        new_estimator = deepcopy(estimator)  # Deep-copy the estimator to prevent the previous fit from effecting
        train_indices = np.setdiff1d(X_indices, manifolds[i])
        new_estimator.fit(X[train_indices], y[train_indices])
        train_pred, validation_pred = new_estimator.predict(X[train_indices]), new_estimator.predict(X[manifolds[i]])
        train_score += scoring(y[train_indices], train_pred)
        validation_score += scoring(y[manifolds[i]], validation_pred)

    # Return normalized scores:
    return train_score / cv, validation_score / cv
