from typing import NoReturn
from IMLearn.base import BaseEstimator
from IMLearn.metrics import misclassification_error
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, d = X.shape  # m is the number of samples, d is the number of features

        # Get the classes and the amount of samples from each class:
        self.classes_, counts = np.unique(y, return_counts=True)

        # Calculate class probabilities:
        self.pi_ = np.array([counts[i] / m for i in range(self.classes_.size)])

        # Calculate estimated features means for each class:
        self.mu_ = np.empty((self.classes_.size, d))
        for i in range(self.classes_.size):
            self.mu_[i, :] = X[y == self.classes_[i], :].mean(axis=0)

        # Calculate estimated features variances for each class:
        self.vars_ = np.empty((self.classes_.size, d))
        for i in range(self.classes_.size):
            centered = X[y == self.classes_[i], :] - self.mu_[i]
            self.vars_[i, :] = (1 / counts[i]) * (centered ** 2).sum(axis=0)

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
        likelihoods = self.likelihood(X)
        return np.apply_along_axis(lambda x: self.classes_[np.argmax(x)], axis=1, arr=likelihoods)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m, d = X.shape  # m is the number of samples, d is the number of features

        likelihoods = np.empty((m, self.classes_.size))
        for i in range(self.classes_.size):
            centered = - ((X - self.mu_[i, :]) ** 2) / (2 * self.vars_[i, :])
            denominator = np.sqrt(2 * np.pi * self.vars_[i, :])
            likelihoods[:, i] = self.pi_[i] * np.prod((1 / denominator) * np.exp(centered), axis=1)

        return likelihoods

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


if __name__ == '__main__':
    data = np.load("G:/My Drive/Semester_4/IML/IML.HUJI/datasets/gaussian1.npy")
    X = data[:, :2]
    y = data[:, 2]
    GNB = GaussianNaiveBayes()
    GNB.fit(X, y)
    print(GNB.predict(X))
    # print(GNB.likelihood(X))
