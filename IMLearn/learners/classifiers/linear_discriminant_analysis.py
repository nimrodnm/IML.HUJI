from typing import NoReturn
from IMLearn.base import BaseEstimator
from IMLearn.metrics import misclassification_error
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        # self.pi_ = {self.classes_[i]: (counts[i] / m) for i in range(self.classes_.size)}
        self.pi_ = np.array([counts[i] / m for i in range(self.classes_.size)])

        # Calculate expectation estimation for every class:
        # self.mu_ = {clas: X[y == clas, :].mean(axis=0) for clas in self.classes_}
        self.mu_ = np.empty((self.classes_.size, d))
        for i in range(self.classes_.size):
            self.mu_[i, :] = X[y == self.classes_[i], :].mean(axis=0)

        # Calculate covariance matrix estimation:
        self.cov_ = np.zeros((d, d))
        for i in range(self.classes_.size):
            centered = X[y == self.classes_[i], :] - self.mu_[i]
            self.cov_ += centered.T @ centered
        self.cov_ /= m - self.classes_.size
        self._cov_inv = inv(self.cov_)

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
        return np.apply_along_axis(self.__predict_row, axis=1, arr=X)

    def __predict_row(self, x: np.ndarray) -> float:
        """
        Predict response for the given row according to the following formula:
        pred(x) = argmax(a_k.T @ x + b_k) for every k from 0 to (self.classes_.size - 1)
        where: a_k = self._cov_inv @ self.mu_[k]
        b_k = log(self.classes_[k]) - 0.5 * np.inner(self._cov_inv.T @ self.mu_[k], self.mu_[k])

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            A single sample (single row of X)

        Returns
        -------
        The prediction for the given sample.
        """
        results = np.array([(self.__a(i).T @ x) + self.__b(i) for i in range(self.classes_.size)])
        return self.classes_[np.argmax(results)]

    def __a(self, i: int) -> np.ndarray:
        """
        Parameters:
            i : index of class of self.classes_
        """
        return self._cov_inv @ self.mu_[i]

    def __b(self, i: int) -> np.ndarray:
        """
        Parameters:
            i : index of class of self.classes_
        """
        return np.log(self.pi_[i]) - 0.5 * np.inner(self._cov_inv.T @ self.mu_[i], self.mu_[i])

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
        const = np.sqrt(((2 * np.pi) ** d) * det(self.cov_))

        likelihoods = np.empty((m, self.classes_.size))
        for i in range(self.classes_.size):
            centered = X - self.mu_[i]
            mahalanobis = np.einsum("bi,ij,bj->b", centered, self._cov_inv, centered)
            likelihoods[:, i] = self.pi_[i] * np.exp(-0.5 * mahalanobis) / const

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
    lda = LDA()
    lda.fit(X, y)
    # print(lda.cov_)
    # print(lda.mu_)
    print(lda.predict(X))
    # print(lda.likelihood(X))
