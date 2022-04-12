from typing import NoReturn
from IMLearn.base import BaseEstimator
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

        # Get the classes and the amount of y values for each class:
        self.classes_, counts = np.unique(y, return_counts=True)

        # Calculate class probabilities:
        self.pi_ = np.array([counts[i] / m for i in range(self.classes_.size)])

        # Calculate expectation estimator for every class:
        mus = dict()
        for clas in self.classes_:
            # estimator is the mean of the rows of X such that the y entry is from the class 'clas'
            mus[clas] = X[y == clas, :].mean(axis=0)  # TODO: validate
        self.mu_ = np.array(list(mus.values()))

        # Calculate covariance matrix estimator:
        self.cov_ = np.zeros((d, d))
        for i in range(m):
            centered_row = X[i, :] - mus[y[i]]
            self.cov_ += np.outer(centered_row, centered_row)
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
        # TODO: important!:
        #       Im assuming here that the values in self.classes_ and in self.mu_ share indices -
        #       meaning that the mu in self.mu_[i] is the correct mu for the class in
        #       self.classes_[i].
        #       It depends on the implementation of _fit (mainly depends on python dict to keep the
        #       order of the items).
        #       It might be better to hold a dict attribute of the class that will map each class
        #       value (aka 'k') in self.classes_ to its corresponding mu (aka 'mu_k') in self.mu_.
        #       This is also true regarding self.pi_ (the class probabilities).
        max_k = (self.__a(0).T @ x) + self.__b(0)
        argmax = self.classes_[0]
        for i in range(1, self.classes_.size):
            cur_k = (self.__a(i).T @ x) + self.__b(i)
            if cur_k > max_k:
                max_k = cur_k
                argmax = self.classes_[i]
        return argmax

    def __a(self, i: int) -> np.ndarray:
        """
        TODO: add doc
        """
        return self._cov_inv @ self.mu_[i]

    def __b(self, i: int) -> np.ndarray:
        """
        TODO: add doc
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

        raise NotImplementedError()

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
        from ...metrics import misclassification_error
        raise NotImplementedError()


if __name__ == '__main__':
    X = np.array([[1, 3], [2, 5], [4, 4]])
    y = np.array([1, 0, 1])
    lda = LDA()
    lda.fit(X, y)
    print(lda.predict(X))
