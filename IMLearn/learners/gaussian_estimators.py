from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet, multi_dot


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # update the expectation (mu) to be the sample mean estimator:
        self.mu_ = X.mean()

        # update the variance (var) to be the sample variance estimator (biased/unbiased):
        m = X.size
        var_factor = (1 / m) if self.biased_ else (1 / (m - 1))
        self.var_ = var_factor * ((X - self.mu_) ** 2).sum()

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        constant = 1 / np.sqrt(2 * np.pi * self.var_)
        normalized_samples = ((X - self.mu_) ** 2) / (-2 * self.var_)
        return constant * np.exp(normalized_samples)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # Calculate the log_likelihood according to the formula we've learned:
        m = X.size
        first_factor = - 1 / (2 * sigma)
        second_factor = (m / 2) * np.log(2 * np.pi * sigma)
        return first_factor * ((X - mu) ** 2).sum() - second_factor


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # update the expectation (mu) to be the sample mean estimator:
        self.mu_ = X.mean(axis=0)

        # update the covariance (cov) to be the sample variance estimator:
        m = X.shape[0]  # the number of samples (i.e. random vectors) in X
        centered_x = X - self.mu_
        self.cov_ = (1 / (m - 1)) * np.dot(centered_x.transpose(), centered_x)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        m, d = X.shape  # d is the dimension of each sample and m is the number of samples
        constant = 1 / np.sqrt(np.power(2 * np.pi, d) * det(self.cov_))
        centered_x = X - self.mu_
        pdf = np.zeros(m)
        for i in range(m):
            row_mult = multi_dot((centered_x[i].T, inv(self.cov_), centered_x[i]))
            pdf[i] = constant * np.exp(-0.5 * row_mult)
        return pdf

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        m, d = X.shape  # d is the dimension of each sample and m is the number of samples
        centered_x = X - mu
        # calculating the "complicated" part of the formula which is:
        # sum((Xi-mu).T * inv(cov) * (Xi-mu))
        # as: trace((X-mu).T * inv(cov) * (X-mu)) which can be computed faster.
        row_prod_sum = (np.dot(centered_x, inv(cov)) * centered_x).sum()  # calculation of the trace
        det_sign, log_det = slogdet(cov)
        const = m * ((d * np.log(2 * np.pi)) + (det_sign * log_det))
        return (-0.5) * (row_prod_sum + const)

        # first version:
        # m, d = X.shape  # d is the dimension of each sample and m is the number of samples
        # centered_x = X - mu
        # row_prod_sum = np.apply_along_axis(lambda x: multi_dot((x.T, inv(cov), x)), 1,
        #                                    centered_x).sum() * (-0.5)
        # return row_prod_sum - (m / 2) * np.log(np.power((2 * np.pi), d) * det(cov))

        # second version:
        # row_prod_sum = (np.dot(centered_x, inv(cov)) * centered_x).sum()
        # return (-0.5) * (row_prod_sum + m * np.log(np.power((2 * np.pi), d) * det(cov))) * 7
