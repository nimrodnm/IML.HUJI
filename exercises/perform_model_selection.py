from __future__ import annotations
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from utils import *
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    poly = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    x = np.random.uniform(-1.2, 2, n_samples)
    x_sorted = np.linspace(-1.2, 2, n_samples)
    epsilon = np.random.normal(0, np.sqrt(noise), n_samples)
    y = poly(x) + epsilon
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), train_proportion=(2 / 3))
    train_x, test_x = train_x.squeeze(), test_x.squeeze()
    fig_1 = go.Figure([go.Scatter(x=x_sorted, y=poly(x_sorted), mode="lines",
                                  name="Full Noiseless Data", line_color="black"),
                       go.Scatter(x=train_x, y=train_y, mode="markers", name="Train Data", marker_color="blue"),
                       go.Scatter(x=test_x, y=test_y, mode="markers", name="Test Data", marker_color="red")])
    fig_1.update_layout(title=f"Scatter Plot of Noiseless Model and of Test and Train Data<br>"
                              f"Noise = {noise}, Number of Samples = {n_samples}",
                        xaxis_title="x", yaxis_title="y")
    fig_1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.arange(11)
    train_errors, validation_errors = [], []
    for k in degrees:
        result = cross_validate(PolynomialFitting(k), train_x.to_numpy(),
                                train_y.to_numpy(), mean_square_error, cv=5)
        train_errors.append(result[0])
        validation_errors.append(result[1])
    fig_2 = go.Figure([go.Scatter(x=degrees, y=train_errors, mode="markers + lines", name="Training Errors"),
                       go.Scatter(x=degrees, y=validation_errors, mode="markers + lines", name="Validation Errors")])
    fig_2.update_layout(title=f"Train and Validation Errors of 5-Fold Cross-Validation on Degrees k=0,1,...,10<br>"
                              f"Noise = {noise}, Number of Samples = {n_samples}",
                        xaxis_title="k", yaxis_title="Error Value")
    fig_2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = int(np.argmin(validation_errors))
    polyfit = PolynomialFitting(k_star).fit(train_x.to_numpy(), train_y.to_numpy())
    print(f"Noise = {noise}, Number of Samples = {n_samples}:")
    print(f"The Value of k_star is {k_star}, "
          f"and the test error is {polyfit.loss(test_x.to_numpy(), test_y.to_numpy())}\n")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    dataset = datasets.load_diabetes()
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(dataset.data), pd.Series(dataset.target),
                                                        train_proportion=(n_samples / dataset.data.shape[0]))
    train_X, test_X = train_X.squeeze().to_numpy(), test_X.squeeze().to_numpy()
    train_y, test_y = train_y.to_numpy(), test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas_range_start, lambdas_range_end = 0.0001, 0.2
    lambdas = np.linspace(lambdas_range_start, lambdas_range_end, n_evaluations)
    ridge_train_errors, lasso_train_errors, ridge_validation_errors, lasso_validation_errors = [], [], [], []
    for lam in lambdas:
        ridge_result = cross_validate(RidgeRegression(lam), train_X, train_y, mean_square_error, cv=5)
        lasso_result = cross_validate(Lasso(alpha=lam), train_X, train_y, mean_square_error, cv=5)
        ridge_train_errors.append(ridge_result[0])
        ridge_validation_errors.append(ridge_result[1])
        lasso_train_errors.append(lasso_result[0])
        lasso_validation_errors.append(lasso_result[1])

    ridge_fig = go.Figure([go.Scatter(x=lambdas, y=ridge_train_errors,
                                      mode="lines", name="Ridge Training Errors"),
                           go.Scatter(x=lambdas, y=ridge_validation_errors,
                                      mode="lines", name="Ridge Validation Errors")])
    ridge_fig.update_layout(title=f"Errors of 5-Fold Cross-Validation on Ridge Regularisation with lambdas "
                                  f"in range [{lambdas_range_start}, {lambdas_range_end}]<br>"
                                  f"Number of Samples in Train Set = {n_samples}",
                            xaxis_title="lambda", yaxis_title="Error Value")
    lasso_fig = go.Figure([go.Scatter(x=lambdas, y=lasso_train_errors,
                                      mode="lines", name="Lasso Training Errors"),
                           go.Scatter(x=lambdas, y=lasso_validation_errors,
                                      mode="lines", name="Lasso Validation Errors")])
    lasso_fig.update_layout(title=f"Errors of 5-Fold Cross-Validation on Lasso Regularisation with lambdas "
                                  f"in range [{lambdas_range_start}, {lambdas_range_end}]<br>"
                                  f"Number of Samples in Train Set = {n_samples}",
                            xaxis_title="lambda", yaxis_title="Error Value")
    ridge_fig.show()
    lasso_fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=100, noise=5)
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter(n_samples=50, n_evaluations=500)
