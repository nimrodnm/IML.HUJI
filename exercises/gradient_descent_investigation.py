import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn.base import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.model_selection import train_test_split
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go
import plotly.io as pio
pio.templates["custom"] = go.layout.Template(layout=go.Layout(margin=dict(l=20, r=20, t=40, b=0)))
pio.templates.default = "simple_white+custom"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path of {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[..., None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[..., None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    weights: List[np.ndarray]
        Recorded parameters

    values: List[np.ndarray]
        Recorded objective values

    deltas: List[float]
        Recorded deltas
    """
    weights_list, values = [], []

    def gd_callback(solver: GradientDescent, weights: np.ndarray, value: np.ndarray,
                    grad: np.ndarray, t: int, eta: float, delta: float):
        weights_list.append(weights.copy())
        values.append(value.copy())

    return gd_callback, weights_list, values


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_fig = go.Figure(layout=dict(title="Convergence Rates of L1 Norm For Different Fixed Step Sizes"))
    l2_fig = go.Figure(layout=dict(title="Convergence Rates of L2 Norm For Different Fixed Step Sizes"))

    for eta in etas:
        l1_callback, l1_descent_path, l1_values = get_gd_state_recorder_callback()
        l2_callback, l2_descent_path, l2_values = get_gd_state_recorder_callback()
        l1_gd = GradientDescent(learning_rate=FixedLR(eta), callback=l1_callback)
        l2_gd = GradientDescent(learning_rate=FixedLR(eta), callback=l2_callback)
        l1_gd.fit(L1(init), X=None, y=None)
        l2_gd.fit(L2(init), X=None, y=None)
        # Question 1:
        plot_descent_path(L1, descent_path=np.array(l1_descent_path), title=f"L1 Norm. Step size = {eta}").show()
        plot_descent_path(L2, descent_path=np.array(l2_descent_path), title=f"L2 Norm. Step size = {eta}").show()
        # Question 3:
        l1_fig.add_trace(go.Scatter(x=np.arange(len(l1_values)), y=np.array(l1_values).flatten(),
                                    mode="lines", name=f"step size={eta}"))
        l2_fig.add_trace(go.Scatter(x=np.arange(len(l2_values)), y=np.array(l2_values).flatten(),
                                    mode="lines", name=f"step size={eta}"))
        # Question 4:
        print(f"Loss achieved when minimizing L1 norm with fixed step size = {eta} was: {l1_values[-1]}")
        print(f"Loss achieved when minimizing L2 norm with fixed step size = {eta} was: {l2_values[-1]}")

    print("\n")
    l1_fig.update_layout(xaxis_title="Iteration", yaxis_title="Norm Value").show()
    l2_fig.update_layout(xaxis_title="Iteration", yaxis_title="Norm Value").show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    fig_1 = go.Figure(layout=dict(title="Convergence Rates of L1 Norm Using Different Decay Rates"))
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    for gamma in gammas:
        gr_callback, descent_path, values = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=gr_callback)
        gd.fit(L1(init), X=None, y=None)
        fig_1.add_trace(go.Scatter(x=np.arange(len(values)), y=np.array(values).flatten(),
                                   mode="lines", name=f"decay rate={gamma}"))
        print(f"Loss achieved when minimizing L1 norm with exp decay value = {gamma} was: {values[-1]}")
    # Plot algorithm's convergence for the different values of gamma
    fig_1.update_layout(xaxis_title="Iteration", yaxis_title="Norm Value").show()

    # Plot descent path for gamma=0.95
    l1_callback, l1_descent_path, _ = get_gd_state_recorder_callback()
    l2_callback, l2_descent_path, _ = get_gd_state_recorder_callback()
    l1_gd = GradientDescent(learning_rate=ExponentialLR(eta, 0.95), callback=l1_callback)
    l2_gd = GradientDescent(learning_rate=ExponentialLR(eta, 0.95), callback=l2_callback)
    l1_gd.fit(L1(init), X=None, y=None)
    l2_gd.fit(L2(init), X=None, y=None)

    plot_descent_path(L1, descent_path=np.array(l1_descent_path), title="L1 Norm. Decay Rate = 0.95").show()
    plot_descent_path(L2, descent_path=np.array(l2_descent_path), title="L2 Norm. Decay Rate = 0.95").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    # return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)
    train_X, test_X, train_y, test_y = train_test_split(df.drop(['chd', 'row.names'], axis=1), df.chd, train_size=train_portion)
    return train_X, train_y, test_X, test_y


def get_roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> go.Figure:
    """
    Return plot of ROC curve of the given fpr (false positive rate), tpr (true positive rate) and thresholds.
    """
    return go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash="dash"),
                                 showlegend=False),
                      go.Scatter(x=fpr, y=tpr, mode="markers + lines", showlegend=False, text=thresholds,
                                 hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
                     layout=dict(title=f"Roc Curve of Logistic Regression Model With No Regularization<br>"
                                       f"AUC = {auc(fpr, tpr):.3f}",
                                 xaxis_title="False Positive Rate (FPR)", yaxis_title="True Positive Rate (TPR)"))


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    callback, descent_path, values = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback, learning_rate=FixedLR(1e-4), max_iter=20000, out_type="last")
    lr = LogisticRegression(solver=gd).fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_train, lr.predict_proba(X_train))
    # Question 8 - Plot the ROC curve:
    get_roc_curve_plot(fpr, tpr, thresholds).show()
    # Plotting the convergence:
    go.Figure(go.Scatter(x=np.arange(len(values)), y=np.array(values).flatten(), mode="lines", name=f"step size")).show()

    # Question 9 - find optimal alpha:
    optimal_alpha_idx = np.argmax(tpr - fpr)
    optimal_alpha = thresholds[optimal_alpha_idx]
    print("\nQuestion 9:")
    print(f"The alpha value that achieved the optimal ROC value is: alpha={optimal_alpha}")
    lr = LogisticRegression(solver=gd, alpha=optimal_alpha).fit(X_train, y_train)
    print(f"The test misclassification error using the optimal alpha is: {lr.loss(X_test, y_test)}")
    print(f"Value of TPR-FPR using the optimal alpha is: {tpr[optimal_alpha_idx] - fpr[optimal_alpha_idx]}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    print("\nQuestions 10 and 11:")
    for norm in ["l1", "l2"]:
        train_errors, validation_errors = [], []
        for lam in lambdas:
            print(f"Running Cross-Validation on {norm.capitalize()}, lambda={lam}")
            result = cross_validate(LogisticRegression(solver=gd, penalty=norm, lam=lam), X_train, y_train,
                                    misclassification_error, cv=5)
            train_errors.append(result[0])
            validation_errors.append(result[1])

        go.Figure([go.Scatter(x=lambdas, y=train_errors, mode="lines", name="Train Errors"),
                   go.Scatter(x=lambdas, y=validation_errors, mode="lines", name="Validation Errors")],
                  layout=dict(title=f"Errors of Cross-Validation using {norm.capitalize()} Regularization",
                              xaxis_title="Lambda", yaxis_title="Error")).show()

        best_lam_idx = int(np.argmin(validation_errors))
        best_lam = lambdas[best_lam_idx]
        model = LogisticRegression(solver=gd, penalty=norm, lam=best_lam).fit(X_train, y_train)
        print("-"*30)
        print(f"Best lambda value was: {best_lam}")
        print(f"Error of LogisticRegression with {norm.capitalize()} Regularization on and optimal lambda on the"
              f" test set is: {model.loss(X_test, y_test)}")
        print("-" * 30)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
