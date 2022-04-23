from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy
from typing import Tuple
from typing import NoReturn
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(model: Perceptron, x_i: np.ndarray, y_i: int) -> NoReturn:
            losses.append(model.loss(X, y))

        Perceptron(callback=loss_callback).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        go.Figure(
            data=go.Scatter(x=np.arange(1, len(losses) + 1), y=losses, mode="markers + lines"),
            layout=dict(title=f"Loss as a Function of Fitting Iteration - The {n} Case",
                        xaxis_title=r"$\text{Iteration}$",
                        yaxis=dict(title=r"$\text{Loss}$", range=[0, max(losses) + 0.1]))).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for name, file in [("Gaussian-1 Dataset", "gaussian1.npy"),
                       ("Gaussian-2 Dataset", "gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{file}")

        # Fit models and predict over training set
        models = [GaussianNaiveBayes().fit(X, y), LDA().fit(X, y)]
        predictions = [model.predict(X) for model in models]

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        symbols = np.array(["circle", "square", "triangle-up"])
        model_names = ["Gaussian Naive Bayes", "LDA"]
        titles = [rf"$\textbf{{Model: {model_names[i]}.  Accuracy: {accuracy(y, predictions[i])}}}$"
                  for i in range(2)]
        fig = make_subplots(rows=1, cols=2, subplot_titles=titles)

        # Add traces for data-points setting symbols and colors
        for i in range(2):
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=predictions[i],
                                                 symbol=symbols[y],
                                                 line=dict(color="black", width=1),
                                                 colorscale=["red", "blue", "green"])),
                          col=i+1, row=1)
        fig.update_layout(title=rf"$\textbf{{Predictions of LDA and GNB Models on the {name}}}$",
                          margin=dict(t=100))

        # Add `X` dots specifying fitted Gaussians' means
        for i in range(2):
            fig.add_trace(go.Scatter(x=models[i].mu_[:, 0], y=models[i].mu_[:, 1],
                                     mode="markers", showlegend=False,
                                     marker=dict(color="black", symbol="x")),
                          col=i+1, row=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_traces([get_ellipse(models[0].mu_[k, :], np.diag(models[0].vars_[k, :]))
                        for k in range(models[0].classes_.size)])
        fig.add_traces([get_ellipse(models[1].mu_[k, :], models[1].cov_)
                        for k in range(models[1].classes_.size)], rows=1, cols=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
