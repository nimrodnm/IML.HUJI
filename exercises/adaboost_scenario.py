from utils import *
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    booster = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    n_learners_range = list(range(1, n_learners + 1))
    train_errors = [booster.partial_loss(train_X, train_y, T) for T in n_learners_range]
    test_errors = [booster.partial_loss(test_X, test_y, T) for T in n_learners_range]
    fig_1 = go.Figure([go.Scatter(x=n_learners_range, y=train_errors, mode='lines', name="Train errors"),
                       go.Scatter(x=n_learners_range, y=test_errors, mode='lines', name="Test errors")])
    fig_1.update_layout(title=f"Q1 - noise={noise}: "
                              f"Train and Test Errors as a Function of the Number of Fitted Learners",
                        xaxis_title="Number of Fitted Learners")
    fig_1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig_2 = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} Weak Learners" for t in T],
                          horizontal_spacing=0.01, vertical_spacing=0.05)
    for i, t in enumerate(T):
        fig_2.add_traces([decision_surface(lambda X: booster.partial_predict(X, t),
                                           lims[0], lims[1], showscale=False),
                          go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=test_y,
                                                 colorscale=[custom[0], custom[-1]],
                                                 line=dict(width=0.5, color="DarkSlateGrey"))
                                     ),
                          ],
                         rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig_2.update_layout(title=f"Q2 - noise={noise}: "
                              f"Decision Boundary Obtained by Using the Ensemble Up to Iteration 5, 50, 100, 250",
                        margin_t=100)
    fig_2.update_xaxes(visible=False).update_yaxes(visible=False)
    fig_2.show()

    # Question 3: Decision surface of best performing ensemble
    lowest_err = min(test_errors)
    best_ensm_size = 1 + test_errors.index(lowest_err)
    fig_3 = go.Figure([decision_surface(lambda X: booster.partial_predict(X, best_ensm_size),
                                        lims[0], lims[1], showscale=False),
                       go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=test_y,
                                              colorscale=[custom[0], custom[-1]],
                                              line=dict(width=1, color="DarkSlateGrey"))
                                  ),
                       ])
    fig_3.update_layout(title=f"Q3 - noise={noise}: Decision Boundary of the Best Performing Ensemble, With Size "
                              f"= {best_ensm_size} and Accuracy = {1 - lowest_err}")
    fig_3.update_xaxes(visible=False).update_yaxes(visible=False)
    fig_3.show()

    # Question 4: Decision surface with weighted samples
    max_marker_size = 50 if noise == 0 else 10
    size_ref = 2 * np.max(booster.D_) / (max_marker_size ** 2)
    fig_4 = go.Figure([decision_surface(booster.predict, lims[0], lims[1], showscale=False),
                       go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=train_y,
                                              colorscale=[custom[0], custom[-1]],
                                              size=booster.D_,
                                              sizemode="area",
                                              sizeref=size_ref,
                                              sizemin=0.5,
                                              line=dict(width=1, color="DarkSlateGrey"))
                                  ),
                       ])
    fig_4.update_layout(title=f"Q4 - noise={noise}: "
                              f"Decision Boundary of full Ensemble With Point Size Proportional to its Weight")
    fig_4.update_xaxes(visible=False).update_yaxes(visible=False)
    fig_4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
