from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    expectation, variance, sample_size = 10, 1, 1000
    samples = np.random.normal(expectation, variance, sample_size)
    univariate_1 = UnivariateGaussian()
    univariate_1.fit(samples)
    print("Q1) Estimated expectation and variance of univariate gaussian:")
    print(f"(expectation, variance) = ({univariate_1.mu_}, {univariate_1.var_})")
    print("\n")

    # Question 2 - Empirically showing sample mean is consistent
    univariate_2 = UnivariateGaussian()
    expectation_error = np.zeros(sample_size)
    sample_sizes = np.arange(10, 1010, 10)
    for i in range(sample_size // 10):
        univariate_2.fit(samples[:10 * (i + 1)])
        expectation_error[i] = abs(univariate_2.mu_ - expectation)

    layout_2 = go.Layout(dict(title="Q2) Error of Estimated Expectation of a Univariate Gaussian",
                              xaxis_title="Sample Size",
                              yaxis_title="Error",
                              yaxis_range=[0, 0.8]))
    fig_2 = go.Figure(data=go.Scatter(x=sample_sizes, y=expectation_error), layout=layout_2)
    fig_2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = univariate_1.pdf(samples)
    data_frame = pd.DataFrame({"Samples": samples, "PDF Values": pdfs})
    fig_3 = px.scatter(data_frame, x="Samples", y="PDF Values",
                       title="Q3) Empirical PDF of the Fitted Model")
    fig_3.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    expectation = np.array([0, 0, 4, 0])
    covariance = np.array([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(expectation, covariance, 1000)
    multivariate = MultivariateGaussian()
    multivariate.fit(samples)
    print("Q4) Estimated expectation and covariance of multivariate gaussian:")
    print("Expectation Vector:")
    print(multivariate.mu_)
    print("Covariance Matrix:")
    print(multivariate.cov_)
    print("\n")

    # Question 5 - Likelihood evaluation
    feature = np.linspace(-10, 10, 200)

    likelihood_mat = np.empty((feature.size, feature.size))
    for i in range(feature.size):
        for j in range(feature.size):
            cur_mu = np.array([feature[i], 0, feature[j], 0])
            likelihood_mat[i, j] = MultivariateGaussian.log_likelihood(cur_mu, covariance, samples)
    fig_5 = go.Figure(data=go.Heatmap(x=feature, y=feature, z=likelihood_mat,
                                      colorbar=dict(title="log-likelihood value")),
                      layout=dict(title="Q5) log-likelihood calculations for mu = [f1, 0, f3, 0]",
                                  xaxis_title="f3", yaxis_title="f1"))
    fig_5.show()

    # Question 6 - Maximum likelihood
    max_value = np.amax(likelihood_mat)
    max_indices = np.where(likelihood_mat == max_value)
    max_indices_zipped = list(zip(max_indices[0], max_indices[1]))
    f1 = round(feature[max_indices_zipped[0][0]], 3)
    f3 = round(feature[max_indices_zipped[0][1]], 3)
    print(f"Q5) The maximum log-likelihood value is: {round(max_value, 3)}\n"
          f"and the model that achieved it is:\n"
          f"[f1, 0, f3, 0] = [{f1}, 0, {f3}, 0]")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
