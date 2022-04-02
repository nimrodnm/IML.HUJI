from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).drop_duplicates()

    # remove rows with invalid values:
    to_remove = pd.concat([df.loc[(df.bedrooms <= 0) & (df.bathrooms <= 0)],
                           df.loc[(df.sqft_living <= 0) | (df.sqft_lot <= 0) | (df.sqft_above <= 0)
                                  | (df.sqft_basement < 0) | (df.sqft_living15 < 0)
                                  | (df.sqft_lot15 < 0) | (df.price < 0)
                                  | (df.price.isnull())]]).drop_duplicates()
    df.drop(to_remove.index, inplace=True)

    # parse the date column:
    df['date'] = pd.to_datetime(df.date, errors='coerce')

    # delete samples with no date:
    df.drop(df.loc[df.date.isnull()].index, inplace=True)

    # replace date column with year month and day columns:
    df['sale_year'] = df.date.dt.year
    df['sale_month'] = df.date.dt.month
    df['sale_day'] = df.date.dt.weekday

    # remove redundant columns:
    df.drop(columns=['id', 'date', 'lat', 'long'], inplace=True)

    # create column for age:= sale_year - max(yr_built, yr_renovated):
    df['age'] = df.sale_year - np.maximum(df.yr_built, df.yr_renovated)

    # Create columns with ratio between avg square-fit of nearby houses to square-fit of each house:
    # (no division by zero because I removed rows with sqft_living==0 or sqft_lot==0)
    df['sqft_living_ratio'] = df.sqft_living15 / df.sqft_living
    df['sqft_lot_ratio'] = df.sqft_lot15 / df.sqft_lot

    # do one-hot encoding for the zipcode feature:
    df = pd.get_dummies(df, columns=['zipcode'])

    response = df.pop('price')
    return df, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_values = np.array(y)

    for col_name, col in X.iteritems():
        values = np.array(col)
        correlation = pearson_correlation(values, y_values)
        print(correlation)
        px.scatter(x=values, y=y_values, trendline="ols", trendline_color_override='skyblue',
                   title=f"Pearson Correlation of {col_name} and price = {correlation}",
                   labels=dict(x=col_name, y="price")).write_image(f"{output_path}/{col_name}.png")


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the pearson correlation of the 2 given sample vectors
    """
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_centered_norm = np.linalg.norm(x_centered)
    y_centered_norm = np.linalg.norm(y_centered)
    return np.inner(x_centered, y_centered) / (x_centered_norm * y_centered_norm)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    plots_path = "G:/My Drive/Semester_4/IML/IML.HUJI/exercises/ex2/plots"
    feature_evaluation(df, response, plots_path)

    # Question 3 - Split samples into training- and testing sets.
    train_data, train_responses, test_data, test_responses = split_train_test(df, response, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    linear_model = LinearRegression()
    percentages = np.arange(10, 101)
    loss_means, loss_stds = [], []
    for percentage in percentages:
        losses = []

        for i in range(10):
            print("\n%=", percentage, "i=", i)
            train_sample = train_data.sample(frac=(percentage / 100), random_state=i)
            response_sample = train_responses.sample(frac=(percentage / 100), random_state=i)
            linear_model.fit(np.array(train_sample), np.array(response_sample))
            losses.append(linear_model.loss(np.array(test_data), np.array(test_responses)))

        loss_means.append(np.array(losses).mean())
        loss_stds.append(np.array(losses).std())

    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    loss_means = np.array(loss_means)
    loss_stds = np.array(loss_stds)
    fig = go.Figure(data=[go.Scatter(x=percentages, y=loss_means, mode="markers+lines",
                                     marker=dict(color="blue", opacity=0.7),
                                     name="MSE values"),
                          go.Scatter(x=percentages, y=(loss_means - (2 * loss_stds)),
                                     fill=None, mode="lines", line=dict(color="lightgrey"),
                                     showlegend=False),
                          go.Scatter(x=percentages, y=(loss_means + (2 * loss_stds)),
                                     fill="tonexty", mode="lines", line=dict(color="lightgrey"),
                                     showlegend=False)],
                    layout=dict(title="Average Loss of House Price Prediction",
                                xaxis_title="Percentages",
                                yaxis_title="Average Loss"))
    fig.show()
