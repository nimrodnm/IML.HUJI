import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from scipy import stats

pio.templates.default = "simple_white"

DATA_PATH = "G:/My Drive/Semester_4/IML/IML.HUJI/datasets/City_Temperature.csv"
MAX_ZSCORE = 5


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date']).drop_duplicates()

    # Adding DayOfYear column:
    data['DayOfYear'] = data['Date'].dt.dayofyear

    # Removing outliers - samples with extreme Temp values:
    data = data[(np.abs(stats.zscore(data['Temp'])) < MAX_ZSCORE)]

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(DATA_PATH)

    # Question 2 - Exploring data for specific country
    il_df = df.loc[df.Country.astype(str) == "Israel"].reset_index(drop=True)
    il_df.Year = il_df.Year.astype(str)
    px.scatter(il_df, x='DayOfYear', y='Temp', color='Year',
               title="Temperature in TLV as a function of DayOfYear").show()
    px.bar(il_df.groupby('Month').agg({'Temp': 'std'}).reset_index().
           rename(columns={'Temp': 'Temp_std'}),
           x='Month', y='Temp_std',
           title="Standard Deviation of Daily Temperatures Per Month").show()

    # Question 3 - Exploring differences between countries
    countries_df = df.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
    countries_df.columns = countries_df.columns.droplevel(0)
    countries_df.columns = ['Country', 'Month', 'Temp_mean', 'Temp_std']
    px.line(countries_df, x='Month', y='Temp_mean', color='Country', error_y='Temp_std',
            title="Average Temperature in 4 Countries Per Month, Including Standard Deviation").\
        update_xaxes(tick0=1, dtick=1).show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(il_df['DayOfYear'], il_df['Temp'])
    mse_per_k = []
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(train_x.to_numpy(), train_y.to_numpy())
        mse = np.round(poly.loss(test_x.to_numpy(), test_y.to_numpy()), 3)
        mse_per_k.append((k, mse))
        print(f"k={k} \t loss = {mse_per_k[-1][1]}")

    mse_per_k_df = pd.DataFrame(mse_per_k, columns=['k', 'MSE'])
    px.bar(mse_per_k_df, x=mse_per_k_df['k'].astype(str), y='MSE',
           title="Test Error For Each Value of k", text='MSE', labels=dict(x='k')).show()

    # Question 5 - Evaluating fitted model on different countries
    best_k = min(mse_per_k, key=lambda tup: tup[1])[0]
    il_poly_fitter = PolynomialFitting(best_k)
    il_poly_fitter.fit(il_df['DayOfYear'].to_numpy(), il_df['Temp'].to_numpy())
    results = []
    for country in df.Country.unique():
        country_df = df[['DayOfYear', 'Temp']].loc[df.Country.astype(str) == country].\
            reset_index(drop=True)
        mse = il_poly_fitter.loss(country_df.DayOfYear.to_numpy(), country_df.Temp)
        results.append((country, mse))

    results_df = pd.DataFrame(results, columns=['Country', 'MSE'])
    px.bar(results_df, x='Country', y='MSE',
           title=f"Prediction Error (MSE) On Each Country of Model Fitted with k={best_k}",
           text='MSE').show()
