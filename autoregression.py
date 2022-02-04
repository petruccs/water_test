import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def plot_auto_corr(df, column, interpolate):
    """
    Plots the autocorrelation function of the given column.
    """
    print(
        f"[INFO] plotting the autocorrelation of {column}; interpolate = {interpolate}"
    )

    plt.figure()
    if interpolate:
        pd.plotting.autocorrelation_plot(df[column].interpolate())
    else:
        pd.plotting.autocorrelation_plot(df[column])

    plt.show()


def test_arima_model(series, p, d, q):
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())