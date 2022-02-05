import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def plot_auto_corr(df, column, interpolate):
    """
    Plots the autocorrelation function of the given column.
    Parameters:
        df: The dataframe containing the data.
        column: The column to be plotted.
        interpolate: If True, the NaN values are interpolated using the linear method.
    """
    print(
        f"[INFO] plotting the autocorrelation of {column}; interpolate = {interpolate}"
    )

    plt.figure()
    if interpolate:
        pd.plotting.autocorrelation_plot(df[column].interpolate())
    else:
        pd.plotting.autocorrelation_plot(df[column])
    plt.title(f"Autocorrelation of {column}")
    plt.show()


def test_arima_model(series, p, d, q):
    """
    Tests the ARIMA model with the given parameters.
    Parameters:
        series: The series to be tested.
        p: Lag
        d: 
        q: 
    """
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())


def run_arima_prediction(series, p, d, q, n_pred):
    """
    Runs the ARIMA model with the given parameters.
    Parameters:
        series: The series to be tested.
        p: Lag
        d: 
        q: 
    """
    #res_seq = []
    #for i in range(n_pred):
    #    print(i)
    #    model = ARIMA(series, order=(p, d, q))
    #    model_fit = model.fit()
    #    #res_seq.append(model_fit.forecast()[0])
    #    print(model_fit.forecast())
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    return model_fit.forecast(n_pred)