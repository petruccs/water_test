import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg


def plot_auto_corr(df, column, interpolate, save_fig):
    """
    Plots the autocorrelation function of the given column.
    Parameters:
        df: The dataframe containing the data.
        column: The column to be plotted.
        interpolate: If True, the NaN values are interpolated using the linear method.
    """
    plt.figure()
    if interpolate:
        pd.plotting.autocorrelation_plot(df[column].interpolate())
    else:
        pd.plotting.autocorrelation_plot(df[column])
    plt.title(f"Autocorrelation of {column}")
    if save_fig:
        plt.savefig(f"plots/{column.lower()}_autocorr.pdf")
    else:
        plt.show()
    plt.close()


def run_ar_model(train_data, test_data, lag, save_fig):
    """
    Runs the AR model with the given parameters and plots the prediction.
    """
    # Train the autoregression model
    model = AutoReg(train_data, lag)
    # fit the model
    model_fit = model.fit()
    #print(f"[INFO] The coefficients of the model are:\n{model_fit.params}")
    # make prediction
    predictions = model_fit.predict(start=len(train_data),
                                    end=len(train_data) + len(test_data) - 1,
                                    dynamic=False)
    compare_df = pd.concat([test_data, predictions],
                           axis=1).rename(columns={
                               test_data.name: "testing sample",
                               0: f"predicted - lag {lag}"
                           })
    compare_df.plot()
    if save_fig:
        plt.savefig(f"plots/ar_{train_data.name.lower()}_prediction.pdf")
    else:
        plt.show()
        plt.close()


def test_arima_model(series, p, d, q):
    """
    Tests the ARIMA model with the given parameters.
    Parameters:
        series: The series to be tested.
        p: number of time lags
        d: degree of differencing
        q: order of the moving average model
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


def run_arima_prediction(train_data, p, d, q, test_data, save_fig):
    """
    Runs the ARIMA model with the given parameters.
    Parameters:
        series: The series to be tested.
        p: number of time lags
        d: degree of differencing
        q: order of the moving average model
    """
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()
    predictions = model_fit.forecast(len(test_data))
    compare_df = pd.concat(
        [test_data, predictions],
        axis=1).rename(columns={
            test_data.name: "testing sample",
            predictions.name: f"predicted - lag {p}"
        })
    compare_df.plot()
    if save_fig:
        plt.savefig(f"plots/arima_{train_data.name.lower()}_prediction.pdf")
    else:
        plt.show()
        plt.close()