import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime


class dataSet:
    """
    Importing the data set from a .csv file.
    """

    def __init__(self, file_name):
        """
        Initializes the data_set.
        Writes the data from file_name to a pandas dataframe.
        Extracts the header from the dataframe and it writes it to a list.
    
        Arguments:
            file_name: The name of the file to be parsed.
        """
        self.file_name = file_name
        #self.df = pd.read_csv(file_name,
        #                      parse_dates=[0],
        #                      infer_datetime_format=True)

        self.df = pd.read_csv(file_name)
        self.header = self.df.columns.values

    def plot_data_set(self):
        """
        Plots all columns as a function of the first column (assumed to be the date).
        """
        for i in range(len(self.header) - 1):
            fig = plt.figure(i)
            ax = plt.subplot(1, 1, 1)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            plt.plot(self.df[self.header[0]], self.df[self.header[i + 1]])
            plt.title(self.header[i + 1])
            fig.autofmt_xdate()
        plt.show()

    def plot_interpolated(self):
        """
        Plots all columns as a function of the first column (assumed to be the date).
        Interpolating the NaN values using the linear method.
        """
        for i in range(len(self.header) - 1):
            fig = plt.figure(i)
            ax = plt.subplot(1, 1, 1)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            plt.plot(self.df[self.header[0]],
                     self.df[self.header[i + 1]].interpolate())
            plt.title(self.header[i + 1])
            fig.autofmt_xdate()
        plt.show()

    def generate_test_train_df(self, train_frac=0.8):
        """
        Splits the dataframe into a train and test dataframe.
        The test dataframe is of size test_size * len(df).
        Arguments:
            train_frac: The fraction of the train dataframe.
        """
        train_size = int(len(self.df) * train_frac)
        self.train_df = self.df[:train_size]
        self.test_df = self.df[train_size:]


if __name__ == '__main__':
    file_name = "data/Lake_Bilancino_cropped.csv"
    my_data_set = dataSet(file_name)
    my_data_set.plot_data_set()
