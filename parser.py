"""
Parser for the data from the .csv files.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class dataSet():
    """
    Class reading the data set from a given .csv file.
    """

    def __init__(self, file_name):
        """
        Initializes the data set.
        Writes the data from file_name to a pandas dataframe.
        Extracts the header from the dataframe and it writes it to a list.

        Arguments:
            file_name: The name of the file to be parsed.
        """
        self.file_name = file_name
        self.df = pd.read_csv(file_name)
        self.header = self.df.columns.values

    def plot_data_set(self):
        """
        Plots all columns as a function of the first column (assumed to be the date).
        """
        for i in range(len(self.header) - 1):
            fig = plt.figure(i)
            ax = plt.subplot(1, 1, 1)
            #ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            plt.plot(self.df[self.header[0]], self.df[self.header[i + 1]])
            plt.title(self.header[i + 1])
            fig.autofmt_xdate()
        plt.show()


if __name__ == '__main__':
    my_data_set = data_set("data/Water_Spring_Lupa.csv")
    print(my_data_set.file_name)
    print(my_data_set.df)
    print(my_data_set.header)
    my_data_set.plot_data_set()