from numpy import save
from dataset import dataSet
from autoregression import *
import seaborn as sns
import matplotlib.pyplot as plt
"""
Tasks:
- L'obiettivo dell'esercizio è di scegliere uno di questi waterbodies (puoi scegliere quello che preferisci, per es., quello con più dati), per:

-- 1 Studiare le relazioni tra dati di input e variabile target (in generale, livello dell'acqua)
-- 2 Fare una previsione dell'andamento della variabile target nei successivi giorni
"""

# General configurations
save_figs = True
lag_lake = 10  # Optimal is 115, too demanding for my system
lag_flow = 10  # Optimal is 65, too demanding for my system
lake_level_name = "Lake_Level"
flow_rate_name = "Flow_Rate"

# Configuring and importing the dataset
file_name = "data/Lake_Bilancino_cropped.csv"
my_parser = dataSet(file_name)
print(my_parser.df.describe(include='all'))

# Generating the test and train dataframes
my_parser.generate_test_train_df(0.998)
print(f"[INFO] Testing data set size: {len(my_parser.test_df)}")

# Computing the correlation matrix of the data set
# This is a preliminary answer to the first question
corr = my_parser.df.corr()

fig, ax = plt.subplots()
sns.heatmap(corr,
            annot=True,
            fmt='.4f',
            cmap=plt.get_cmap('coolwarm'),
            cbar=False,
            ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
if save_figs:
    plt.savefig("plots/corr_vars.pdf")
else:
    plt.show()
plt.close()

# Computing the autoregression of the data set
# Studying this plot allows to find the optimal lag for the autoregression

plot_auto_corr(my_parser.df, flow_rate_name, True, save_figs)
plot_auto_corr(my_parser.df, lake_level_name, True, save_figs)

# Running the AR model for both target variables
run_ar_model(my_parser.train_df[flow_rate_name],
             my_parser.test_df[flow_rate_name], lag_flow, save_figs)
run_ar_model(my_parser.train_df[lake_level_name],
             my_parser.test_df[lake_level_name], lag_lake, save_figs)

# Functions to the ARIMA performance
test_arima_model(my_parser.df[flow_rate_name], lag_flow, 1, 0)
test_arima_model(my_parser.df[lake_level_name], lag_lake, 1, 0)

# Computing the ARIMA prediction for both target variables
run_arima_prediction(my_parser.train_df[flow_rate_name], lag_flow, 1, 0,
                     my_parser.test_df[flow_rate_name], save_figs)
run_arima_prediction(my_parser.train_df[lake_level_name], lag_lake, 1, 0,
                     my_parser.test_df[lake_level_name], save_figs)