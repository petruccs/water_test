from dataset import dataSet
from arima import *
"""
Tasks:
- L'obiettivo dell'esercizio è di scegliere uno di questi waterbodies (puoi scegliere quello che preferisci, per es., quello con più dati), per:

-- 1 Studiare le relazioni tra dati di input e variabile target (in generale, livello dell'acqua)
-- 2 Fare una previsione dell'andamento della variabile target nei successivi giorni
"""

file_name = "data/Lake_Bilancino.csv"
my_parser = dataSet(file_name)

#plot_auto_corr(my_parser.df, my_parser.df.columns[-1], True)

test_arima_model(my_parser.df[my_parser.df.columns[-1]], 5, 1, 0)