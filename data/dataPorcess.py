import pandas as pd 

wine = pd.read_csv('data\winedata.csv',sep=';')

wine.to_csv('wine.csv',index=False)