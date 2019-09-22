# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 03:55:18 2019

@author: brand
"""

import pickle
import tensorflow as tf
import pandas_datareader as data
import pandas as pd
from scipy import signal
import numpy as np
from flask import Flask, jsonify
app = Flask(__name__)


basket = ['AXP','AAPL','BA','CAT','CSCO','CVX','XOM','GS','HD','IBM','INTC','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NKE','PFE','PG','TRV','UNH','UTX','VZ','V','WBA','WMT','DIS','DOW']

X_scalers = pickle.load(open('C:/Users/brand/OneDrive/Documents/ShellHacks2019/FinetunedModels/x_scalers.pkl', 'rb'))
y_scalers = pickle.load(open('C:/Users/brand/OneDrive/Documents/ShellHacks2019/FinetunedModels/y_scalers.pkl', 'rb'))


specific_models = {}
specific_models = {}
for stock in basket:
    specific_models[stock] = tf.keras.models.load_model('C:/Users/brand/OneDrive/Documents/ShellHacks2019/FinetunedModels/finetuned_{}.h5'.format(stock))
    
cluster = []
for ticker in basket:
  r = data.DataReader(ticker,'yahoo',start = '2012-09-21')
  r['Symbol'] = ticker
  cluster.append(r)
new_days = pd.concat(cluster)
new_day = new_days.sort_index(ascending=True)
print(new_day.head())
new_day = new_day.reset_index()

for stock in basket:
    for col in ('Close', 'High', 'Low', 'Open', 'Volume','Adj Close'):
        new_day[col] = new_day[col].astype(float)
        new_day.loc[new_day['Symbol'] == stock, col] = signal.detrend(new_day[new_day['Symbol'] == stock][col])
    new_day.loc[new_day['Symbol'] == stock, 'mean_close_price_2'] = new_day.loc[new_day['Symbol'] == stock, 'Close'].rolling(window=2).mean()
    new_day.loc[new_day['Symbol'] == stock, 'mean_close_price_3'] = new_day.loc[new_day['Symbol'] == stock, 'Close'].rolling(window=3).mean()
    new_day.loc[new_day['Symbol'] == stock, 'std_close_price_2'] = new_day.loc[new_day['Symbol'] == stock, 'Close'].rolling(window=2).std()
    new_day.loc[new_day['Symbol'] == stock, 'std_close_price_3'] = new_day.loc[new_day['Symbol'] == stock, 'Close'].rolling(window=3).std()
    
new_day['Tomo_gain'] = new_day['Close'].shift(-1) - new_day['Close']
new_day['Yday_gain'] = new_day['Tomo_gain'].shift(1)

as_date = new_day['Date'].dt
new_day = new_day.drop(['Date'], axis=1)
new_day = new_day.dropna(axis=0)
new_day = new_day.reset_index(drop=True)
for stock in basket:
    new_day = new_day.drop(new_day.index[len(new_day[new_day['Symbol'] == stock]) - 1], axis=0)
    outliers = abs(new_day[new_day['Symbol'] == stock]['Tomo_gain']) < new_day[new_day['Symbol'] == stock]['Tomo_gain'].std() * 3
    new_day[new_day['Symbol'] == stock] = new_day[new_day['Symbol'] == stock].loc[:, :][outliers]
    new_day = new_day.drop(new_day[new_day['Symbol'] == stock].iloc[-1].name)
    for col in ('Close', 'High', 'Low', 'Open', 'Volume','Adj Close', 'mean_close_price_2', \
               'mean_close_price_3', 'std_close_price_2', 'std_close_price_3', 'Yday_gain'):
        pre_x = new_day[new_day['Symbol'] == stock][col]
        new_day.loc[new_day['Symbol'] == stock, col] = X_scalers[stock][col].transform(pre_x.values.reshape(-1,1))
new_day = new_day.dropna(axis=0)

dummies = pd.get_dummies(new_day['Symbol'], columns=['Symbol'])
num_df_cols = new_day.shape[1] - 1 + len(basket) - 1
print(num_df_cols)
def pad_stock(symbol):
    dumdums = np.zeros(len(basket))
    dumdums[list(dummies.columns.values).index(symbol)] = 1.
    return dumdums
#Ran everyday
s = []
pr = []
for stock in basket:
    today = new_day[new_day['Symbol'] == stock].iloc[-1].drop(['Tomo_gain', 'Symbol'])
    today = np.append(today, pad_stock(stock))
    today = np.array(today,np.float32)
    specific_models[stock].reset_states()
    pred = specific_models[stock].predict(np.reshape(today, (-1, 1, num_df_cols)))
    pred = y_scalers[stock].inverse_transform(pred)
    s.append(str(stock))
    pr.append(float(np.asscalar(pred)))
    print("{}, {}".format(stock, np.asscalar(pred)))


@app.route('/stocks', methods=['GET'])
def get_data():
  return "[{'Stock':'AXP','Pred':-10.19293499},{'Stock':'AAPL','Pred':-6.8215518},{'Stock':'BA','Pred':-26.69425392},{'Stock':'CAT','Pred':7.279784679},{'Stock':'CSCO','Pred':1.42284739},{'Stock':'CVX','Pred':9.909654617},{'Stock':'XOM','Pred':6.21008873},{'Stock':'GS','Pred':20.28046989},{'Stock':'HD','Pred':-17.59545135},{'Stock':'IBM','Pred':-5.495876312},{'Stock':'INTC','Pred':-0.03144864738},{'Stock':'JNJ','Pred':0.7567970753},{'Stock':'KO','Pred':3.664332628},{'Stock':'JPM','Pred':-0.8951063752},{'Stock':'MCD','Pred':-18.83733749},{'Stock':'MMM','Pred':47.27815247},{'Stock':'MRK','Pred':-8.549324989},{'Stock':'MSFT','Pred':-8.150383949},{'Stock':'NKE','Pred':4.573824406},{'Stock':'PFE','Pred':-1.110576391},{'Stock':'PG','Pred':-16.45113373},{'Stock':'TRV','Pred':1.322250247},{'Stock':'UNH','Pred':21.20621681},{'Stock':'UTX','Pred':-2.709911585},{'Stock':'VZ','Pred':-1.475743651},{'Stock':'V','Pred':-18.6704216},{'Stock':'WBA','Pred':30.91255188},{'Stock':'WMT','Pred':-8.505377769},{'Stock':'DIS','Pred':-7.191668034},{'Stock':'DOW','Pred':20.12534714}]"

if __name__ == '__main__':
    app.run()
    
#da.to_json()

