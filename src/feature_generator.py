import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

# read the csv file
data=pd.read_csv('C:/Users/BRENDA MUTAI/Documents/crypto-classifier/data/processed/crypto_data.csv')
data.head()

# check 1-day return
data['1_day_return'] = data['close'].pct_change(periods=1)
data[['close', '1_day_return']].head()

# check 7-day return
data['7_day_return'] = data['close'].pct_change(periods=7)
data[['close', '7_day_return']].head()

# checking the rolling volatility
data['7_day_volatility'] = data['1_day_return'].rolling(window=7).std()
data[['1_day_return', '7_day_volatility']].tail()


#Using `ta` or `ta-lib`:

# RSI\
 #MACD\
 #Moving averages (SMA20, SMA50, SMA200)\
 #Bollinger Bands\
 #Stochastic Oscillator
data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
data['macd'] = ta.trend.MACD(data['close']).macd()
data['sma20'] = ta.trend.SMAIndicator(data['close'], window=20).sma_indicator()
data['sma50'] = ta.trend.SMAIndicator(data['close'], window=50).sma_indicator()
data['sma200'] = ta.trend.SMAIndicator(data['close'], window=200).sma_indicator()
bb_indicator = ta.volatility.BollingerBands(data['close'])
data['bb_high'] = bb_indicator.bollinger_hband()
data['bb_low'] = bb_indicator.bollinger_lband()
data['stochastic_oscillator'] = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close']).stoch()
data.tail()

#need to save the new data with features and labels
# data.to_csv('../data/processed/crypto_data_features_labels.csv', index=False)
data= data.dropna().reset_index(drop=True)
data.info()

# drop ignore column
#data = data.drop(columns=['ignore'])

# change date to datetime
data["open_time"] = pd.to_datetime(data["open_time"])
data["close_time"] = pd.to_datetime(data["close_time"])
data["close"] = data["close"].astype(float)
data.head()
data.info()

# save to csv
data.to_csv('data\processed\crypto_data_features_labels.csv', index=False)