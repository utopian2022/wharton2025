import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

tickers = ["MSFT"]
data = []

for i in tickers:
    t = yf.download(i, period="6mo")
    t["Close"] = t["Close"].pct_change()
    t["20SMA"] = t["Close"].rolling(20).mean()
    t["50SMA"] = t["Close"].rolling(50).mean()
    t = t[["Close", "20SMA", "50SMA", "Volume"]]
    t = t.dropna()
    t = t.reset_index(level='Date')
    t.columns = t.columns.droplevel(1)
    print(t)
    data.append(t)

data = pd.concat(data)

scalers = {}
scaled_data = []

for i in data:
    df = i
    scaler = MinMaxScaler()
    df[["Close", "20SMA", "50SMA", "Volume"]] = scaler.fit_transform(df)
    scalers[ticker] = scaler
    scaled_data.append(df)

data = pd.concat(scaled_data)

print(data)

