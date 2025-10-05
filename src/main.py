import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

tickers = [""]
data = []

for i in tickers:
    t = yf.download(i, period="1mo")
    t["Close"] = t["Close"].pct_change()
    t["20SMA"] = t["Close"].rolling(20).mean()
    t["50SMA"] = t["Close"].rolling(50).mean()
    t = t[["Close", "20SMA", "50SMA", "Volume"]]
    data.append(t)

data = pd.concat(data)

scalers = {}
scaled_data = []

for ticker in tickers:
    df = data[data["Ticker"] == ticker].copy()
    scaler = MinMaxScaler()
    df[["Close", "20SMA", "50SMA", "Volume"]] = scaler.fit_transform(
        df[["Close", "20SMA", "50SMA", "Volume"]]
    )
    scalers[ticker] = scaler
    scaled_data.append(df)

data = pd.concat(scaled_data)


