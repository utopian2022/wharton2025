import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

tickers = [""]
df

for i in tickers:
    t = yf.download(i, period="1mo")
    t["Close"] = t["Close"].pct_change()
    t["20SMA"] = t["Close"].rolling(20).mean()
    t["50SMA"] = t["Close"].rolling(50).mean()
