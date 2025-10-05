import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

tickers = ["MSFT"]
data = []

for ticker in tickers:
    t = yf.download(ticker, period="6mo")
    if t.empty:
        print(f"No data for {ticker}")
        continue

    t["Close"] = t["Close"].pct_change()
    t["20SMA"] = t["Close"].rolling(20).mean()
    t["50SMA"] = t["Close"].rolling(50).mean()

    t = t[["Close", "20SMA", "50SMA", "Volume"]]

    cols_to_scale = ["Close", "20SMA", "50SMA", "Volume"]
    t[cols_to_scale] = t[cols_to_scale].apply(pd.to_numeric, errors='coerce')

    t = t.dropna()
    t["Ticker"] = ticker
    print(t)
    data.append(t)

if not data:
    raise SystemExit("No data downloaded for any tickers")

data = pd.concat(data, ignore_index=True)
scalers = {}
scaled_data = []

for ticker, group in data.groupby("Ticker", sort=False):
    df = group.copy()
    cols = ["Close", "20SMA", "50SMA", "Volume"]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    scalers[ticker] = scaler
    scaled_data.append(df)

data = pd.concat(scaled_data, ignore_index=True)

print(data)

