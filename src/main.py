
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import sys
import os
import argparse
import pickle


tickers = ["PEN", "LSCC", "AUR", "WBS", "MDGL", "ALV", "RRX"]
data = []

for ticker in tickers:
    t = yf.download(ticker, period="6mo")
    if t.empty:
        print(f"No data for {ticker}")
        continue

    if isinstance(t.columns, pd.MultiIndex) and t.columns.nlevels >= 2:
        t.columns = t.columns.droplevel(1)

    t = t.reset_index()

    t["Price"] = t["Close"].astype(float)
    t["Return"] = t["Price"].pct_change()
    t["20SMA"] = t["Price"].rolling(20).mean()
    t["50SMA"] = t["Price"].rolling(50).mean()

    lookahead = 21
    t["FuturePrice"] = t["Price"].shift(-lookahead)
    t["Target"] = ((t["FuturePrice"] / t["Price"] - 1) >= 0.01).astype(float)

    t = t[["Date", "Price", "Return", "20SMA", "50SMA", "Volume", "FuturePrice", "Target"]]

    cols_to_scale = ["Return", "20SMA", "50SMA", "Volume"]
    t[cols_to_scale] = t[cols_to_scale].apply(pd.to_numeric, errors='coerce')

    t = t.dropna()
    t["Ticker"] = ticker
    data.append(t)

if not data:
    raise SystemExit("No data downloaded for any tickers")

data = pd.concat(data, ignore_index=True)
scalers = {}
scaled_frames = []

for ticker, group in data.groupby("Ticker", sort=False):
    df = group.copy()
    df = df.sort_values("Date")
    cols = ["Return", "20SMA", "50SMA", "Volume"]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    scalers[ticker] = scaler
    scaled_frames.append(df)

data = pd.concat(scaled_frames, ignore_index=True)

def make_sequences(df, feature_cols, target_col, window_size=20, lookahead=21):
    X, y = [], []
    arr = df[feature_cols].values
    target = df[target_col].values
    max_start = len(df) - window_size - lookahead + 1
    for i in range(max_start):
        X.append(arr[i:i+window_size])
        y.append(target[i + window_size - 1])
    return np.array(X), np.array(y)


feature_cols = ["Return", "20SMA", "50SMA", "Volume"]
target_col = "Target"
window_size = 20

all_X, all_y = [], []
for ticker, group in data.groupby("Ticker", sort=False):
    g = group.sort_values("Date")
    if len(g) <= window_size:
        continue
    Xg, yg = make_sequences(g, feature_cols, target_col, window_size)
    all_X.append(Xg)
    all_y.append(yg)

if not all_X:
    raise SystemExit("Not enough data to create sequences")

X = np.vstack(all_X)
y = np.hstack(all_y)

print(f"Prepared sequences: X.shape={X.shape}, y.shape={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


n_features = X.shape[2]
model = Sequential([
    LSTM(64, input_shape=(window_size, n_features), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, callbacks=[es], verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.6f}, Test acc: {test_acc:.4f}")

preds = model.predict(X_test[:20])
print("pred_prob, pred_class, true (first 20):")
for p, tval in zip(preds.flatten(), y_test[:20]):
    cls = int(p >= 0.5)
    print(f"{p:.4f}, {cls}, {int(tval)}")


def predict_ticker(ticker_symbol, model, scalers, window_size=20):
    """Download recent data for ticker_symbol, compute features, scale, and predict.

    Returns (probability, class) or raises ValueError on insufficient data.
    """
    t = yf.download(ticker_symbol, period="6mo")
    if t.empty:
        raise ValueError(f"No data for {ticker_symbol}")
    if isinstance(t.columns, pd.MultiIndex) and t.columns.nlevels >= 2:
        t.columns = t.columns.droplevel(1)
    t = t.reset_index()
    t["Price"] = t["Close"].astype(float)
    t["Return"] = t["Price"].pct_change()
    t["20SMA"] = t["Price"].rolling(20).mean()
    t["50SMA"] = t["Price"].rolling(50).mean()
    cols = ["Return", "20SMA", "50SMA", "Volume"]
    t = t[["Date"] + cols].dropna()
    if len(t) < window_size:
        raise ValueError(f"Not enough data for {ticker_symbol} (need >= {window_size} rows after dropna)")

    if ticker_symbol in scalers:
        scaler = scalers[ticker_symbol]
        Xcols = scaler.transform(t[cols])
    else:
        scaler = StandardScaler()
        Xcols = scaler.fit_transform(t[cols])

    X_recent = Xcols[-window_size:]
    X_recent = np.expand_dims(X_recent, axis=0)
    prob = float(model.predict(X_recent, verbose=0).flatten()[0])
    cls = int(prob >= 0.5)
    return prob, cls


def save_artifacts(model, scalers, model_path='model.h5', scalers_path='scalers.pkl'):
    model.save(model_path)
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)


def load_artifacts(model_path='model.h5', scalers_path='scalers.pkl'):
    from tensorflow.keras.models import load_model
    if not os.path.exists(model_path) or not os.path.exists(scalers_path):
        raise FileNotFoundError('Saved model or scalers not found. Please train first.')
    m = load_model(model_path)
    with open(scalers_path, 'rb') as f:
        s = pickle.load(f)
    return m, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-only', action='store_true', help='Load saved model and predict for --ticker without training')
    parser.add_argument('--ticker', type=str, help='Ticker to predict when --predict-only is used')
    args = parser.parse_args()

    if args.predict_only:
        if not args.ticker:
            print('When using --predict-only you must provide --ticker TICKER', file=sys.stderr)
            sys.exit(2)
        try:
            m, s = load_artifacts()
            prob, cls = predict_ticker(args.ticker.strip().upper(), m, s, window_size=window_size)
            print(cls)
            return
        except Exception as e:
            print(f'Could not predict for {args.ticker}: {e}', file=sys.stderr)
            sys.exit(1)

    save_artifacts(model, scalers)
    while True:
        user_t = input('\nEnter a ticker to predict (or blank to exit): ').strip().upper()
        if user_t == '':
            break
        try:
            _, cls = predict_ticker(user_t, model, scalers, window_size=window_size)
            print(cls)
        except Exception as e:
            print(f"Could not predict for {user_t}: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()



