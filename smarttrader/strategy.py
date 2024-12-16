# if True save bullish for that day else save bearish, if decison remains the same for another day, keep idle
# Initialize previous decision as None to track changes
import pandas as pd

def strategy(predictions):
    
    dates = predictions["Date"].reset_index(drop=True).iloc[1:].to_numpy()
    predictions = predictions[['Open', 'High', 'Low', 'Close']].reset_index(drop=True)
    # Weighted sum for open and high
    predictions["Open"] = predictions["Open"] * 1.01
    predictions["High"] = predictions["High"] * 1.005
    # print(predictions)
    close = predictions["Close"].shift(1).dropna()
    predictions = predictions.iloc[1:].to_numpy()
    close = close.to_numpy()
    predictions = predictions - close.reshape(-1, 1)
    decision  = predictions.mean(axis=1) > 0
    # print(predictions)
    trading_signals = []
    # dates = range(len(decision))

    for date, is_bullish in zip(dates, decision):
        signal = 'BULLISH' if is_bullish else 'BEARISH'
        trading_signals.append([date, signal])

    signals_df = pd.DataFrame(trading_signals, columns=['Date', 'Signal'])
    return signals_df