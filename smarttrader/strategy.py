import pandas as pd

def strategy(prediction):
    prediction = (prediction - prediction["Close"].shift(1))[1:]
    decision  = prediction.mean(axis=1) > 0
    prev_decision = None
    trading_signals = []

    for date, is_bullish in decision.items():
        if prev_decision is None:
            signal = 'bullish' if is_bullish else 'bearish'
        elif is_bullish != prev_decision:
            signal = 'bullish' if is_bullish else 'bearish'
            
        trading_signals.append((date, signal.upper()))
        prev_decision = is_bullish

    signals_df = pd.DataFrame(trading_signals, columns=['Date', 'Signal']).set_index('Date')
    return signals_df

