from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def calculate_rsi(prices, periods=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Parameters:
    prices (pd.Series): Series of prices
    periods (int): RSI period (default=14)
    
    Returns:
    pd.Series: RSI values
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Create two series: gains (positive changes) and losses (negative changes)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate initial average gains and losses
    avg_gains = gains.rolling(window=periods).mean()
    avg_losses = losses.rolling(window=periods).mean()
    
    # Calculate subsequent average gains and losses
    for i in range(periods, len(prices)):
        avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (periods-1) + gains.iloc[i]) / periods
        avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (periods-1) + losses.iloc[i]) / periods
    
    # Calculate RS (Relative Strength)
    rs = avg_gains / (avg_losses + 1e-6)
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def create_features(df):
    df_features = df.copy()
    
    df_features['Prev_Close'] = df_features['Close'].shift(1)
    df_features['Prev_High'] = df_features['High'].shift(1)
    df_features['Prev_Low'] = df_features['Low'].shift(1)
    df_features['Prev_Volume'] = df_features['Volume'].shift(1)
    
    columns = ['Close', 'Volume', 'Open', 'High', 'Low']
    
    for col in columns:
        df_features[f'MA5_{col}'] = df_features[col].rolling(window=5).mean()
        df_features[f'MA10_{col}'] = df_features[col].rolling(window=10).mean()
        df_features[f'Volatility_{col}'] = df_features[col].rolling(window=5).std()

    # MACD
    for col in columns:
        short_ema = df_features[col].ewm(span=12).mean()
        long_ema = df_features[col].ewm(span=26).mean()
        df_features[f'MACD_{col}'] = short_ema - long_ema

    for col in columns:
        df_features[f'RSI_14_{col}'] = calculate_rsi(df_features[col], periods=14)
    
    df_features.dropna(inplace=True)
    df_features = df_features.drop(df_features.index[:12])
    return df_features



def main():
    df = pd.read_csv('data/orig/nvda_jan_dec.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df_processed = create_features(df)
    df_processed.to_csv('data/proc/nvda_jan_dec.csv')

if __name__ == "__main__":
    main()