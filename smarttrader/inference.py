import torch
import pandas as pd
import numpy as np
from transformer import StockTransformer
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import argparse
import pandas_market_calendars as mcal
import yaml

# pd.set_option('display.max_columns', 10)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

features = ['Open', 'High', 'Low', 'Close', 'Volume',
    'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Volume', 
    'MA5_Close', 'MA10_Close', 'MA5_Volume', 'MA10_Volume', 'MA5_Open', 
    'MA10_Open', 'MA5_High', 'MA10_High', 'MA5_Low', 'MA10_Low',
    'Volatility_Close', 'Volatility_Volume', 'Volatility_Open', 'Volatility_High', 'Volatility_Low',
    'MACD_Close', 'MACD_Volume', 'MACD_Open', 'MACD_High', 'MACD_Low',
    'RSI_14_Close', 'RSI_14_Volume', 'RSI_14_Open', 'RSI_14_High', 'RSI_14_Low']

def calculate_rsi(prices, periods=14):
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gains = gains.rolling(window=periods).mean()
    avg_losses = losses.rolling(window=periods).mean()
    for i in range(periods, len(prices)):
        avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (periods-1) + gains.iloc[i]) / periods
        avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (periods-1) + losses.iloc[i]) / periods
    rs = avg_gains / (avg_losses + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def load_model(model_path, config):
    model = StockTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward']
    )
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location='cpu'))
    model.eval()
    return model

def get_historic_data(df, target_date, seq_length=10):
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d')
    
    df['Date'] = pd.to_datetime(df['Date'])
    target_idx = df[df['Date'] <= target_date].index[-1]
    if target_idx < seq_length:
        raise ValueError("Not enough historical data for prediction")
    
    sequence = df.iloc[target_idx-seq_length:target_idx]
    # print(f"Predicting using data from: {sequence['Date'].iloc[0].date()} to {sequence['Date'].iloc[-1].date()}")
    return sequence

def predict_next_day(model, scaler, sequence_data, features):
    sequence = sequence_data[features].values
    sequence_normalized = scaler.transform(sequence)
    sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0)
    
    with torch.no_grad():
        prediction_normalized = model(sequence_tensor)
        
    dummy = np.zeros((1, len(features)))
    dummy[:, :5] = prediction_normalized.numpy()
    prediction = scaler.inverse_transform(dummy)[:, :5]
    
    return prediction[0]

def predict_sequential(model, scaler, initial_sequence, features, dates):
    current_sequence = initial_sequence.copy()
    predictions = []
    
    for i in range(len(dates)):
        pred = predict_next_day(model, scaler, current_sequence, features)
        predictions.append(pred)
        
        new_row = pd.DataFrame([{
            'Date': dates[i],
            'Open': float(pred[0]),
            'High': float(pred[1]),
            'Low': float(pred[2]),
            'Close': float(pred[3]),
            'Volume': float(pred[4]),
            'Prev_Close': float(current_sequence['Close'].iloc[-1]),
            'Prev_High': float(current_sequence['High'].iloc[-1]),
            'Prev_Low': float(current_sequence['Low'].iloc[-1]),
            'Prev_Volume': float(current_sequence['Volume'].iloc[-1]),
        }])
        temp_sequence = pd.concat([current_sequence, new_row], ignore_index=True)
        columns = ['Close', 'Volume', 'Open', 'High', 'Low']
        
        for col in columns:
            new_row[f'MA5_{col}'] = temp_sequence[col].rolling(window=5).mean().iloc[-1]
            new_row[f'MA10_{col}'] = temp_sequence[col].rolling(window=10).mean().iloc[-1]
            new_row[f'Volatility_{col}'] = temp_sequence[col].rolling(window=5).std().iloc[-1]

        for col in columns:
            short_ema = temp_sequence[col].ewm(span=12).mean().iloc[-1]
            long_ema = temp_sequence[col].ewm(span=26).mean().iloc[-1]
            new_row[f'MACD_{col}'] = short_ema - long_ema
        
        for col in columns:
            new_row[f'RSI_14_{col}'] = calculate_rsi(temp_sequence[col], periods=14).iloc[-1]
        
        current_sequence = pd.concat([current_sequence.iloc[1:], new_row], ignore_index=True)
        
        # print(new_row.iloc[:, :6].to_string(index=False))
    # return predictions[-1]
    # return last 5 rows of current_sequence
    return current_sequence[['Date', 'Open', 'High', 'Low', 'Close']].iloc[-6:]
    # return predictions

def run_inference(date):
    model_config = {
        'input_dim': len(features),
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256
    }
    model = load_model(config['MODEL_PATH'], model_config)
    df = pd.read_csv(config['DATA_PATH'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    train_size = int(0.9 * len(df))
    train_data = df[features].values[:train_size]
    # print(df['Date'].iloc[train_size-1])

    scaler = MinMaxScaler()
    scaler.fit(train_data)
    
    nyse = mcal.get_calendar('NYSE')
    start_date = pd.to_datetime(date)
    
    schedule = nyse.schedule(start_date=start_date + pd.Timedelta(days=1), end_date=start_date + pd.Timedelta(days=10))
    trading_dates = schedule.index[:5].tolist()
    # print("\nNext 5 trading dates:")
    # for date in trading_dates:
    #     print(date.strftime('%Y-%m-%d'), end=' ')
    # print()

    try:
        sequence = get_historic_data(df, trading_dates[0], config['SEQ_LENGTH'])
        predictions = predict_sequential(model, scaler, sequence, features, trading_dates)
        # print(prediction)
        return predictions
    except ValueError as e:
        print(f"[INFERENCE] ValueError: {e}")
        return None
    except Exception as e:
        print(f"[INFERENCE] Exception: {e}")
        return None

# def main():
#     parser = argparse.ArgumentParser(description='Stock price prediction inference')
#     parser.add_argument('--model', type=str, required=True,
#                       help='Path to the trained model file')
#     parser.add_argument('--date', type=str, default='2024-11-20',
#                       help='Target date for prediction (YYYY-MM-DD)')
#     args = parser.parse_args()

#     model_config = {
#         'input_dim': len(features),
#         'd_model': 128,
#         'nhead': 4,
#         'num_layers': 2,
#         'dim_feedforward': 256
#     }
    
#     model = load_model(args.model, model_config)
    
#     df = pd.read_csv('data/proc/nvda_may_dec.csv')
#     df = df.sort_values('Date').reset_index(drop=True)
    
#     train_size = int(0.9 * len(df))
#     train_data = df[features].values[:train_size]

#     # print(df['Date'].iloc[train_size-1])

#     scaler = MinMaxScaler()
#     scaler.fit(train_data)
    
#     nyse = mcal.get_calendar('NYSE')
#     start_date = pd.to_datetime(args.date)
    
#     schedule = nyse.schedule(start_date=start_date + pd.Timedelta(days=1), end_date=start_date + pd.Timedelta(days=10))
#     trading_dates = schedule.index[:5].tolist()
#     print("\nNext 5 trading dates:")
#     for date in trading_dates:
#         print(date.strftime('%Y-%m-%d'), end=' ')
#     print()

#     try:
#         sequence = get_historic_data(df, trading_dates[0], config['SEQ_LENGTH'])
#         prediction = predict_sequential(model, scaler, sequence, features, trading_dates)
#         # prediction = predict_next_day(model, scaler, sequence, features)
        
#         print(f"\nPrediction for {trading_dates[-1].date()}:")
#         print(f"Open:   ${prediction[0]:.2f}")
#         print(f"High:   ${prediction[1]:.2f}")
#         print(f"Low:    ${prediction[2]:.2f}")
#         print(f"Close:  ${prediction[3]:.2f}")
#         print(f"Volume: {prediction[4]:.0f}")
        
        
#         # actual = df[df['Date'] == trading_dates[-1]]
#         # if not actual.empty:
#         #     print("\nActual values:")
#         #     print(f"Open:   ${actual['Open'].values[0]:.2f}")
#         #     print(f"High:   ${actual['High'].values[0]:.2f}")
#         #     print(f"Low:    ${actual['Low'].values[0]:.2f}")
#         #     print(f"Close:  ${actual['Close'].values[0]:.2f}")
#         #     print(f"Volume: {actual['Volume'].values[0]:.0f}")
            
#     except ValueError as e:
#         print(f"Error: {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# def main_test():
#     parser = argparse.ArgumentParser(description='Stock price prediction inference')
#     parser.add_argument('--date', type=str, default='2024-12-01',
#                       help='Target date for prediction (YYYY-MM-DD)')
#     args = parser.parse_args()

#     df = pd.read_csv('nvda_apr_oct_final.csv')
#     df = df.sort_values('Date').reset_index(drop=True)
    
#     nyse = mcal.get_calendar('NYSE')
#     current_date = pd.to_datetime(args.date)
#     schedule = nyse.schedule(start_date=current_date, end_date=current_date + pd.Timedelta(days=10))
#     if current_date not in schedule.index:
#         print(f"The current date {current_date} is not a trading day, using next trading day instead: {schedule.index[0]}")
#         current_date = schedule.index[0]
#     sequence = get_historic_data(df, current_date)
#     # print(schedule.index)
#     # print(sequence.index)

# if __name__ == "__main__":
#     main()
