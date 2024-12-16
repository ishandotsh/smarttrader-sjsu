import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
import yaml

class StockDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length, :5]
        return x, y

class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        # Use the last sequence element for prediction
        x = x[:, -1, :]
        x = self.decoder(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.train()
    total_train_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_val_loss += loss.item()
    
    return (total_train_loss / len(train_loader), 
            total_val_loss / len(val_loader))

def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    df = pd.read_csv('data/proc/nvda_feb_dec.csv')
    df = df.sort_values('Date').reset_index(drop=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
        'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Volume', 
        'MA5_Close', 'MA10_Close', 'MA5_Volume', 'MA10_Volume', 'MA5_Open', 
        'MA10_Open', 'MA5_High', 'MA10_High', 'MA5_Low', 'MA10_Low',
        'Volatility_Close', 'Volatility_Volume', 'Volatility_Open', 'Volatility_High', 'Volatility_Low',
        'MACD_Close', 'MACD_Volume', 'MACD_Open', 'MACD_High', 'MACD_Low',
        'RSI_14_Close', 'RSI_14_Volume', 'RSI_14_Open', 'RSI_14_High', 'RSI_14_Low']
    
    data = df[features].values
    
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # print("train_data shape: ", train_data.shape)
    # print("val_data shape: ", val_data.shape)
    # print("last date of train_data: ", df['Date'].iloc[train_size-1])
    
    scaler = MinMaxScaler()
    train_data_normalized = scaler.fit_transform(train_data)
    val_data_normalized = scaler.transform(val_data)

    train_dataset = StockDataset(train_data_normalized, seq_length=config['SEQ_LENGTH'])
    val_dataset = StockDataset(val_data_normalized, seq_length=config['SEQ_LENGTH'])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(
        input_dim=len(features),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'saved_models/{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    num_epochs = 100
    best_val_loss = float('inf')
    final_model_path = ""
    for epoch in range(num_epochs):
        train_loss, val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer, device
        )
        # if (epoch) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], '
        #           f'Train Loss: {train_loss}, '
        #           f'Val Loss: {val_loss}')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model_{epoch+1}.pth')
            final_model_path = f'{save_dir}/best_model_{epoch+1}.pth'
    print("final_model_path: ", final_model_path)

if __name__ == "__main__":
    main()
