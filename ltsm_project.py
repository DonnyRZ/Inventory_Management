## Step 1: Install required libraries
!pip install openpyxl
!pip install statsmodels

## Step 2: Import modules
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from google.colab import files
import plotly.express as px
import plotly.io as pio
import torch.nn.utils.parametrize as parametrize
from pathlib import Path
import pickle
from datetime import datetime

## Step 3: Load data
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_excel(file_name, sheet_name=0)

## Step 4: Global preprocessing
## Convert date and sort
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

## One-hot encode categorical variables globally
cat_cols = ['Sales Channel', 'Farm Size', 'Brand', 'Storage Condition']
df_encoded = pd.get_dummies(df, columns=cat_cols)

## Define target and remove leaky/non-predictive columns
target_col = 'Quantity in Stock (liters/kg)'
leaky_cols = ['Quantity Sold (liters/kg)', 'Approx. Total Revenue(INR)', 'Price per Unit (sold)']
columns_to_drop = [target_col, 'Date', 'Customer Location', 'Product Name', 'Location'] + leaky_cols
features_all = df_encoded.drop(columns=columns_to_drop)
target_all = df_encoded[target_col]

## Global feature selection using variance threshold
selector = VarianceThreshold(threshold=0.1)  # Adjust threshold as needed
features_selected = selector.fit_transform(features_all)
selected_features = features_all.columns[selector.get_support()]
print(f"Global selected features: {len(selected_features)}")

## Step 5: Determine global window size using ACF
acf_values = acf(target_all, nlags=30)
global_window_size = np.argmax(acf_values < 0.2)  # First lag where ACF < 0.2
global_window_size = max(7, min(global_window_size, 30))  # Limit between 7-30
print(f"Global window size: {global_window_size}")

## Step 6: Grouped processing (FIXED)
all_sequences_train = []  # NEW: Separate training data
all_sequences_test = []   # NEW: Separate test data
MIN_TRAIN_SAMPLES = global_window_size + 1

for (product, location), group in df_encoded.groupby(['Product Name', 'Location']):
    ## Sort by date and filter features
    group = group.sort_values('Date')
    group_features = group[selected_features]
    group_target = group[target_col]

    ## Time-based split (80-20)
    split_idx = int(len(group) * 0.8)
    train_feats, test_feats = group_features.iloc[:split_idx], group_features.iloc[split_idx:]
    train_targs, test_targs = group_target.iloc[:split_idx], group_target.iloc[split_idx:]

    ## Skip groups with insufficient training data
    if len(train_feats) < MIN_TRAIN_SAMPLES:
        print(f"Skipping {product} in {location} - insufficient training data")
        continue

    ## Step 7: Scaling
    feat_scaler = StandardScaler()
    targ_scaler = StandardScaler()

    X_train_scaled = feat_scaler.fit_transform(train_feats)
    X_test_scaled = feat_scaler.transform(test_feats)
    y_train_scaled = targ_scaler.fit_transform(train_targs.values.reshape(-1, 1))
    y_test_scaled = targ_scaler.transform(test_targs.values.reshape(-1, 1))

    ## Step 8: Create sequences
    def create_sequences(feats, targs, window_size):
        X, y = [], []
        for i in range(len(feats) - window_size):
            X.append(feats[i:i+window_size])
            y.append(targs[i+window_size])
        return np.array(X), np.array(y)

    ## Training sequences (added to train list)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, global_window_size)
    if len(X_train_seq) > 0:
        all_sequences_train.append((X_train_seq, y_train_seq))

    ## Testing sequences (added to test list)
    X_test_seq, y_test_seq = [], []
    if len(test_feats) >= global_window_size:
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, global_window_size)
    if len(X_test_seq) > 0:
        all_sequences_test.append((X_test_seq, y_test_seq))

## Step 9: Combine separately (FIXED)
if not all_sequences_train:
    raise ValueError("No valid training sequences created.")

X_train_final = np.concatenate([seq[0] for seq in all_sequences_train])
y_train_final = np.concatenate([seq[1] for seq in all_sequences_train])
X_test_final = np.concatenate([seq[0] for seq in all_sequences_test]) if all_sequences_test else np.array([])
y_test_final = np.concatenate([seq[1] for seq in all_sequences_test]) if all_sequences_test else np.array([])

## Step 10: Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_final)
y_train_tensor = torch.FloatTensor(y_train_final)
X_test_tensor = torch.FloatTensor(X_test_final) if len(X_test_final) > 0 else None
y_test_tensor = torch.FloatTensor(y_test_final) if len(y_test_final) > 0 else None

## Step 11: DataLoader (FIXED)
class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

BATCH_SIZE = 32
train_dataset = StockDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = StockDataset(X_test_tensor, y_test_tensor) if X_test_tensor is not None else None
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if test_dataset else None

## Step 12: Final checks
print("\nProcessing complete!")
print(f"Training sequences: {len(train_dataset)}")
print(f"Test sequences: {len(test_dataset) if test_dataset else 0}")
print(f"Timesteps per sequence: {X_train_final.shape[1]}")
print(f"Features per timestep: {X_train_final.shape[2]}")
print(f"Example batch shape: {next(iter(train_loader))[0].shape}")

# Step 13: Correct Weight-Dropped LSTM Implementation using Parametrization
class WeightDropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.0:
            return nn.functional.dropout(x, p=self.p, training=True) * (1 - self.p)
        return x

class WeightDropLSTM(nn.LSTM):
    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dropout = weight_dropout
        for i in range(self.num_layers):
            wname = f'weight_hh_l{i}'
            parametrize.register_parametrization(self, wname, WeightDropout(weight_dropout))

class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = WeightDropLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            weight_dropout=dropout*0.5
        )

        self.linear = nn.Linear(hidden_size, 1)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='linear')

    def forward(self, x, hidden=None):
        x = self.input_norm(x)
        lstm_out, hidden = self.lstm(x, hidden)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions, hidden

# Step 14: Training Configuration
INPUT_SIZE = X_train_final.shape[2]  # Replace with actual input size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

model = LSTMStockPredictor(
    input_size=INPUT_SIZE,
    hidden_size=256 if torch.cuda.device_count() > 1 else 128,
    num_layers=2,
    dropout=0.4 if torch.cuda.device_count() > 1 else 0.3
)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-3,
    steps_per_epoch=len(train_loader),
    epochs=200,
    pct_start=0.3
)
criterion = nn.MSELoss()
GRAD_CLIP = 0.5
scaler = torch.cuda.amp.GradScaler()

# Step 15: Correct Training Loop
def train_model(model, train_loader, test_loader, num_epochs=200):
    best_loss = float('inf')
    history = {'train_loss': [], 'test_loss': [], 'mae': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for features, targets in progress_bar:
            features = features.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predictions, _ = model(features)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        # Validation
        if test_loader:
            model.eval()
            epoch_test_loss = 0
            total_mae = 0
            with torch.no_grad(), torch.cuda.amp.autocast():
                for features, targets in test_loader:
                    features = features.to(DEVICE)
                    targets = targets.to(DEVICE)
                    predictions, _ = model(features)
                    loss = criterion(predictions, targets)
                    epoch_test_loss += loss.item()
                    total_mae += torch.mean(torch.abs(predictions - targets)).item()

            avg_test_loss = epoch_test_loss / len(test_loader)
            avg_mae = total_mae / len(test_loader)
            history['test_loss'].append(avg_test_loss)
            history['mae'].append(avg_mae)

            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                save_model = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(save_model.state_dict(), 'best_model.pth')

        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | MAE: {avg_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    if test_loader and Path('best_model.pth').exists():
        model.load_state_dict(torch.load('best_model.pth'))
    return history

# Step 17: Enhanced Data Loading
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    persistent_workers=True,
    num_workers=2
) if test_dataset else None

# Step 18: Enhanced Evaluation Metrics Class
class TSEMetrics:
    def __init__(self, y_true, y_pred, epsilon=1e-8):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.epsilon = epsilon

    def calculate_all(self):
        return {
            'mae': self.mae(),
            'rmse': self.rmse(),
            'smape': self.smape(),
            'r2': self.r2(),
            'mase': None  # Modified to avoid error if not calculated
        }

    def mae(self):
        return np.mean(np.abs(self.y_true - self.y_pred))

    def rmse(self):
        return np.sqrt(np.mean((self.y_true - self.y_pred)**2))

    def smape(self):
        denominator = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2
        return 100 * np.mean(np.abs(self.y_true - self.y_pred) / (denominator + self.epsilon))

    def r2(self):
        ss_res = np.sum((self.y_true - self.y_pred)**2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true))**2)
        return 1 - (ss_res / (ss_tot + self.epsilon))

    def mase(self, train_y=None, seasonality=1):
        if train_y is None:
            return None  # Return None instead of raising error
        naive_error = np.mean(np.abs(train_y[seasonality:] - train_y[:-seasonality]))
        if naive_error == 0:
            return float('inf')
        return np.mean(np.abs(self.y_true - self.y_pred)) / naive_error

# Step 19: Robust Baseline Models
class BaselineModels:
    def __init__(self, X_test, y_test, target_series):
        self.X_test = X_test  # Historical features including target lags
        self.y_test = y_test
        self.target_series = target_series  # Full target series for baselines

    def calculate_all_baselines(self):
        return {
            'persistence': self.persistence(),
            'moving_avg_3': self.moving_average(window=3),
            'seasonal_7': self.seasonal(seasonality=7)
        }

    def persistence(self):
        return self.X_test[:, -1, -1]  # Assuming last feature is the target lag

    def moving_average(self, window=3):
        return np.mean(self.X_test[:, -window:, -1], axis=1)

    def seasonal(self, seasonality=7):
        return self.target_series[-len(self.y_test)-seasonality:-seasonality]

# Step 20: Enhanced Evaluation Pipeline
def evaluate_model(model, test_loader, scalers, train_series=None):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for features, targets in test_loader:
            features = features.to(DEVICE, non_blocking=True)
            preds = model(features)[0].cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    preds = scalers['target'].inverse_transform(np.concatenate(all_preds).reshape(-1, 1)).flatten()
    targets = scalers['target'].inverse_transform(np.concatenate(all_targets).reshape(-1, 1)).flatten()

    metrics = TSEMetrics(targets, preds).calculate_all()

    if train_series is not None:
        scaled_train = scalers['target'].transform(train_series.reshape(-1, 1)).flatten()
        metrics['mase'] = TSEMetrics(targets, preds).mase(scaled_train)

    baselines = BaselineModels(
        np.concatenate([batch[0].numpy() for batch in test_loader]),
        targets,
        scalers['target'].inverse_transform(train_series.reshape(-1, 1)).flatten() if train_series is not None else None
    ).calculate_all_baselines()

    baseline_metrics = {
        name: TSEMetrics(targets, baseline_preds).calculate_all()
        for name, baseline_preds in baselines.items()
    }

    return metrics, baseline_metrics, preds, targets

# Visualization Requirements
!pip install plotly pandas ipywidgets
!pip install tqdm  # Install tqdm

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider, IntSlider
from IPython.display import display
from tqdm import tqdm  # Import tqdm

class StockVisualizer:
    def __init__(self, model, train_loader, test_loader, scalers, train_series, dates):
        self.model = model
        self.scalers = scalers
        self.train_series = train_series
        self.dates = pd.to_datetime(dates)

        # Get predictions and targets
        self.metrics, self.baseline_metrics, self.preds, self.targets = evaluate_model(
            model, test_loader, scalers, train_series
        )

        # Prepare full timeline data
        self.full_df = self._prepare_timeline_data()

    def _prepare_timeline_data(self):
        # Inverse scale training data
        train_values = self.scalers['target'].inverse_transform(
            self.train_series.reshape(-1, 1)
        ).flatten()

        # Create DataFrame with all data
        test_size = len(self.targets)
        return pd.DataFrame({
            'date': self.dates,
            'actual': np.concatenate([train_values[:-test_size], self.targets]),
            'predicted': np.concatenate([np.full(len(train_values)-test_size, np.nan), self.preds])
        })

    def create_interactive_dashboard(self):
        self._plot_loss_curves()
        self._plot_predictions_comparison()
        self._plot_metrics_radar()
        self._plot_residual_analysis()

    def _plot_loss_curves(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(history['train_loss']))),
            y=history['train_loss'],
            name='Training Loss',
            line=dict(color='royalblue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(history['test_loss']))),
            y=history['test_loss'],
            name='Validation Loss',
            line=dict(color='firebrick')
        ))
        fig.update_layout(
            title='Training & Validation Loss',
            xaxis_title='Epoch',
            yaxis_title='MSE Loss',
            hovermode='x unified',
            template='plotly_dark'
        )
        fig.show()

    def _plot_predictions_comparison(self):
        @interact
        def plot_time_window(
            lookback=IntSlider(value=90, min=30, max=365, step=30, description='Lookback (days):'),
            forecast=IntSlider(value=30, min=7, max=90, step=7, description='Forecast window:')
        ):
            subset = self.full_df[-lookback-forecast:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=subset['date'], y=subset['actual'],
                name='Actual Price', line=dict(color='lightgrey')
            ))
            fig.add_trace(go.Scatter(
                x=subset['date'], y=subset['predicted'],
                name='Predicted Price', line=dict(color='#FF6F00'),
                opacity=0.8
            ))
            fig.update_layout(
                title=f'Price Predictions (Last {lookback} Days vs Forecast)',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x',
                template='plotly_dark',
                showlegend=True
            )
            fig.add_vline(
                x=subset['date'].iloc[-forecast],
                line_dash="dash", line_color="green"
            )
            fig.show()

    def _plot_metrics_radar(self):
        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'SMAPE', 'R²'],
            'Model': [
                self.metrics['mae'],
                self.metrics['rmse'],
                self.metrics['smape'],
                self.metrics['r2']
            ],
            'Baseline': [
                self.baseline_metrics['persistence']['mae'],
                self.baseline_metrics['persistence']['rmse'],
                self.baseline_metrics['persistence']['smape'],
                self.baseline_metrics['persistence']['r2']
            ]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=metrics_df['Model'],
            theta=metrics_df['Metric'],
            fill='toself',
            name='LSTM Model',
            line_color='#FF6F00'
        ))
        fig.add_trace(go.Scatterpolar(
            r=metrics_df['Baseline'],
            theta=metrics_df['Metric'],
            fill='toself',
            name='Baseline (Persistence)',
            line_color='lightgrey'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title='Model Performance vs Baseline',
            template='plotly_dark',
            showlegend=True
        )
        fig.show()

    def _plot_residual_analysis(self):
        residuals = self.targets - self.preds
        fig = px.scatter(
            x=self.preds, y=residuals,
            trendline="lowess",
            labels={'x': 'Predicted Values', 'y': 'Residuals'},
            title='Residual Analysis',
            color=abs(residuals),
            color_continuous_scale='viridis'
        )
        fig.update_layout(template='plotly_dark')
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.show()

# Usage Example
# Assuming you have these variables from previous steps:
# - model: Trained model
# - train_loader, test_loader: Data loaders
# - scalers: Feature scalers
# - train_series: Original training series
# - dates: Corresponding datetime index for entire series
# - history: Training history object

scalers = {'target': targ_scaler}
y_train_full = df[target_col].values  # Assuming 'target_col' is defined earlier
full_dates = df['Date'].values       # Assuming 'Date' is your date column
history = train_model(model, train_loader, test_loader)

# Initialize visualizer
visualizer = StockVisualizer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    scalers=scalers,  # Now 'scalers' is defined
    train_series=y_train_full,
    dates=full_dates
)

# Launch dashboard
visualizer.create_interactive_dashboard()
