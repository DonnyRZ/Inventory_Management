# Step 1: Install required libraries
!pip install openpyxl statsmodels --quiet
!pip install --upgrade torch torchvision --quiet

# Step 2: Import modules
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from google.colab import files
import matplotlib.pyplot as plt
from tqdm import tqdm

# Verify PyTorch version
assert torch.__version__ >= '1.9.0', "Update PyTorch: !pip install torch==1.13.1"

# Step 3: Load and preprocess data
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_excel(file_name, sheet_name=0)

print("\n=== Enhanced Physics-Aware Processing ===")
# Temporal processing
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(['Product Name', 'Date'], inplace=True)

# Product-specific zero handling with physical constraints
target_col = 'Quantity in Stock (liters/kg)'
product_mins = df[df[target_col] > 0].groupby('Product Name')[target_col].min()
df[target_col] = df.groupby('Product Name')[target_col].transform(
    lambda x: x.replace(0, product_mins[x.name]).clip(lower=0)  # Physical constraint
)

# Temporal feature engineering
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_of_month'] = df['Date'].dt.day
df['days_since_first'] = df.groupby('Product Name')['Date'].transform(
    lambda x: (x - x.min()).dt.days
)

# Step 4: Temporal split with product preservation
def temporal_product_split(df, test_size=0.2):
    """Chronological split maintaining product distributions"""
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.groupby('Product Name', group_keys=False).apply(
        lambda x: x.iloc[:int(len(x)*0.8)]
    )
    test_df = df.drop(train_df.index)
    return train_df, test_df

train_df, test_df = temporal_product_split(df)

# Step 5: Leakage-proof feature engineering
def create_physical_features(df, train_ref=None):
    """Create features with physical inventory constraints"""
    df = df.copy()
    
    # Product-specific rolling features using training reference
    for product in df['Product Name'].unique():
        product_mask = df['Product Name'] == product
        if train_ref is not None:
            ref_data = train_ref[train_ref['Product Name'] == product]
            if len(ref_data) == 0:
                continue
            weekly_median = ref_data[target_col].rolling(7).median().median()
            monthly_max = ref_data[target_col].rolling(30).max().median()
        else:
            weekly_median = df.loc[product_mask, target_col].rolling(7).median().median()
            monthly_max = df.loc[product_mask, target_col].rolling(30).max().median()
        
        df.loc[product_mask, 'weekly_median'] = weekly_median
        df.loc[product_mask, 'monthly_max'] = monthly_max
    
    # Safe lag features
    lags = [3, 7, 14]
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('Product Name')[target_col].shift(lag)
        if train_ref is not None:
            fill_values = train_ref.groupby('Product Name')[target_col].median()
        else:
            fill_values = df.groupby('Product Name')[target_col].median()
        df[f'lag_{lag}'] = df.groupby('Product Name')[f'lag_{lag}'].transform(
            lambda x: x.fillna(method='ffill').fillna(fill_values[x.name])
        )
    
    return df

train_df = create_physical_features(train_df)
test_df = create_physical_features(test_df, train_ref=train_df)

# Step 6: Enhanced categorical encoding with unknown handling
cat_cols = ['Sales Channel', 'Farm Size', 'Brand', 'Storage Condition']
ordinal_encoders = {}
for col in cat_cols:
    # Handle unseen categories by encoding them as -1
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_df[col] = oe.fit_transform(train_df[[col]].astype(str))
    test_df[col] = oe.transform(test_df[[col]].astype(str))
    ordinal_encoders[col] = oe

# Step 7: Product-aware scaling with physical limits
scaler_dict = {}
for product in train_df['Product Name'].unique():
    product_data = train_df[train_df['Product Name'] == product]
    scaler = RobustScaler(quantile_range=(5, 95))
    scaler.fit(product_data[[target_col]])
    scaler_dict[product] = {
        'scaler': scaler,
        'max_capacity': product_data[target_col].max() * 1.2  # Physical constraint
    }

def product_scale(df):
    scaled = np.zeros(len(df))
    for product, data in scaler_dict.items():
        mask = df['Product Name'] == product
        if sum(mask) > 0:
            scaled[mask] = data['scaler'].transform(df.loc[mask, [target_col]]).flatten()
    return scaled

train_df['scaled_target'] = product_scale(train_df)
test_df['scaled_target'] = product_scale(test_df)

# Step 8: Physics-informed sequence generation
WINDOW_SIZE = 21  # Longer window for inventory cycles

class InventoryDataset(Dataset):
    def __init__(self, num_data, cat_data, targets, products):
        self.num_data = num_data
        self.cat_data = cat_data
        self.targets = targets
        self.products = products
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.num_data[idx],
            self.cat_data[idx],
            self.targets[idx],
            self.products[idx]
        )

def create_sequences(df):
    X_num, X_cat, y, products = [], [], [], []
    for product in df['Product Name'].unique():
        product_data = df[df['Product Name'] == product]
        if len(product_data) < WINDOW_SIZE + 1:
            continue
            
        features = [
            'days_since_first', 'day_of_week', 'day_of_month',
            'weekly_median', 'monthly_max', 'lag_3', 'lag_7', 'lag_14'
        ]
        num_features = product_data[features].values.astype(np.float32)
        cat_features = product_data[cat_cols].values.astype(np.int64)
        targets = product_data['scaled_target'].values
        
        for i in range(len(product_data) - WINDOW_SIZE):
            seq_num = num_features[i:i+WINDOW_SIZE]
            seq_cat = cat_features[i:i+WINDOW_SIZE]
            target = targets[i+WINDOW_SIZE]
            
            # Physical validity checks
            if np.isnan(seq_num).any() or np.isnan(target):
                continue
                
            X_num.append(seq_num)
            X_cat.append(seq_cat)
            y.append(target)
            products.append(product_data['product_id'].iloc[i])
    
    return np.array(X_num), np.array(X_cat), np.array(y), np.array(products)

# Product encoding with minimum sample threshold
product_counts = train_df['Product Name'].value_counts()
valid_products = product_counts[product_counts >= 50].index
train_df = train_df[train_df['Product Name'].isin(valid_products)]
test_df = test_df[test_df['Product Name'].isin(valid_products)]

product_encoder = {name: idx for idx, name in enumerate(valid_products)}
train_df['product_id'] = train_df['Product Name'].map(product_encoder)
test_df['product_id'] = test_df['Product Name'].map(product_encoder)

X_train_num, X_train_cat, y_train, train_products = create_sequences(train_df)
X_test_num, X_test_cat, y_test, test_products = create_sequences(test_df)

# Convert to tensors
X_train_num_tensor = torch.FloatTensor(X_train_num)
X_train_cat_tensor = torch.LongTensor(X_train_cat)
y_train_tensor = torch.FloatTensor(y_train)
train_products_tensor = torch.LongTensor(train_products)

X_test_num_tensor = torch.FloatTensor(X_test_num)
X_test_cat_tensor = torch.LongTensor(X_test_cat)
y_test_tensor = torch.FloatTensor(y_test)
test_products_tensor = torch.LongTensor(test_products)

# Step 9: Enhanced Physics-Aware Model
class InventoryTransformer(nn.Module):
    def __init__(self, num_numerical, cat_sizes, num_products):
        super().__init__()
        self.product_emb = nn.Embedding(num_products, 16)
        self.cat_embs = nn.ModuleList([
            nn.Embedding(size, 8) for size in cat_sizes.values()
        ])
        
        self.input_proj = nn.Linear(num_numerical + 16 + 8*len(cat_sizes), 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024,
                dropout=0.2, activation='gelu'
            ),
            num_layers=4
        )
        self.temporal_attn = nn.MultiheadAttention(256, 8, dropout=0.2)
        
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Softplus()  # Physical non-negative output
        )
        
        # Physical capacity constraints
        self.max_capacities = torch.tensor([
            scaler_dict[product]['max_capacity'] for product in valid_products
        ], dtype=torch.float32)
        
    def forward(self, x_num, x_cat, product_ids):
        batch_size, seq_len, _ = x_num.size()
        
        # Product embeddings
        product_emb = self.product_emb(product_ids).unsqueeze(1)
        
        # Categorical embeddings
        cat_embs = []
        for i, emb_layer in enumerate(self.cat_embs):
            cat_embs.append(emb_layer(x_cat[:, :, i]))
        cat_embs = torch.cat(cat_embs, dim=-1)
        
        # Combine features
        x = torch.cat([x_num, product_emb.expand(-1, seq_len, -1), cat_embs], dim=-1)
        x = self.input_proj(x)
        
        # Transformer processing
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        x = self.transformer(x)
        x, _ = self.temporal_attn(x, x, x)
        x = x[-1]  # Take last time step
        
        # Prediction with capacity constraints
        pred = self.regressor(x).squeeze()
        capacities = self.max_capacities.to(pred.device)[product_ids]
        return torch.minimum(pred, capacities)

# Step 10: Optimized Loss Function
class InventoryLoss(nn.Module):
    def __init__(self, overstock_weight=1.5, stockout_weight=4.0):
        super().__init__()
        self.over = overstock_weight
        self.stockout = stockout_weight
        
    def forward(self, preds, targets):
        errors = preds - targets
        return torch.mean(torch.where(
            errors > 0,
            self.over * torch.abs(errors),
            self.stockout * torch.square(errors)
        )) + 0.1 * torch.mean(torch.abs(errors))  # Hybrid loss

# Step 11: Enhanced Training Configuration
BATCH_SIZE = 128
train_dataset = InventoryDataset(X_train_num_tensor, X_train_cat_tensor, y_train_tensor, train_products_tensor)
test_dataset = InventoryDataset(X_test_num_tensor, X_test_cat_tensor, y_test_tensor, test_products_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)  # Allow partial last batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cat_sizes = {col: len(ordinal_encoders[col].categories_[0]) + 1 for col in cat_cols}  # +1 for unknown
model = InventoryTransformer(
    num_numerical=X_train_num.shape[-1],
    cat_sizes=cat_sizes,
    num_products=len(product_encoder)
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4,
    steps_per_epoch=len(train_loader), epochs=50
)
criterion = InventoryLoss(overstock_weight=1.5, stockout_weight=4.0)
grad_scaler = torch.cuda.amp.GradScaler()

# Step 12: Improved Training Loop
def calculate_smape(true, pred):
    denominator = (np.abs(true) + np.abs(pred)) / 2 + 1e-8
    return 200 * np.mean(np.abs(pred - true) / denominator)

if __name__ == '__main__':
    best_mae = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'mae': [], 'smape': []}
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        
        for num_x, cat_x, targets, products in tqdm(train_loader):
            num_x = num_x.to(DEVICE)
            cat_x = cat_x.to(DEVICE)
            targets = targets.to(DEVICE)
            products = products.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(num_x, cat_x, products)
                loss = criterion(preds, targets)
            
            grad_scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for num_x, cat_x, targets, products in test_loader:
                num_x = num_x.to(DEVICE)
                cat_x = cat_x.to(DEVICE)
                targets = targets.to(DEVICE)
                products = products.to(DEVICE)
                
                preds = model(num_x, cat_x, products)
                val_loss += criterion(preds, targets).item()
                all_preds.append(preds.cpu().numpy().flatten())  # Ensure 1D
                all_targets.append(targets.cpu().numpy().flatten())
        
        # Metrics calculation
        avg_val_loss = val_loss / len(test_loader)
        preds = np.concatenate(all_preds) if all_preds else np.array([])
        targets = np.concatenate(all_targets) if all_targets else np.array([])
        
        # Handle empty predictions
        if len(preds) == 0 or len(targets) == 0:
            print("No predictions to evaluate")
            continue
        
        # Inverse scaling with capacity constraints
        preds_inv = np.zeros_like(preds)
        targets_inv = np.zeros_like(targets)
        for product in valid_products:
            product_mask = test_df['Product Name'].iloc[:len(preds)] == product
            if product_mask.sum() > 0:
                scaler = scaler_dict[product]['scaler']
                preds_inv[product_mask] = scaler.inverse_transform(preds[product_mask].reshape(-1, 1)).flatten()
                targets_inv[product_mask] = scaler.inverse_transform(targets[product_mask].reshape(-1, 1)).flatten()
                
                # Apply physical maximum capacity
                max_cap = scaler_dict[product]['max_capacity']
                preds_inv[product_mask] = np.clip(preds_inv[product_mask], 0, max_cap)
        
        mae = mean_absolute_error(targets_inv, preds_inv)
        smape = calculate_smape(targets_inv, preds_inv)
        
        print(f"Epoch {epoch+1}:")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"MAE: {mae:.2f} | SMAPE: {smape:.2f}%")
        
        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), 'best_model.pth')
            print("✅ New best model saved")
        
        history['train_loss'].append(epoch_loss/len(train_loader))
        history['val_loss'].append(avg_val_loss)
        history['mae'].append(mae)
        history['smape'].append(smape)
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Plotting
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Training Progress")
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(history['mae'], label='MAE')
    plt.plot(history['smape'], label='SMAPE')
    plt.title("Validation Metrics")
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.scatter(targets_inv, preds_inv, alpha=0.3)
    plt.plot([0, targets_inv.max()], [0, targets_inv.max()], 'r--')
    plt.xlabel("Actual Stock")
    plt.ylabel("Predicted Stock")
    plt.title(f"Final Predictions\nMAE: {mae:.2f} | SMAPE: {smape:.2f}%")
    plt.tight_layout()
    plt.show()

    # Error analysis
    error_df = pd.DataFrame({
        'Product': test_df['Product Name'].iloc[:len(preds_inv)],
        'Actual': targets_inv,
        'Predicted': preds_inv,
        'Error': preds_inv - targets_inv
    })
    
    print("\nEnhanced Error Analysis:")
    print(error_df.groupby('Product').agg({
        'Error': ['mean', 'std', 'median', lambda x: np.percentile(x, 95)]
    }))
