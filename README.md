# How to Use the LTSM Stock Prediction Code

## Introduction

This guide provides a step-by-step tutorial on how to use the LSTM-based stock prediction code. The code trains a Long Short-Term Memory (LSTM) model to predict stock inventory levels based on historical data and features. It is designed to be flexible and extensible for various time-series prediction tasks.

---

## Prerequisites

Ensure the following requirements are met before using the code:

1. **Python Environment**:
   - Python version 3.7+
2. **Libraries**:
   Install the required libraries by running:
   ```bash
   pip install openpyxl statsmodels pandas numpy torch matplotlib scikit-learn plotly tqdm
   ```
3. **Data**:
   Prepare a dataset in `.xlsx` format. The dataset must include the following columns:
   - `Date`: Timestamps for data points.
   - `Quantity in Stock (liters/kg)`: Target variable.
   - Other categorical or numerical features to use for prediction.

---

## Code Overview

### Main Steps

1. **Data Preparation**:
   - Loads data from an Excel file.
   - Preprocesses the data by handling categorical variables, scaling features, and creating time-windowed sequences.
2. **Model Training**:
   - Trains an LSTM model with weight dropout to enhance robustness.
   - Uses a custom training loop with learning rate scheduling and mixed precision training.
3. **Evaluation**:
   - Evaluates the model using metrics like MAE, RMSE, SMAPE, and R².
   - Compares the model's performance against baseline methods.
4. **Visualization**:
   - Generates interactive dashboards for model insights.
5. **Hyperparameter Tuning**:
   - Performs random search across a predefined parameter grid.
   - Saves the best model and tuning results for reference.
6. **Production Inference**:
   - Implements a forecasting pipeline to generate multi-step predictions for specific product-location groups.

---

## How to Use

### Step 1: Upload Your Data

Ensure your data is prepared in `.xlsx` format. When prompted, upload your file during the script execution.

### Step 2: Run the Code

Follow these steps:

1. **Clone the Repository**:
   Clone or download the repository containing the code.

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Run the Script**:
   Open the script in an environment like Jupyter Notebook or Colab. If using Colab:

   - Upload the code file and run it step-by-step.
   - The following line will prompt you to upload your dataset:
     ```python
     uploaded = files.upload()
     ```

3. **Inspect Outputs**:
   After training, inspect the training/testing loss and metrics printed in the output.

### Step 3: Customize Hyperparameters

Modify the following parameters in the script as needed:

- **Window Size**: Adjust the `global_window_size` for sequence length.
- **LSTM Parameters**: Configure `hidden_size`, `num_layers`, and `dropout` in the `LSTMStockPredictor` class.
- **Learning Rate**: Adjust the learning rate in the optimizer setup.

### Step 4: Hyperparameter Tuning

Run the hyperparameter tuning function to optimize model performance:

```python
best_params, tuning_results = tune_hyperparameters(
    train_loader_tune,
    val_loader,
    INPUT_SIZE,
    DEVICE
)
print("Best parameters:", best_params)
```

The best parameters will be saved and used for final training.

### Step 5: Evaluate Model

The trained model will be saved as `best_model.pth`. Use the evaluation pipeline to generate metrics and visualize predictions.

### Step 6: Visualization

Run the visualization dashboard to interactively explore results:

```python
visualizer.create_interactive_dashboard()
```

### Step 7: Production Inference

Use the `StockForecaster` class to perform multi-step forecasting:

```python
forecaster = StockForecaster(
    model_path="best_tuned_model.pth",
    scalers_path="group_scalers.pkl",
    device=DEVICE
)

forecast = forecaster.predict(
    group_key="Organic Milk||Maharashtra",
    recent_data=sample_data,
    forecast_steps=14
)
```

Visualize results using Plotly or save forecasts for further analysis.

---

## Advanced Options

1. **Add New Features**:
   Add more relevant columns to your dataset to improve predictions. Ensure you update the preprocessing section to include the new features.

2. **Baseline Comparisons**:
   Use the provided `BaselineModels` class to compare the LSTM model with:

   - Persistence (last value prediction).
   - Moving Average.
   - Seasonal Baselines.

3. **Distributed Training**:
   The code supports multi-GPU training. Ensure PyTorch detects multiple GPUs:

   ```python
   model = nn.DataParallel(model)
   ```

---

## Troubleshooting

- **Insufficient Data**:

  - Ensure each product-location group has enough data to create sequences.
  - Increase the sequence window size only if sufficient data is available.

- **Slow Training**:

  - Use GPU acceleration. Verify the environment supports CUDA:
    ```python
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ```

- **Scaling Issues**:

  - Check the feature scaling logic in the preprocessing step. Ensure all numerical columns are properly scaled.

---
## Contact

Email: donny.landscape@gmail.com
Twitter: @donny_sant71053
