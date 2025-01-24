# Step-by-Step Guide for Using the LTSM Project in Google Colab

This document provides a detailed guide on how to use the provided LTSM project code in Google Colab for the Hiring Team.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Uploading the Code to Colab](#uploading-the-code-to-colab)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Uploading and Preprocessing the Dataset](#uploading-and-preprocessing-the-dataset)
5. [Training and Testing the Model](#training-and-testing-the-model)
6. [Visualizing Results](#visualizing-results)

---

### 1. Prerequisites

Before you start, ensure the following:
- You have a Google account.
- Your dataset is available in `.xlsx` format.
- Basic familiarity with Google Colab and Python is recommended.

### 2. Uploading the Code to Colab

1. Download the provided `ltsm_project.py` file to your local machine.
2. Open [Google Colab](https://colab.research.google.com/).
3. Create a new notebook or open an existing one.
4. Upload the `ltsm_project.py` file by selecting `Files` (on the left sidebar), then clicking the upload button. Alternatively, use the following code to upload:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
5. After uploading, verify that the file is visible in the file explorer.

### 3. Setting Up the Environment

1. Install the required dependencies:
   ```python
   !pip install openpyxl statsmodels plotly tqdm torch torchvision
   ```
2. Import the necessary modules by running the following code:
   ```python
   import pandas as pd
   import numpy as np
   import torch
   import torch.nn as nn
   from torch.utils.data import Dataset, DataLoader
   from sklearn.preprocessing import StandardScaler
   ```
3. Run the `ltsm_project.py` file in your Colab notebook by using:
   ```python
   %run ltsm_project.py
   ```

### 4. Uploading and Preprocessing the Dataset

1. Ensure your dataset contains columns like `Date`, `Sales Channel`, `Farm Size`, `Brand`, and other features mentioned in the code.
2. Upload your dataset using the following code snippet:
   ```python
   from google.colab import files
   uploaded = files.upload()
   file_name = list(uploaded.keys())[0]
   df = pd.read_excel(file_name, sheet_name=0)
   ```
3. The preprocessing steps (such as one-hot encoding and scaling) are automatically handled by the code.
   - Verify that your dataset meets the column requirements mentioned in the code.

### 5. Training and Testing the Model

1. The code automatically splits the data into training and testing sets based on an 80-20 split.
2. Define your batch size, number of epochs, and other configurations directly in the code.
3. To train the model, run the `train_model` function:
   ```python
   history = train_model(model, train_loader, test_loader, num_epochs=200)
   ```
4. Monitor the training progress using the `tqdm` progress bar displayed in the output.
5. The best-performing model weights are saved as `best_model.pth`.

### 6. Visualizing Results

1. The project includes a `StockVisualizer` class to analyze and visualize results.
2. Initialize the visualizer as follows:
   ```python
   visualizer = StockVisualizer(
       model=model,
       train_loader=train_loader,
       test_loader=test_loader,
       scalers=scalers,  # Scalers defined during preprocessing
       train_series=y_train_full,  # Target values from training data
       dates=df['Date'].values  # Date column
   )
   ```
3. Launch the interactive dashboard to visualize:
   ```python
   visualizer.create_interactive_dashboard()
   ```
4. Use the dashboard to explore metrics, residuals, predictions, and more.

---

### Additional Notes

- If you encounter any errors related to dataset formatting, ensure that your columns align with the script's preprocessing steps.
- You can modify hyperparameters such as the learning rate, dropout rate, and number of layers directly in the `LSTMStockPredictor` class.
- The dashboard requires the `plotly` library; ensure it is installed using the steps in [Section 3](#setting-up-the-environment).

---

### Contact for Support
For any issues or questions, reach out to the project owner or the Hiring Team's technical lead.

