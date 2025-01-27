# How to Use This Code in Google Colab

This guide provides a step-by-step process to execute the provided code in Google Colab efficiently, focusing on practical steps without delving into the code's internal explanations.

---

## Prerequisites
Before you begin, ensure the following:
1. You have a Google account to access Google Colab.
2. Your dataset file is available and ready for upload in `.xlsx` format.
3. Familiarity with Python basics and Google Colab interface is helpful but not required.

---

## Step-by-Step Instructions

### Step 1: Open Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Create a new notebook by clicking on **File > New Notebook**.

### Step 2: Install Required Libraries
1. Copy and paste the following command into a cell and run it to install necessary Python libraries:
   ```python
   !pip install openpyxl statsmodels --quiet
   !pip install --upgrade torch torchvision --quiet
   ```
2. Verify successful installation by ensuring no errors appear after running the cell.

### Step 3: Import Necessary Modules
1. Copy and paste the following code into a new cell and run it to import required Python libraries and modules:
   ```python
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
   ```

### Step 4: Upload Your Dataset
1. Use the following code to upload your `.xlsx` file:
   ```python
   uploaded = files.upload()
   file_name = list(uploaded.keys())[0]
   ```
2. After running the cell, a file upload dialog will appear. Select your dataset file.

### Step 5: Preprocess Your Data
1. Copy and paste the preprocessing code provided into a cell and run it.
2. Ensure no errors are encountered during execution, and confirm output messages to verify data preparation.

### Step 6: Split Data for Training and Testing
1. Use the provided function for temporal split:
   ```python
   train_df, test_df = temporal_product_split(df)
   ```
2. Verify the training and testing datasets are correctly prepared by inspecting their contents:
   ```python
   print(train_df.head())
   print(test_df.head())
   ```

### Step 7: Generate Features and Prepare Data
1. Execute the feature engineering code to create features:
   ```python
   train_df = create_physical_features(train_df)
   test_df = create_physical_features(test_df, train_ref=train_df)
   ```
2. Confirm the feature columns are added to the datasets using `train_df.head()`.

### Step 8: Prepare Data for Model Training
1. Run the sequence generation code:
   ```python
   X_train_num, X_train_cat, y_train, train_products = create_sequences(train_df)
   X_test_num, X_test_cat, y_test, test_products = create_sequences(test_df)
   ```
2. Convert data to PyTorch tensors:
   ```python
   X_train_num_tensor = torch.FloatTensor(X_train_num)
   X_train_cat_tensor = torch.LongTensor(X_train_cat)
   y_train_tensor = torch.FloatTensor(y_train)
   train_products_tensor = torch.LongTensor(train_products)

   X_test_num_tensor = torch.FloatTensor(X_test_num)
   X_test_cat_tensor = torch.LongTensor(X_test_cat)
   y_test_tensor = torch.FloatTensor(y_test)
   test_products_tensor = torch.LongTensor(test_products)
   ```

### Step 9: Configure and Train the Model
1. Initialize the model with the following code:
   ```python
   DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = InventoryTransformer(...).to(DEVICE)
   ```
2. Set up the optimizer, scheduler, and loss function, and execute the training loop provided in the code.
3. Monitor training progress by reviewing loss and metric outputs printed during execution.

### Step 10: Evaluate and Visualize Results
1. After training, load the best model and evaluate it on the test dataset:
   ```python
   model.load_state_dict(torch.load('best_model.pth'))
   model.eval()
   ```
2. Run the visualization and analysis code to plot training progress and validate predictions.

### Step 11: Save Your Work
1. Save important results, plots, and the model:
   ```python
   torch.save(model.state_dict(), 'final_model.pth')
   ```
2. Download the saved model to your local machine using:
   ```python
   files.download('final_model.pth')
   ```

---

## Troubleshooting
- **Installation Errors:** Ensure internet connectivity and retry installing the libraries.
- **Dataset Errors:** Check for correct file format and consistent data structure.
- **Runtime Errors:** Verify each code block runs without skipping prior steps.

For additional support, please email me on donny.landscape@gmail.com
