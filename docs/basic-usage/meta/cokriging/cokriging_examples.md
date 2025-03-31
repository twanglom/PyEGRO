# PyEGRO Co-Kriging Examples

This document provides practical examples of using the PyEGRO Co-Kriging module for multi-fidelity modeling tasks.

## Table of Contents
- [Introduction to Co-Kriging](#introduction-to-co-kriging)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Working with Synthetic Data](#working-with-synthetic-data)
- [Working with CSV Data](#working-with-csv-data)
- [2D Examples](#2d-examples)
- [Visualization Examples](#visualization-examples)
- [Model Loading and Prediction](#model-loading-and-prediction)
- [Advanced Usage](#advanced-usage)

## Introduction to Co-Kriging

Co-Kriging is a multi-fidelity modeling approach that combines data from different fidelity levels, typically:
- **Low-fidelity data**: Larger dataset that is cheaper to obtain but less accurate
- **High-fidelity data**: Smaller dataset that is more expensive to obtain but more accurate

The Kennedy & O'Hagan (2000) approach used in this module creates a statistical relationship between the fidelity levels, allowing for more accurate predictions with fewer high-fidelity samples than would be needed with standard Gaussian Process Regression.

## Installation

The PyEGRO Co-Kriging module depends on the following packages:
- numpy
- pandas
- torch
- gpytorch
- scikit-learn
- matplotlib
- joblib
- rich (for enhanced progress displays)

Ensure these dependencies are installed before using the module:

```bash
pip install numpy pandas torch gpytorch scikit-learn matplotlib joblib rich
```

## Basic Usage

Here's a minimal example of training a Co-Kriging model with the PyEGRO module:

```python
import numpy as np
from PyEGRO.meta.cokriging import MetaTrainingCoKriging

# Define high and low fidelity functions (for demonstration)
def high_fidelity_function(x):
    return np.sin(8 * x) + 0.2 * x

def low_fidelity_function(x):
    return 0.5 * np.sin(8 * x) + 0.15 * x + 0.5

# Generate synthetic data
n_high = 15  # Fewer high-fidelity points
n_low = 50   # More low-fidelity points

X_high = np.random.uniform(0, 1, n_high).reshape(-1, 1)
y_high = high_fidelity_function(X_high) + 0.05 * np.random.randn(n_high, 1)

X_low = np.random.uniform(0, 1, n_low).reshape(-1, 1)
y_low = low_fidelity_function(X_low) + 0.1 * np.random.randn(n_low, 1)

# Initialize meta-training for Co-Kriging
meta = MetaTrainingCoKriging(
    num_iterations=300,
    prefer_gpu=True,
    kernel='matern25'
)

# Train the model
model, scaler_X, scaler_y = meta.train(
    X_low=X_low, 
    y_low=y_low, 
    X_high=X_high, 
    y_high=y_high
)

# Make predictions
X_new = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred_high, y_std_high = meta.predict(X_new, fidelity='high')
y_pred_low, y_std_low = meta.predict(X_new, fidelity='low')

print("Co-Kriging Model trained successfully")
```

## Working with Synthetic Data

The following example demonstrates how to use the Co-Kriging module with synthetic data and visualize the results:

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.cokriging import MetaTrainingCoKriging
from PyEGRO.meta.cokriging.visualization import visualize_cokriging

# Set random seed for reproducibility
np.random.seed(42)

# Define high and low fidelity functions
def high_fidelity_function(x):
    return (6*x - 2)**2 * np.sin(12*x - 4)

def low_fidelity_function(x):
    return 0.5 * high_fidelity_function(x) + 10 * (x - 0.5) - 5

# Generate synthetic data
# Low fidelity: more samples but less accurate
n_low = 80
X_low = np.random.uniform(0, 1, n_low).reshape(-1, 1)
y_low = low_fidelity_function(X_low) + np.random.normal(0, 1.0, X_low.shape)  # More noise

# High fidelity: fewer samples but more accurate
n_high = 20
X_high = np.random.uniform(0, 1, n_high).reshape(-1, 1)
y_high = high_fidelity_function(X_high) + np.random.normal(0, 0.5, X_high.shape)  # Less noise

# Test data
n_test = 40
X_test = np.linspace(0, 1, n_test).reshape(-1, 1)
y_test = high_fidelity_function(X_test) + np.random.normal(0, 0.3, X_test.shape)  # Even less noise

# Define bounds and variable names
bounds = np.array([[0, 1]])
variable_names = ['x']

# Initialize and train Co-Kriging model
print("Training Co-Kriging model with synthetic data...")
meta = MetaTrainingCoKriging(
    num_iterations=300,
    prefer_gpu=True,
    show_progress=True,
    output_dir='RESULT_MODEL_COKRIGING_SYNTHETIC'
)

# Train model with synthetic data
model, scaler_X, scaler_y = meta.train(
    X_low=X_low, 
    y_low=y_low, 
    X_high=X_high, 
    y_high=y_high,
    X_test=X_test,
    y_test=y_test,
    feature_names=variable_names
)

# Generate visualization
figures = visualize_cokriging(
    meta=meta,
    X_low=X_low,
    y_low=y_low,
    X_high=X_high,
    y_high=y_high,
    X_test=X_test,
    y_test=y_test,
    variable_names=variable_names,
    bounds=bounds,
    savefig=True
)

# Display figures
plt.show()
```

## Working with CSV Data

This example demonstrates how to use the Co-Kriging module with data stored in CSV files:

```python
import numpy as np
import pandas as pd
import json
import os
from PyEGRO.meta.cokriging import MetaTrainingCoKriging
from PyEGRO.meta.cokriging.visualization import visualize_cokriging

# Load initial data and problem configuration
with open('DATA_PREPARATION/data_info.json', 'r') as f:
    data_info = json.load(f)

# Load high and low fidelity training data
high_fidelity_data = pd.read_csv('DATA_PREPARATION/training_data_high.csv')
low_fidelity_data = pd.read_csv('DATA_PREPARATION/training_data_low.csv')

# Load testing data
test_data = pd.read_csv('DATA_PREPARATION/testing_data.csv')

# Get problem configuration
bounds = np.array(data_info['input_bound'])
variable_names = [var['name'] for var in data_info['variables']]

# Get target column name (default to 'y' if not specified)
target_column = data_info.get('target_column', 'y')

# Extract features and targets
X_high = high_fidelity_data.drop([target_column], axis=1, errors='ignore').values
y_high = high_fidelity_data[target_column].values.reshape(-1, 1)

X_low = low_fidelity_data.drop([target_column], axis=1, errors='ignore').values
y_low = low_fidelity_data[target_column].values.reshape(-1, 1)

# Extract testing data
X_test = test_data.drop([target_column], axis=1, errors='ignore').values
y_test = test_data[target_column].values.reshape(-1, 1)

# Initialize and train model
print("Training Co-Kriging model with CSV data...")
meta = MetaTrainingCoKriging(
    num_iterations=300,
    prefer_gpu=True,
    show_progress=True,
    output_dir='RESULT_MODEL_COKRIGING'
)

# Train model
model, scaler_X, scaler_y = meta.train(
    X_low=X_low, 
    y_low=y_low, 
    X_high=X_high, 
    y_high=y_high,
    X_test=X_test,
    y_test=y_test,
    feature_names=variable_names
)

# Generate visualization
figures = visualize_cokriging(
    meta=meta,
    X_low=X_low,
    y_low=y_low,
    X_high=X_high,
    y_high=y_high,
    X_test=X_test,
    y_test=y_test,
    variable_names=variable_names,
    bounds=bounds,
    savefig=True
)
```

## Custom Data Preparation

Example of preparing data files for Co-Kriging:

```python
import json
import numpy as np
import pandas as pd
import os

# Define fidelity functions
def high_fidelity_function(x1, x2):
    return np.sin(x1) * np.cos(x2) + 0.2 * x1 * x2

def low_fidelity_function(x1, x2):
    return 0.5 * high_fidelity_function(x1, x2) + 0.2 * x1 - 0.1 * x2 + 0.5

# Generate data
np.random.seed(42)

# High-fidelity data (fewer samples)
n_high = 30
X1_high = np.random.uniform(-2, 2, n_high)
X2_high = np.random.uniform(-2, 2, n_high)
y_high = high_fidelity_function(X1_high, X2_high) + np.random.normal(0, 0.05, n_high)

# Low-fidelity data (more samples)
n_low = 100
X1_low = np.random.uniform(-2, 2, n_low)
X2_low = np.random.uniform(-2, 2, n_low)
y_low = low_fidelity_function(X1_low, X2_low) + np.random.normal(0, 0.1, n_low)

# Test data
n_test = 50
X1_test = np.random.uniform(-2, 2, n_test)
X2_test = np.random.uniform(-2, 2, n_test)
y_test = high_fidelity_function(X1_test, X2_test) + np.random.normal(0, 0.05, n_test)

# Create DataFrames
high_fidelity_df = pd.DataFrame({
    'x1': X1_high,
    'x2': X2_high,
    'y': y_high
})

low_fidelity_df = pd.DataFrame({
    'x1': X1_low,
    'x2': X2_low,
    'y': y_low
})

test_df = pd.DataFrame({
    'x1': X1_test,
    'x2': X2_test,
    'y': y_test
})

# Create data_info.json
data_info = {
    "variables": [
        {"name": "x1", "type": "continuous"},
        {"name": "x2", "type": "continuous"}
    ],
    "input_bound": [
        [-2, 2],
        [-2, 2]
    ],
    "target_column": "y"
}

# Create directory
os.makedirs("DATA_PREPARATION", exist_ok=True)

# Save files
with open("DATA_PREPARATION/data_info.json", "w") as f:
    json.dump(data_info, f, indent=4)

high_fidelity_df.to_csv("DATA_PREPARATION/training_data_high.csv", index=False)
low_fidelity_df.to_csv("DATA_PREPARATION/training_data_low.csv", index=False)
test_df.to_csv("DATA_PREPARATION/testing_data.csv", index=False)

print("Data preparation complete for Co-Kriging.")
```

## 2D Examples

Example of using Co-Kriging with 2D inputs:

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.cokriging import MetaTrainingCoKriging
from PyEGRO.meta.cokriging.visualization import visualize_cokriging

# Define high and low fidelity 2D functions
def high_fidelity_function(x1, x2):
    return np.sin(x1) * np.cos(x2) + 0.2 * x1 * x2

def low_fidelity_function(x1, x2):
    return 0.5 * high_fidelity_function(x1, x2) + 0.2 * x1 - 0.1 * x2 + 0.5

# Generate synthetic data
# Low fidelity: more samples but less accurate
n_low = 100
X1_low = np.random.uniform(-2, 2, n_low)
X2_low = np.random.uniform(-2, 2, n_low)
X_low = np.column_stack([X1_low, X2_low])
y_low = low_fidelity_function(X1_low, X2_low).reshape(-1, 1) + np.random.normal(0, 0.1, (n_low, 1))

# High fidelity: fewer samples but more accurate
n_high = 25
X1_high = np.random.uniform(-2, 2, n_high)
X2_high = np.random.uniform(-2, 2, n_high)
X_high = np.column_stack([X1_high, X2_high])
y_high = high_fidelity_function(X1_high, X2_high).reshape(-1, 1) + np.random.normal(0, 0.05, (n_high, 1))

# Test data: grid for visualization
n_test = 16
X1_test = np.linspace(-2, 2, 4)
X2_test = np.linspace(-2, 2, 4)
X1_grid, X2_grid = np.meshgrid(X1_test, X2_test)
X1_test = X1_grid.flatten()
X2_test = X2_grid.flatten()
X_test = np.column_stack([X1_test, X2_test])
y_test = high_fidelity_function(X1_test, X2_test).reshape(-1, 1) + np.random.normal(0, 0.02, (n_test, 1))

# Define bounds and variable names
bounds = np.array([[-2, 2], [-2, 2]])
variable_names = ['x1', 'x2']

# Initialize and train Co-Kriging model
print("Training Co-Kriging model with 2D synthetic data...")
meta = MetaTrainingCoKriging(
    num_iterations=300,
    prefer_gpu=True,
    show_progress=True,
    output_dir='RESULT_MODEL_COKRIGING_2D'
)

# Train model with synthetic data
model, scaler_X, scaler_y = meta.train(
    X_low=X_low, 
    y_low=y_low, 
    X_high=X_high, 
    y_high=y_high,
    X_test=X_test,
    y_test=y_test,
    feature_names=variable_names
)

# Generate visualization
figures = visualize_cokriging(
    meta=meta,
    X_low=X_low,
    y_low=y_low,
    X_high=X_high,
    y_high=y_high,
    X_test=X_test,
    y_test=y_test,
    variable_names=variable_names,
    bounds=bounds,
    savefig=True
)

plt.show()
```

## Visualization Examples

The Co-Kriging module provides comprehensive visualization tools, accessible through the `visualize_cokriging` function:

```python
from PyEGRO.meta.cokriging.visualization import visualize_cokriging

# After training a model (continuing from previous examples)
figures = visualize_cokriging(
    meta=meta,
    X_low=X_low,
    y_low=y_low,
    X_high=X_high,
    y_high=y_high,
    X_test=X_test,
    y_test=y_test,
    variable_names=variable_names,
    bounds=bounds,
    savefig=True,
    output_dir='visualization_results'
)

# Access individual figures
actual_vs_predicted = figures['actual_vs_predicted']
r2_comparison = figures['r2_comparison']
response_surface = figures['response_surface']

# Customize and save a specific figure
import matplotlib.pyplot as plt
fig = figures['response_surface']
fig.suptitle('Custom Title for Response Surface')
fig.savefig('custom_response_surface.png', dpi=300)
```

## Model Loading and Prediction

Example of loading a trained model and making predictions:

```python
import numpy as np
from PyEGRO.meta.cokriging.cokriging_utils import DeviceAgnosticCoKriging

# Initialize device-agnostic loader
cokriging_loader = DeviceAgnosticCoKriging(prefer_gpu=True)

# Load the trained model
loaded = cokriging_loader.load_model(model_dir='RESULT_MODEL_COKRIGING')

if loaded:
    # Generate new input data for prediction
    X_new = np.random.rand(10, 2) * 4 - 2  # Values between -2 and 2
    
    # Make high-fidelity predictions
    y_pred_high, y_std_high = cokriging_loader.predict(X_new, fidelity='high')
    
    # Make low-fidelity predictions
    y_pred_low, y_std_low = cokriging_loader.predict(X_new, fidelity='low')
    
    # Print results
    print("High-fidelity predictions:")
    for i in range(len(X_new)):
        print(f"Input: {X_new[i]}, Prediction: {y_pred_high[i][0]:.4f} ± {y_std_high[i][0]:.4f}")
    
    print("\nLow-fidelity predictions:")
    for i in range(len(X_new)):
        print(f"Input: {X_new[i]}, Prediction: {y_pred_low[i][0]:.4f} ± {y_std_low[i][0]:.4f}")
else:
    print("Failed to load model")
```

Alternatively, use the `MetaTrainingCoKriging` class to load and use the model:

```python
from PyEGRO.meta.cokriging import MetaTrainingCoKriging

# Initialize meta
meta = MetaTrainingCoKriging()

# Load model
meta.load_model('RESULT_MODEL_COKRIGING/cokriging_model.pth')

# Make predictions
X_new = np.random.rand(10, 2) * 4 - 2  # Values between -2 and 2

# High-fidelity predictions
y_pred_high, y_std_high = meta.predict(X_new, fidelity='high')

# Low-fidelity predictions
y_pred_low, y_std_low = meta.predict(X_new, fidelity='low')

# Print results
for i in range(len(X_new)):
    print(f"Input: {X_new[i]}")
    print(f"  High-fidelity: {y_pred_high[i][0]:.4f} ± {y_std_high[i][0]:.4f}")
    print(f"  Low-fidelity: {y_pred_low[i][0]:.4f} ± {y_std_low[i][0]:.4f}")

# Print model hyperparameters
meta.print_hyperparameters()
```

## Advanced Usage

### Comparing Different Kernels

Co-Kriging can benefit from different kernel choices depending on the smoothness of your function:

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.cokriging import MetaTrainingCoKriging

# Generate data (continuing with previous synthetic data example)
# ...

# List of kernels to compare
kernels = ['matern25', 'matern15', 'matern05', 'rbf']
models = {}
metrics = {}

# Train a model with each kernel
for kernel in kernels:
    print(f"Training with {kernel} kernel...")
    meta = MetaTrainingCoKriging(
        num_iterations=300,
        kernel=kernel,
        output_dir=f'RESULT_MODEL_COKRIGING_{kernel}'
    )
    
    model, _, _ = meta.train(
        X_low=X_low, 
        y_low=y_low, 
        X_high=X_high, 
        y_high=y_high,
        X_test=X_test,
        y_test=y_test
    )
    
    models[kernel] = model
    metrics[kernel] = meta.metrics

# Compare test R² scores
plt.figure(figsize=(10, 6))
plt.bar(kernels, [metrics[k]['test_r2'] for k in kernels])
plt.ylim(0, 1)
plt.title('Test R² Score by Kernel Type')
plt.ylabel('R² Score')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

### Handling Large-Scale Multi-Fidelity Data

For larger datasets, you can use batch processing:

```python
import numpy as np
from PyEGRO.meta.cokriging.cokriging_utils import DeviceAgnosticCoKriging

# Generate or load a large dataset
n_samples = 10000
X_large = np.random.rand(n_samples, 5) * 4 - 2  # 10,000 samples, 5 features

# Load a trained model
cokriging = DeviceAgnosticCoKriging(prefer_gpu=True)
cokriging.load_model('RESULT_MODEL_COKRIGING')

# Make predictions with batch processing
y_pred, y_std = cokriging.predict(X_large, fidelity='high', batch_size=500)

print(f"Processed {n_samples} samples with shapes: {y_pred.shape}, {y_std.shape}")
```

### Uncertainty Quantification

Co-Kriging is particularly useful for uncertainty quantification in multi-fidelity simulations:

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.cokriging import MetaTrainingCoKriging

# After training a model (continuing from previous examples)
# Create a fine grid for predictions
x_grid = np.linspace(-2, 2, 200).reshape(-1, 1)

# Get predictions at both fidelity levels
y_high, std_high = meta.predict(x_grid, fidelity='high')
y_low, std_low = meta.predict(x_grid, fidelity='low')

# Plot with uncertainty bounds
plt.figure(figsize=(10, 6))
plt.plot(x_grid, y_high, 'r-', label='High-fidelity prediction')
plt.fill_between(x_grid.flatten(), 
                (y_high - 2*std_high).flatten(), 
                (y_high + 2*std_high).flatten(), 
                alpha=0.2, color='red', label='95% confidence interval')

plt.plot(x_grid, y_low, 'b--', label='Low-fidelity prediction')
plt.fill_between(x_grid.flatten(), 
                (y_low - 2*std_low).flatten(), 
                (y_low + 2*std_low).flatten(), 
                alpha=0.1, color='blue')

# Plot training data
plt.scatter(X_high, y_high, c='red', s=60, label='High-fidelity data', 
           marker='o', edgecolor='black', zorder=5)
plt.scatter(X_low, y_low, c='blue', s=40, label='Low-fidelity data', 
           marker='s', alpha=0.7, zorder=4)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Multi-fidelity predictions with uncertainty quantification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Cross-Validation for Co-Kriging

Implementing cross-validation for Co-Kriging models:

```python
import numpy as np
from sklearn.model_selection import KFold
from PyEGRO.meta.cokriging import MetaTrainingCoKriging

# Assuming X_high, y_high, X_low, y_low are already defined
# Set up cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Metrics storage
high_r2_scores = []
low_r2_scores = []

# Perform cross-validation for high-fidelity data
for train_idx, test_idx in kf.split(X_high):
    # Split data
    X_high_train, X_high_test = X_high[train_idx], X_high[test_idx]
    y_high_train, y_high_test = y_high[train_idx], y_high[test_idx]
    
    # Use all low-fidelity data (this is a common approach in multi-fidelity modeling)
    meta = MetaTrainingCoKriging(num_iterations=200, show_progress=False)
    
    # Train model
    meta.train(
        X_low=X_low, 
        y_low=y_low, 
        X_high=X_high_train, 
        y_high=y_high_train
    )
    
    # Make predictions
    y_high_pred, _ = meta.predict(X_high_test, fidelity='high')
    y_low_pred, _ = meta.predict(X_low, fidelity='low')
    
    # Calculate R² scores
    high_r2 = 1 - np.sum((y_high_test - y_high_pred) ** 2) / np.sum((y_high_test - np.mean(y_high_test)) ** 2)
    low_r2 = 1 - np.sum((y_low - y_low_pred) ** 2) / np.sum((y_low - np.mean(y_low)) ** 2)
    
    high_r2_scores.append(high_r2)
    low_r2_scores.append(low_r2)

print(f"High-fidelity CV R² scores: {high_r2_scores}")
print(f"Mean high-fidelity CV R²: {np.mean(high_r2_scores):.4f} ± {np.std(high_r2_scores):.4f}")
print(f"Low-fidelity CV R² scores: {low_r2_scores}")
print(f"Mean low-fidelity CV R²: {np.mean(low_r2_scores):.4f} ± {np.std(low_r2_scores):.4f}")
```