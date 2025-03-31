# PyEGRO GPR Examples

This document provides practical examples of using the PyEGRO GPR module for Gaussian Process Regression modeling tasks.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Working with Synthetic Data](#working-with-synthetic-data)
- [Working with CSV Data](#working-with-csv-data)
- [Custom Data Preparation](#custom-data-preparation)
- [Visualization Examples](#visualization-examples)
- [Model Loading and Prediction](#model-loading-and-prediction)
- [Advanced Usage](#advanced-usage)


## Basic Usage

Here's a minimal example of training a GPR model with the PyEGRO module:

```python
import numpy as np
from PyEGRO.meta.gpr import MetaTraining

# Generate some synthetic data
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)

# Initialize meta-training
meta = MetaTraining(
    num_iterations=500,
    prefer_gpu=True,
    kernel='matern15'
)

# Train the model
model, scaler_X, scaler_y = meta.train(X=X, y=y)

# Make predictions
X_new = np.random.rand(10, 2)
y_pred, y_std = meta.predict(X_new)

print("Predictions:", y_pred.flatten())
print("Uncertainties:", y_std.flatten())
```

## Working with Synthetic Data

The following example demonstrates how to use the GPR module with synthetic data and visualize the results:

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.gpr import MetaTraining
from PyEGRO.meta.gpr.visualization import visualize_gpr

# Set random seed for reproducibility
np.random.seed(42)

# Define a true function to sample from
def true_function(x):
    return x * np.sin(x)

# Generate synthetic data
# Training data
n_train = 30
X_train = np.random.uniform(0, 10, n_train).reshape(-1, 1)
y_train = true_function(X_train) + 0.5 * np.random.randn(n_train, 1)  # Add noise

# Testing data
n_test = 50
X_test = np.linspace(0, 12, n_test).reshape(-1, 1)
y_test = true_function(X_test) + 0.25 * np.random.randn(n_test, 1)  # Add less noise

# Define bounds for visualization
bounds = np.array([[0, 12]])
variable_names = ['x']

# Initialize and train GPR model
print("Training GPR model with synthetic data...")
meta = MetaTraining(
    num_iterations=500,
    prefer_gpu=True,
    show_progress=True,
    output_dir='RESULT_MODEL_GPR_SYNTHETIC',
    kernel='matern05',
    learning_rate=0.01,
    patience=50
)

# Train model with synthetic data
model, scaler_X, scaler_y = meta.train(
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=variable_names
)

# Generate visualization
figures = visualize_gpr(
    meta=meta,
    X_train=X_train,
    y_train=y_train,
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

This example demonstrates how to use the GPR module with data stored in CSV files:

```python
import numpy as np
import pandas as pd
import json
import os
from PyEGRO.meta.gpr import MetaTraining
from PyEGRO.meta.gpr.visualization import visualize_gpr

# Load initial data and problem configuration
with open('DATA_PREPARATION/data_info.json', 'r') as f:
    data_info = json.load(f)

# Load training data
training_data = pd.read_csv('DATA_PREPARATION/training_data.csv')

# Load testing data (if available)
test_data = pd.read_csv('DATA_PREPARATION/testing_data.csv')

# Get problem configuration
bounds = np.array(data_info['input_bound'])
variable_names = [var['name'] for var in data_info['variables']]

# Get target column name (default to 'y' if not specified)
target_column = data_info.get('target_column', 'y')

# Extract features and targets
X_train = training_data[variable_names].values
y_train = training_data[target_column].values.reshape(-1, 1)

# Extract testing data
X_test = test_data[variable_names].values
y_test = test_data[target_column].values.reshape(-1, 1)

# Initialize and train GPR model
print("Training GPR model with CSV data...")
meta = MetaTraining(
    num_iterations=500,
    prefer_gpu=True,
    show_progress=True,
    output_dir='RESULT_MODEL_GPR',
    kernel='matern05',
    learning_rate=0.01,
    patience=50
)

# Train model
model, scaler_X, scaler_y = meta.train(
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=variable_names
)

# Generate visualization
figures = visualize_gpr(
    meta=meta,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    variable_names=variable_names,
    bounds=bounds,
    savefig=True
)
```

Alternatively, you can let the `MetaTraining` class load the data automatically:

```python
# Initialize with data paths
meta = MetaTraining(
    data_dir='DATA_PREPARATION',
    data_info_file='DATA_PREPARATION/data_info.json',
    data_training_file='DATA_PREPARATION/training_data.csv',
    num_iterations=500,
    prefer_gpu=True,
    kernel='matern05'
)

# Train the model (it will load data from the specified files)
model, scaler_X, scaler_y = meta.train()
```

## Custom Data Preparation

Example of preparing a data_info.json file:

```python
import json
import numpy as np
import pandas as pd

# Sample data generation
np.random.seed(42)
n_samples = 100
X1 = np.random.uniform(0, 10, n_samples)
X2 = np.random.uniform(-5, 5, n_samples)
y = 2 * X1 + 3 * X2 + np.sin(X1) * np.cos(X2) + np.random.randn(n_samples) * 0.5

# Create a DataFrame
data = pd.DataFrame({
    'x1': X1,
    'x2': X2,
    'output': y
})

# Split into training (80%) and testing (20%)
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Calculate bounds
x1_min, x1_max = data['x1'].min(), data['x1'].max()
x2_min, x2_max = data['x2'].min(), data['x2'].max()

# Create data_info.json
data_info = {
    "variables": [
        {"name": "x1", "type": "continuous"},
        {"name": "x2", "type": "continuous"}
    ],
    "input_bound": [
        [x1_min, x1_max],
        [x2_min, x2_max]
    ],
    "target_column": "output"
}

# Create DATA_PREPARATION directory
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)

# Save files
with open("DATA_PREPARATION/data_info.json", "w") as f:
    json.dump(data_info, f, indent=4)

train_data.to_csv("DATA_PREPARATION/training_data.csv", index=False)
test_data.to_csv("DATA_PREPARATION/testing_data.csv", index=False)

print("Data preparation complete.")
```

## Visualization Examples

Creating visualizations for 1D and 2D models:

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.gpr import MetaTraining
from PyEGRO.meta.gpr.visualization import visualize_gpr

# 1D Example (using previous synthetic data example)
# ...

# Generate visualizations
figures = visualize_gpr(
    meta=meta,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    variable_names=['x'],
    bounds=np.array([[0, 12]]),
    savefig=True,
    output_dir='visualizations/1d_model'
)

# 2D Example
n_samples = 100
X_train = np.random.rand(n_samples, 2) * np.array([10, 8])
y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + 0.1 * np.random.randn(n_samples)
y_train = y_train.reshape(-1, 1)

meta_2d = MetaTraining(
    num_iterations=300,
    kernel='rbf',
    output_dir='RESULT_MODEL_GPR_2D'
)

model_2d, _, _ = meta_2d.train(X=X_train, y=y_train)

figures_2d = visualize_gpr(
    meta=meta_2d,
    X_train=X_train,
    y_train=y_train,
    variable_names=['x1', 'x2'],
    bounds=np.array([[0, 10], [0, 8]]),
    savefig=True,
    output_dir='visualizations/2d_model'
)

plt.show()
```

## Model Loading and Prediction

Example of loading a trained model and making predictions:

```python
import numpy as np
from PyEGRO.meta.gpr.gpr_utils import DeviceAgnosticGPR

# Initialize device-agnostic loader
gpr_loader = DeviceAgnosticGPR(prefer_gpu=True)

# Load the trained model
loaded = gpr_loader.load_model(model_dir='RESULT_MODEL_GPR')

if loaded:
    # Generate new input data for prediction
    X_new = np.random.rand(10, 2) * np.array([10, 8])
    
    # Make predictions
    y_pred, y_std = gpr_loader.predict(X_new)
    
    # Print results
    for i in range(len(X_new)):
        print(f"Input: {X_new[i]}, Prediction: {y_pred[i][0]:.4f} ± {y_std[i][0]:.4f}")
else:
    print("Failed to load model")
```

Alternatively, use the `MetaTraining` class to load and use the model:

```python
from PyEGRO.meta.gpr import MetaTraining

# Initialize meta
meta = MetaTraining()

# Load model
meta.load_model('RESULT_MODEL_GPR/gpr_model.pth')

# Make predictions
X_new = np.random.rand(10, 2) * np.array([10, 8])
y_pred, y_std = meta.predict(X_new)

# Print results
for i in range(len(X_new)):
    print(f"Input: {X_new[i]}, Prediction: {y_pred[i][0]:.4f} ± {y_std[i][0]:.4f}")

# Print model hyperparameters
meta.print_hyperparameters()
```

## Advanced Usage

### Customizing Kernels

The GPR module supports different kernel types for different kinds of data:

```python
# For smooth functions
meta_smooth = MetaTraining(kernel='rbf')

# For less smooth functions with continuous derivatives
meta_matern25 = MetaTraining(kernel='matern25')  # Matérn 5/2

# For functions with continuous first derivatives
meta_matern15 = MetaTraining(kernel='matern15')  # Matérn 3/2

# For continuous but non-differentiable functions
meta_matern05 = MetaTraining(kernel='matern05')  # Matérn 1/2
```

### Batch Processing for Large Datasets

For larger datasets, you can use the DeviceAgnosticGPR class with batch processing:

```python
import numpy as np
from PyEGRO.meta.gpr.gpr_utils import DeviceAgnosticGPR

# Generate a large dataset
n_samples = 10000
X_large = np.random.rand(n_samples, 5)  # 10,000 samples with 5 features

# Load a trained model
gpr = DeviceAgnosticGPR(prefer_gpu=True)
gpr.load_model('RESULT_MODEL_GPR')

# Make predictions with batch processing
y_pred, y_std = gpr.predict(X_large, batch_size=500)  # Process 500 samples at a time

print(f"Processed {n_samples} samples with shapes: {y_pred.shape}, {y_std.shape}")
```

### Early Stopping and Learning Rate Scheduling

The MetaTraining class includes built-in early stopping and learning rate scheduling:

```python
meta = MetaTraining(
    num_iterations=1000,  # Maximum iterations
    learning_rate=0.01,   # Initial learning rate
    patience=50           # Patience for early stopping
)

# The optimizer will reduce the learning rate when progress plateaus
# Early stopping will trigger if no improvement after 'patience' iterations
model, _, _ = meta.train(X=X_train, y=y_train)
```