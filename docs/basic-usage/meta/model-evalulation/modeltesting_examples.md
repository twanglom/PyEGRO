# PyEGRO ModelTesting Examples

This document provides practical examples of using the PyEGRO ModelTesting module to evaluate trained GPR and Co-Kriging models on unseen data.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Testing GPR Models](#testing-gpr-models)
- [Testing Co-Kriging Models](#testing-co-kriging-models)
- [Working with Custom Test Data](#working-with-custom-test-data)
- [Customizing Plots](#customizing-plots)
- [Command Line Interface](#command-line-interface)
- [Integration with Logging](#integration-with-logging)
- [Advanced Usage](#advanced-usage)

## Introduction

The ModelTesting module provides tools to evaluate trained surrogate models (GPR and Co-Kriging) on unseen data. It offers functionality for:

- Loading trained models saved by the PyEGRO training modules
- Loading or generating test data
- Computing performance metrics (R², RMSE, MAE)
- Visualizing model predictions and uncertainty
- Saving evaluation results and plots

## Installation

The PyEGRO ModelTesting module depends on the following packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- scipy
- torch
- gpytorch
- joblib

These should already be installed if you're using the PyEGRO GPR or Co-Kriging modules.

## Basic Usage

Here's a minimal example of testing a model using the ModelTesting module:

```python
from PyEGRO.meta.modeltesting import ModelTester

# Initialize the tester with the path to your trained model
tester = ModelTester(model_dir='RESULT_MODEL_GPR')

# Load the model
tester.load_model()

# Load or generate test data
X_test, y_test = tester.load_test_data(data_path='test_data.csv')

# Evaluate the model
tester.evaluate(X_test, y_test)

# Save results and generate plots
tester.save_results(output_dir='test_results')
tester.plot_results(output_dir='test_results')

# Print metrics
results = tester.test_results
print(f"R² Score: {results['r2']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")
print(f"MAE: {results['mae']:.4f}")
```

## Testing GPR Models

Example of testing a GPR model with synthetic data:

```python
from PyEGRO.meta.modeltesting import ModelTester
import numpy as np
import matplotlib.pyplot as plt

# Create a tester for a GPR model
tester = ModelTester(
    model_dir='RESULT_MODEL_GPR_SYNTHETIC',
    model_name='gpr_model'  # Optional, will be inferred if not provided
)

# Load the model
tester.load_model()

# Create synthetic test data for 1D function
n_samples = 50
X_test = np.linspace(0, 10, n_samples).reshape(-1, 1)
y_true = X_test * np.sin(X_test) + 0.1 * np.random.randn(n_samples, 1)

# Evaluate the model
tester.evaluate(X_test, y_true)

# Generate and display plots
figures = tester.plot_results(
    output_dir='test_results',
    show_plots=True,
    save_plots=True
)

# Access and customize the uncertainty plot
fig = figures['uncertainty']
ax = fig.axes[0]
ax.set_title('GPR Prediction Uncertainty')
ax.set_xlabel('Sample Index (ordered by input value)')

# Save the customized figure
fig.savefig('custom_uncertainty.png', dpi=300)

# Print summary metrics
print(f"Performance Summary:")
print(f"R² Score: {tester.test_results['r2']:.4f}")
print(f"RMSE: {tester.test_results['rmse']:.4f}")
```

## Testing Co-Kriging Models

Example of testing a Co-Kriging model:

```python
from PyEGRO.meta.modeltesting import ModelTester

# Create a tester for a Co-Kriging model
tester = ModelTester(
    model_dir='RESULT_MODEL_COKRIGING',
    model_name='cokriging_model'  # Optional, will be inferred if not provided
)

# Load the model
tester.load_model()

# Generate test data (if no actual test data is available)
X_test, y_test = tester.load_test_data(n_samples=100, n_features=2)

# Evaluate the model
tester.evaluate(X_test, y_test)

# Generate and save plots
tester.plot_results(
    output_dir='cokriging_test_results',
    show_plots=True
)

# Print summary metrics
print(f"Co-Kriging Model Performance:")
print(f"R² Score: {tester.test_results['r2']:.4f}")
print(f"RMSE: {tester.test_results['rmse']:.4f}")
print(f"MAE: {tester.test_results['mae']:.4f}")
```

## Working with Custom Test Data

Example of loading and using custom test data from a CSV file:

```python
from PyEGRO.meta.modeltesting import ModelTester

# Create the tester
tester = ModelTester(model_dir='RESULT_MODEL_GPR')

# Load the model
tester.load_model()

# Load test data from CSV with custom column names
X_test, y_test = tester.load_test_data(
    data_path='my_test_data.csv',
    feature_cols=['x1', 'x2', 'x3'],  # Specific feature columns to use
    target_col='output'  # Target column name
)

# Evaluate and save results
tester.evaluate(X_test, y_test)
tester.save_results(output_dir='custom_data_results')
tester.plot_results()
```

## Using the Convenience Function

For quick testing, use the `load_and_test_model` convenience function:

```python
from PyEGRO.meta.modeltesting import load_and_test_model

# Test a model in one call
results = load_and_test_model(
    data_path='test_data.csv',
    model_dir='RESULT_MODEL_GPR',
    output_dir='quick_test_results',
    target_col='y',
    show_plots=True
)

# Print metrics
print(f"Quick Test Results:")
print(f"R² Score: {results['r2']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")
```

## Customizing Plots

Example of customizing the generated plots:

```python
from PyEGRO.meta.modeltesting import ModelTester
import matplotlib.pyplot as plt

# Create and evaluate as before
tester = ModelTester(model_path='RESULT_MODEL_GPR/gpr_model.pth')
tester.load_model()
X_test, y_test = tester.load_test_data()
tester.evaluate(X_test, y_test)

# Get plot figures without saving them
figures = tester.plot_results(save_plots=False, show_plots=False)

# Customize the prediction vs actual plot
pred_fig = figures['predictions']
ax = pred_fig.axes[0]
ax.set_title('Custom Prediction Plot')
ax.set_facecolor('#f5f5f5')  # Light gray background
ax.grid(True, linestyle='--', alpha=0.7)

# Customize the uncertainty plot
if 'uncertainty' in figures:
    uncert_fig = figures['uncertainty']
    ax = uncert_fig.axes[0]
    ax.set_title('Custom Uncertainty Visualization')
    ax.set_facecolor('#f0f8ff')  # Light blue background
    
    # Add a text annotation
    r2 = tester.test_results['r2']
    ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

# Save customized figures
pred_fig.savefig('custom_predictions.png', dpi=300, bbox_inches='tight')
if 'uncertainty' in figures:
    uncert_fig.savefig('custom_uncertainty.png', dpi=300, bbox_inches='tight')

# Show all figures
plt.show()
```

## Command Line Interface

The ModelTesting module can also be used from the command line:

```bash
# Test a model from a specific directory
python -m PyEGRO.meta.modeltesting --model-dir RESULT_MODEL_GPR --data test_data.csv --output test_results

# Directly specify the model file
python -m PyEGRO.meta.modeltesting --model-path RESULT_MODEL_GPR/gpr_model.pth --data test_data.csv

# Specify a different target column
python -m PyEGRO.meta.modeltesting --model-dir RESULT_MODEL_GPR --data test_data.csv --target output_value

# Don't show plots (only save them)
python -m PyEGRO.meta.modeltesting --model-dir RESULT_MODEL_GPR --data test_data.csv --no-plot
```

## Integration with Logging

Example of integrating with Python's logging system:

```python
import logging
from PyEGRO.meta.modeltesting import ModelTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_tester")

# Create tester with logger
tester = ModelTester(
    model_dir='RESULT_MODEL_GPR',
    logger=logger
)

# Now all operations will be logged
tester.load_model()
X_test, y_test = tester.load_test_data()
tester.evaluate(X_test, y_test)
tester.save_results()
```

## Advanced Usage

### Comparing Multiple Models

Example of comparing multiple models on the same test data:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEGRO.meta.modeltesting import ModelTester

# Generate test data
n_samples = 100
n_features = 2
X_test = np.random.rand(n_samples, n_features) * 10
y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1]) + 0.1 * np.random.randn(n_samples)
y_test = y_test.reshape(-1, 1)

# Models to compare
model_dirs = [
    'RESULT_MODEL_GPR_RBF',
    'RESULT_MODEL_GPR_MATERN',
    'RESULT_MODEL_COKRIGING'
]

# Store results
results = {}

# Test each model
for model_dir in model_dirs:
    # Extract model name from directory
    model_name = model_dir.split('_')[-1].lower()
    
    # Create tester
    tester = ModelTester(model_dir=model_dir)
    
    # Load and evaluate
    tester.load_model()
    tester.evaluate(X_test, y_test)
    
    # Store results
    results[model_name] = {
        'r2': tester.test_results['r2'],
        'rmse': tester.test_results['rmse'],
        'mae': tester.test_results['mae']
    }
    
    # Save individual results
    tester.save_results(output_dir='comparison_results')

# Create comparison bar chart
metrics = ['r2', 'rmse', 'mae']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # Extract values for this metric
    values = [results[model][metric] for model in results]
    
    # Create bar chart
    ax.bar(list(results.keys()), values)
    ax.set_title(f'{metric.upper()} Comparison')
    ax.set_ylim(0, max(values) * 1.2)
    
    # Add value labels on bars
    for j, v in enumerate(values):
        ax.text(j, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# Create comparison table
comparison_df = pd.DataFrame(results).T
comparison_df.columns = ['R²', 'RMSE', 'MAE']
print(comparison_df)
comparison_df.to_csv('model_comparison.csv')
```

### Custom Model Evaluation Workflow

Example of a custom evaluation workflow using the ModelTester:

```python
from PyEGRO.meta.modeltesting import ModelTester
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('full_dataset.csv')
X = data.drop('target', axis=1).values
y = data['target'].values.reshape(-1, 1)

# Setup cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# Path to model
model_path = 'RESULT_MODEL_GPR/gpr_model.pth'

# Loop through folds
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Evaluating fold {fold+1}/5")
    
    # Split data
    X_test_fold = X[test_idx]
    y_test_fold = y[test_idx]
    
    # Create tester
    tester = ModelTester(model_path=model_path)
    tester.load_model()
    
    # Evaluate on this fold
    tester.evaluate(X_test_fold, y_test_fold)
    
    # Store results for this fold
    fold_results.append({
        'fold': fold+1,
        'r2': tester.test_results['r2'],
        'rmse': tester.test_results['rmse'],
        'mae': tester.test_results['mae'],
        'n_samples': len(test_idx)
    })
    
    # Save plots for this fold
    tester.plot_results(output_dir=f'cv_results/fold_{fold+1}')

# Create summary DataFrame
results_df = pd.DataFrame(fold_results)
print(results_df)

# Calculate average metrics
avg_metrics = {
    'r2_mean': results_df['r2'].mean(),
    'r2_std': results_df['r2'].std(),
    'rmse_mean': results_df['rmse'].mean(),
    'rmse_std': results_df['rmse'].std(),
    'mae_mean': results_df['mae'].mean(),
    'mae_std': results_df['mae'].std()
}

# Print summary
print("\nCross-Validation Results:")
print(f"R² Score: {avg_metrics['r2_mean']:.4f} ± {avg_metrics['r2_std']:.4f}")
print(f"RMSE: {avg_metrics['rmse_mean']:.4f} ± {avg_metrics['rmse_std']:.4f}")
print(f"MAE: {avg_metrics['mae_mean']:.4f} ± {avg_metrics['mae_std']:.4f}")

# Save summary
with open('cv_results/summary.txt', 'w') as f:
    f.write("Cross-Validation Results:\n")
    f.write(f"R² Score: {avg_metrics['r2_mean']:.4f} ± {avg_metrics['r2_std']:.4f}\n")
    f.write(f"RMSE: {avg_metrics['rmse_mean']:.4f} ± {avg_metrics['rmse_std']:.4f}\n")
    f.write(f"MAE: {avg_metrics['mae_mean']:.4f} ± {avg_metrics['mae_std']:.4f}\n")
```