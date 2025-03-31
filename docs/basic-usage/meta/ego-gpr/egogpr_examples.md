# PyEGRO.meta.egogpr Usage Examples

This document provides examples of how to use the PyEGRO.meta.egogpr module to build accurate surrogate models through efficient adaptive sampling. EGO is particularly valuable when function evaluations are expensive and metamodel accuracy is critical.

## Table of Contents

1. [Basic 1D Optimization](#basic-1d-optimization)
2. [2D Function Optimization](#2d-function-optimization)
3. [Custom Configuration](#custom-configuration)
4. [Different Acquisition Functions](#different-acquisition-functions)
5. [Using Initial Data](#using-initial-data)
6. [Saving and Loading Models](#saving-and-loading-models)
7. [Multi-Dimensional Optimization](#multi-dimensional-optimization)
8. [Custom Visualization](#custom-visualization)

---

## Basic 1D Metamodel Building

In this example, we build an accurate surrogate model of a 1D function using adaptive sampling to efficiently allocate evaluation points where they most improve model accuracy.

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.egogpr import EfficientGlobalOptimization

# Define a 1D objective function to minimize
def objective_function(x):
    """Simple 1D function with multiple local minima"""
    return np.sin(x) + np.sin(10*x/3)

# Set up the bounds and variable names
bounds = np.array([[0, 10]])  # 1D bounds from 0 to 10
variable_names = ['x']

# Create and run the optimizer
optimizer = EfficientGlobalOptimization(
    objective_func=objective_function,
    bounds=bounds,
    variable_names=variable_names
)

# Run the optimization
history = optimizer.run()

# Best point found
best_x = optimizer.X_train[np.argmin(optimizer.y_train)]
best_y = np.min(optimizer.y_train)

print(f"Best point found: x = {best_x[0]:.4f}, y = {best_y:.4f}")
```

## 2D Function Optimization

This example demonstrates optimizing a 2D function - the Branin function, which is a common benchmark.

```python
import numpy as np
from PyEGRO.meta.egogpr import EfficientGlobalOptimization

# Define the 2D Branin function
def branin(x):
    """
    Branin function with three global minima
    Global minima: f(−π, 12.275) = f(π, 2.275) = f(9.42478, 2.475) ≈ 0.397887
    """
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    
    return term1 + term2 + s

# Set up the bounds and variable names
bounds = np.array([[-5, 10], [0, 15]])  # 2D bounds
variable_names = ['x1', 'x2']

# Create and run the optimizer
optimizer = EfficientGlobalOptimization(
    objective_func=branin,
    bounds=bounds,
    variable_names=variable_names
)

# Run the optimization
history = optimizer.run()

# Best point found
best_x = optimizer.X_train[np.argmin(optimizer.y_train)]
best_y = np.min(optimizer.y_train)

print(f"Best point found: x1 = {best_x[0]:.4f}, x2 = {best_x[1]:.4f}, y = {best_y:.4f}")
print(f"Known global minima: f(−π, 12.275) = f(π, 2.275) = f(9.42478, 2.475) ≈ 0.397887")
```

## Custom Configuration

This example shows how to customize the optimization process using the `TrainingConfig` class.

```python
import numpy as np
from PyEGRO.meta.egogpr import EfficientGlobalOptimization, TrainingConfig

# Define a simple objective function
def rosenbrock(x):
    """
    Rosenbrock function (a.k.a. banana function)
    Global minimum at (1, 1) with value 0
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2

# Set up bounds and variable names
bounds = np.array([[-2, 2], [-2, 2]])
variable_names = ['x1', 'x2']

# Create custom configuration
config = TrainingConfig(
    max_iterations=50,                  # Maximum optimization iterations
    rmse_threshold=0.0005,              # RMSE threshold for early stopping
    rmse_patience=15,                   # Patience for RMSE improvement
    acquisition_name="lcb",             # Use Lower Confidence Bound
    acquisition_params={"beta": 2.5},   # Custom acquisition parameters
    training_iter=150,                  # GP model training iterations
    learning_rate=0.008,                # Learning rate for model training
    kernel="rbf",                       # Use RBF kernel
    save_dir="RESULT_ROSENBROCK"        # Custom save directory
)

# Create and run the optimizer with custom config
optimizer = EfficientGlobalOptimization(
    objective_func=rosenbrock,
    bounds=bounds,
    variable_names=variable_names,
    config=config
)

# Run the optimization
history = optimizer.run()

# Best point found
best_x = optimizer.X_train[np.argmin(optimizer.y_train)]
best_y = np.min(optimizer.y_train)

print(f"Best point found: x1 = {best_x[0]:.4f}, x2 = {best_x[1]:.4f}, y = {best_y:.4f}")
```

## Different Acquisition Functions

This example demonstrates how to use different acquisition functions.

```python
import numpy as np
from egogpr import EfficientGlobalOptimization, TrainingConfig

# Define a simple objective function
def ackley(x):
    """
    Ackley function - a multimodal function with many local minima
    Global minimum at (0, 0) with value 0
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x[:, 0]**2 + x[:, 1]**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x[:, 0]) + np.cos(c * x[:, 1])))
    
    return term1 + term2 + a + np.exp(1)

# Set up bounds and variable names
bounds = np.array([[-5, 5], [-5, 5]])
variable_names = ['x1', 'x2']

# List of acquisition functions to try
acquisition_functions = [
    "ei",    # Expected Improvement
    "pi",    # Probability of Improvement
    "lcb",   # Lower Confidence Bound
    "e3i",   # Exploration Enhanced Expected Improvement
    "eigf",  # Expected Improvement for Global Fit
    "cri3"   # Criterion 3
]

results = {}

# Try each acquisition function
for acq_func in acquisition_functions:
    print(f"\nOptimizing with {acq_func.upper()} acquisition function")
    
    # Create custom config with this acquisition function
    config = TrainingConfig(
        max_iterations=30,
        acquisition_name=acq_func,
        save_dir=f"RESULT_ACKLEY_{acq_func.upper()}"
    )
    
    # Create and run optimizer
    optimizer = EfficientGlobalOptimization(
        objective_func=ackley,
        bounds=bounds,
        variable_names=variable_names,
        config=config
    )
    
    # Run optimization
    history = optimizer.run()
    
    # Store results
    best_y = np.min(optimizer.y_train)
    results[acq_func] = best_y
    
    print(f"Best value found with {acq_func.upper()}: {best_y:.6f}")

# Print summary of results
print("\n--- Summary of Results ---")
for acq_func, best_y in results.items():
    print(f"{acq_func.upper()}: {best_y:.6f}")
```

## Using Initial Data

This example shows how to start optimization with some initial data points.

```python
import numpy as np
import pandas as pd
from egogpr import EfficientGlobalOptimization

# Define objective function
def himmelblau(x):
    """
    Himmelblau's function with four equal local minima
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Create initial data with Latin Hypercube Sampling
def lhs_samples(bounds, n_samples):
    from scipy.stats.qmc import LatinHypercube
    
    n_dims = bounds.shape[0]
    sampler = LatinHypercube(d=n_dims)
    samples = sampler.random(n=n_samples)
    
    # Scale samples to bounds
    for i in range(n_dims):
        samples[:, i] = samples[:, i] * (bounds[i, 1] - bounds[i, 0]) + bounds[i, 0]
    
    return samples

# Set up bounds and variable names
bounds = np.array([[-5, 5], [-5, 5]])
variable_names = ['x1', 'x2']

# Generate initial samples
X_initial = lhs_samples(bounds, 10)
y_initial = np.array([himmelblau(x.reshape(1, -1)) for x in X_initial]).flatten()

# Create DataFrame with initial data
initial_data = pd.DataFrame({
    'x1': X_initial[:, 0],
    'x2': X_initial[:, 1],
    'y': y_initial
})

# Create and run optimizer with initial data
optimizer = EfficientGlobalOptimization(
    objective_func=himmelblau,
    bounds=bounds,
    variable_names=variable_names,
    initial_data=initial_data
)

# Run optimization
history = optimizer.run()

# Best point found
best_x = optimizer.X_train[np.argmin(optimizer.y_train)]
best_y = np.min(optimizer.y_train)

print(f"Best point found: x1 = {best_x[0]:.4f}, x2 = {best_x[1]:.4f}, y = {best_y:.4f}")
```

## Saving and Loading Models

This example demonstrates how to save and load models for later use.

```python
import numpy as np
from PyEGRO.meta.egogpr import (
    EfficientGlobalOptimization, 
    TrainingConfig,
    load_model_data,
    GPRegressionModel
)
import torch
import gpytorch

# Define objective function
def levy(x):
    """Levy function with multiple local minima and one global minimum at (1,1)"""
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    w1 = 1 + (x1 - 1) / 4
    w2 = 1 + (x2 - 1) / 4
    
    term1 = np.sin(np.pi * w1)**2
    term2 = ((w1 - 1)**2) * (1 + 10 * np.sin(np.pi * w1 + 1)**2)
    term3 = ((w2 - 1)**2) * (1 + np.sin(2 * np.pi * w2)**2)
    
    return term1 + term2 + term3

# Set up bounds and variable names
bounds = np.array([[-10, 10], [-10, 10]])
variable_names = ['x1', 'x2']

# Create config with custom save directory
config = TrainingConfig(
    max_iterations=20,
    save_dir="LEVY_MODEL"
)

# Create and run optimizer
optimizer = EfficientGlobalOptimization(
    objective_func=levy,
    bounds=bounds,
    variable_names=variable_names,
    config=config
)

# Run optimization
history = optimizer.run()

# Load the saved model
model_data = load_model_data("LEVY_MODEL")

# Extract model components
state_dict = model_data['state_dict']
scalers = model_data['scalers']
metadata = model_data['metadata']

# Recreate the model
train_x = state_dict['train_inputs'][0]
train_y = state_dict['train_targets']
kernel = state_dict['kernel']
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood, kernel=kernel)

# Load state dictionaries
model.load_state_dict(state_dict['model'])
likelihood.load_state_dict(state_dict['likelihood'])

print("Model loaded successfully")
print(f"Kernel type: {kernel}")
print(f"Number of training points: {len(train_y)}")
print(f"Input dimensions: {metadata['variable_names']}")
print(f"Best value found: {train_y.min().item():.6f}")

# Example: Make prediction at new points
def predict_with_loaded_model(x_new):
    # Scale inputs
    x_scaled = scalers['X'].transform(x_new)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    
    # Make prediction
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(x_tensor))
        mean = predictions.mean.numpy()
        std = predictions.variance.sqrt().numpy()
        
    # Transform back to original scale
    mean = scalers['y'].inverse_transform(mean.reshape(-1, 1)).flatten()
    std = std.reshape(-1, 1) * scalers['y'].scale_
    
    return mean, std

# Try a new point
new_point = np.array([[1.0, 1.0]])
mean, std = predict_with_loaded_model(new_point)

print(f"\nPrediction at {new_point[0]}:")
print(f"Mean: {mean[0]:.6f}")
print(f"Std: {std[0]:.6f}")
```

## Multi-Dimensional Optimization

This example shows how to optimize a higher-dimensional function.

```python
import numpy as np
from egogpr import EfficientGlobalOptimization, TrainingConfig

# Define a multi-dimensional objective function (Sphere function)
def sphere(x):
    """
    Sphere function - simple quadratic function
    Global minimum at origin with value 0
    """
    return np.sum(x**2, axis=1)

# Set number of dimensions
n_dims = 5

# Set up bounds and variable names
bounds = np.array([[-5, 5]] * n_dims)
variable_names = [f'x{i+1}' for i in range(n_dims)]

# Create custom config
config = TrainingConfig(
    max_iterations=50,
    acquisition_name="ei",
    save_dir=f"RESULT_SPHERE_{n_dims}D"
)

# Create and run optimizer
optimizer = EfficientGlobalOptimization(
    objective_func=sphere,
    bounds=bounds,
    variable_names=variable_names,
    config=config
)

# Run optimization
history = optimizer.run()

# Best point found
best_x = optimizer.X_train[np.argmin(optimizer.y_train)]
best_y = np.min(optimizer.y_train)

print(f"Best value found: {best_y:.6f}")
print("Best point coordinates:")
for i, name in enumerate(variable_names):
    print(f"{name} = {best_x[i]:.6f}")
```

## Custom Visualization

This example demonstrates how to customize visualizations for optimization results using matplotlib and the trained model.

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.egogpr import (
    EfficientGlobalOptimization, 
    ModelVisualizer,
    TrainingConfig
)

# Define a 1D function
def sinusoidal(x):
    """Sinusoidal function with multiple local minima"""
    return np.sin(5 * x) * np.exp(-0.5 * x) + 0.2 * x

# Set up bounds and variable names
bounds = np.array([[-2, 2]])
variable_names = ['x']

# Create and run optimizer
config = TrainingConfig(
    max_iterations=15,
    save_dir="CUSTOM_VIZ"
)

optimizer = EfficientGlobalOptimization(
    objective_func=sinusoidal,
    bounds=bounds,
    variable_names=variable_names,
    config=config
)

# Run optimization
history = optimizer.run()

# Create a custom visualization to show the optimization progress
plt.figure(figsize=(10, 6))

# Generate a fine grid of points for the true function
x_fine = np.linspace(bounds[0, 0], bounds[0, 1], 200).reshape(-1, 1)
y_fine = sinusoidal(x_fine)

# Plot the true function
plt.plot(x_fine, y_fine, 'k--', label='True Function')

# Plot all sampled points
plt.scatter(optimizer.X_train, optimizer.y_train, color='blue', 
           s=50, label='Evaluated Points')

# Highlight the best point
best_idx = np.argmin(optimizer.y_train)
plt.scatter(optimizer.X_train[best_idx], optimizer.y_train[best_idx], 
          color='red', s=100, marker='*', label='Best Point')

# Get model predictions
model_cpu = optimizer.model.cpu()
likelihood_cpu = optimizer.likelihood.cpu()
x_tensor = torch.tensor(optimizer.scaler_x.transform(x_fine), dtype=torch.float32)

model_cpu.eval()
likelihood_cpu.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood_cpu(model_cpu(x_tensor))
    mean = predictions.mean.numpy()
    std = predictions.variance.sqrt().numpy()

# Convert predictions back to original scale
mean = optimizer.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
std = std.reshape(-1, 1) * optimizer.scaler_y.scale_

# Plot the model prediction and uncertainty
plt.plot(x_fine, mean, 'b-', label='GP Prediction')
plt.fill_between(x_fine.flatten(), 
                (mean - 2 * std.flatten()), 
                (mean + 2 * std.flatten()), 
                color='blue', alpha=0.2, label='95% Confidence')

# Customize plot
plt.title('Optimization Progress with Custom Visualization', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the custom visualization
plt.savefig('custom_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Best point found: x = {optimizer.X_train[best_idx][0]:.4f}, y = {optimizer.y_train[best_idx][0]:.4f}")
print("Custom visualization saved as 'custom_visualization.png'")