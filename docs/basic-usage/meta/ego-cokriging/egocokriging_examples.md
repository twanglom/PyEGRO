# PyEGRO.meta.egocokriging Usage Examples

This document provides examples of how to use the PyEGRO.meta.egocokriging module to build accurate surrogate models through multi-fidelity optimization. The Co-Kriging approach is particularly valuable when high-fidelity function evaluations are expensive and low-fidelity approximations are available.

## Table of Contents

1. [Basic Multi-Fidelity Modeling](#basic-multi-fidelity-modeling)
2. [Working with Initial Data](#working-with-initial-data)
3. [Custom Configuration](#custom-configuration)
4. [Different Acquisition Functions](#different-acquisition-functions)
5. [Multi-Dimensional Multi-Fidelity Optimization](#multi-dimensional-multi-fidelity-optimization)
6. [Saving and Loading Models](#saving-and-loading-models)
7. [Visualizing Multi-Fidelity Data](#visualizing-multi-fidelity-data)
8. [Custom Acquisition Strategies](#custom-acquisition-strategies)

---

## Basic Multi-Fidelity Modeling

In this example, we build a surrogate model using both low-fidelity (cheap) and high-fidelity (expensive) evaluations.

```python
import numpy as np
import matplotlib.pyplot as plt
from PyEGRO.meta.egocokriging import EfficientGlobalOptimization

# Define high-fidelity (expensive but accurate) function
def high_fidelity_func(x):
    """Expensive but accurate simulation"""
    return np.sin(8*x) + x

# Define low-fidelity (cheap but less accurate) function
def low_fidelity_func(x):
    """Cheap approximation"""
    return 0.8*np.sin(8*x) + 1.2*x

# Set up the bounds and variable names
bounds = np.array([[0, 1]])
variable_names = ['x']

# Create and run the optimizer with both fidelity levels
optimizer = EfficientGlobalOptimization(
    objective_func=high_fidelity_func,
    bounds=bounds,
    variable_names=variable_names,
    low_fidelity_func=low_fidelity_func
)

# Run the optimization
history = optimizer.run()

# Print results
best_x = optimizer.X_train[np.argmin(optimizer.y_train)]
best_y = np.min(optimizer.y_train)
print(f"Best point found: x = {best_x[0]:.4f}, y = {best_y:.4f}")

# Print final correlation parameter (rho)
final_rho = optimizer.rho_history[-1]
print(f"Final correlation parameter (rho): {final_rho:.4f}")
```

## Working with Initial Data

This example demonstrates how to start with pre-existing data from different fidelity levels.

```python
import numpy as np
import pandas as pd
from PyEGRO.meta.egocokriging import EfficientGlobalOptimization

# Define objective functions
def high_fidelity_func(x):
    return (6*x - 2)**2 * np.sin(12*x - 4)

def low_fidelity_func(x):
    return 0.5 * high_fidelity_func(x) + 10*(x - 0.5) + 5

# Set up bounds and variable names
bounds = np.array([[0, 1]])
variable_names = ['x']

# Create initial data with different fidelity levels
# High fidelity points (expensive, use fewer points)
x_high = np.array([0.0, 0.4, 0.8, 1.0]).reshape(-1, 1)
y_high = high_fidelity_func(x_high)

# Low fidelity points (cheap, use more points)
x_low = np.linspace(0, 1, 10).reshape(-1, 1)
y_low = low_fidelity_func(x_low)

# Combine data, marking fidelity levels
initial_data = pd.DataFrame({
    'x': np.vstack([x_high, x_low]).flatten(),
    'y': np.vstack([y_high, y_low]).flatten(),
    'fidelity': np.vstack([
        np.ones_like(x_high),   # 1 = high fidelity
        np.zeros_like(x_low)    # 0 = low fidelity
    ]).flatten()
})

# Create optimizer with initial data
optimizer = EfficientGlobalOptimization(
    objective_func=high_fidelity_func,
    bounds=bounds,
    variable_names=variable_names,
    low_fidelity_func=low_fidelity_func,
    initial_data=initial_data
)

# Run optimization (will adaptively choose which fidelity to evaluate)
history = optimizer.run()

# Analyze results
best_x = optimizer.X_train[np.argmin(optimizer.y_train)]
best_y = np.min(optimizer.y_train)

print(f"Best point found: x = {best_x[0]:.4f}, y = {best_y:.4f}")
print(f"Final rho value: {optimizer.rho_history[-1]:.4f}")

# Count high vs low fidelity evaluations
high_fidelity_count = sum(optimizer.fidelities == 1)
low_fidelity_count = sum(optimizer.fidelities == 0)
print(f"High fidelity evaluations: {high_fidelity_count}")
print(f"Low fidelity evaluations: {low_fidelity_count}")
```

## Custom Configuration

This example shows how to customize the multi-fidelity optimization process.

```python
import numpy as np
from PyEGRO.meta.egocokriging import EfficientGlobalOptimization, TrainingConfig

# Define objective functions
def high_fidelity_func(x):
    """Expensive but accurate 2D function"""
    return (np.sin(10*x[:, 0]) + np.cos(10*x[:, 1])) * np.exp(-0.5 * (x[:, 0]**2 + x[:, 1]**2))

def low_fidelity_func(x):
    """Cheap approximation"""
    return 0.8 * (np.sin(10*x[:, 0]) + np.cos(10*x[:, 1])) * np.exp(-0.5 * (x[:, 0]**2 + x[:, 1]**2)) + 0.2

# Set up bounds and variable names
bounds = np.array([[-2, 2], [-2, 2]])
variable_names = ['x1', 'x2']

# Create custom configuration for multi-fidelity optimization
config = TrainingConfig(
    max_iterations=50,                  # Maximum optimization iterations
    acquisition_name="eigf",            # Global fit acquisition function
    multi_fidelity=True,                # Enable multi-fidelity modeling
    rho_threshold=0.03,                 # Tighter convergence on rho parameter
    rho_patience=5,                     # More patience for rho convergence
    kernel="matern25",                  # Use Matérn 2.5 kernel
    learning_rate=0.005,                # Lower learning rate for stability
    training_iter=200,                  # More training iterations
    early_stopping_patience=30,         # More patience for early stopping
    save_dir="RESULT_MULTIFIDELITY"     # Custom save directory
)

# Create and run the optimizer with custom config
optimizer = EfficientGlobalOptimization(
    objective_func=high_fidelity_func,
    bounds=bounds,
    variable_names=variable_names,
    low_fidelity_func=low_fidelity_func,
    config=config
)

# Run the optimization
history = optimizer.run()

# Get results
best_x = optimizer.X_train[np.argmin(optimizer.y_train)]
best_y = np.min(optimizer.y_train)

print(f"Best point found: x1 = {best_x[0]:.4f}, x2 = {best_x[1]:.4f}, y = {best_y:.4f}")
print(f"Final rho value: {optimizer.rho_history[-1]:.4f}")
```

## Different Acquisition Functions

This example demonstrates how different acquisition functions perform in multi-fidelity optimization.

```python
import numpy as np
from PyEGRO.meta.egocokriging import EfficientGlobalOptimization, TrainingConfig

# Define objective functions
def high_fidelity_func(x):
    """Branin function (expensive)"""
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

def low_fidelity_func(x):
    """Approximation of Branin with bias"""
    result = high_fidelity_func(x)
    # Add systematic bias and noise to create low-fidelity approximation
    return 0.75 * result + 15 + np.sin(x[:, 0] * 2)

# Set up bounds and variable names
bounds = np.array([[-5, 10], [0, 15]])
variable_names = ['x1', 'x2']

# List of acquisition functions to try
acquisition_functions = [
    "ei",    # Expected Improvement
    "pi",    # Probability of Improvement
    "lcb",   # Lower Confidence Bound
    "eigf",  # Expected Improvement for Global Fit
    "e3i",   # Exploration Enhanced Expected Improvement
    "cri3"   # Criterion 3
]

results = {}

# Try each acquisition function
for acq_func in acquisition_functions:
    print(f"\nOptimizing with {acq_func.upper()}