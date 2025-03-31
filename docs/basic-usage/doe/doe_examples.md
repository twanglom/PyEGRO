# PyEGRO.doe Module - Usage Examples

This document provides comprehensive examples of how to use the PyEGRO Design of Experiments (DOE) module for various scenarios. These examples illustrate different sampling methods, variable types, uncertainty specifications, and objective functions.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Uncertainty Specification Methods](#uncertainty-specification-methods)
- [1D Test Function Examples](#1d-test-function-examples)
- [2D Test Function Examples](#2d-test-function-examples)
- [Comparing Sampling Methods](#comparing-sampling-methods)
- [Environmental Variables](#environmental-variables)
- [Advanced Usage Patterns](#advanced-usage-patterns)

## Basic Usage

Here's a simple example to get started with the DOE module:

```python
import numpy as np
from PyEGRO.doe import InitialDesign

# Define a simple objective function
def simple_objective(x):
    return np.sum(x**2, axis=1)

# Create an initial design
design = InitialDesign(
    output_dir='simple_doe',
    sampling_method='lhs'
)

# Add two design variables
design.add_design_variable(name='x1', range_bounds=[-5, 5], cov=0.1)
design.add_design_variable(name='x2', range_bounds=[-5, 5], cov=0.1)

# Save variable data to data_info.json (useful when using metamodel in optimization loop)
design.save('data_info')

# Run sampling with 20 samples
results = design.run(
    objective_function=simple_objective,
    num_samples=20
)

# Display the first few results
print(results.head())
```

## Uncertainty Specification Methods

There are multiple ways to specify uncertainty for design variables. Here are examples for each method:

### Using Coefficient of Variation (CoV)

```python
import numpy as np
from PyEGRO.doe import InitialDesign

# Define a simple objective function
def objective_func(x):
    return np.sum(x**2, axis=1)

# Create design with CoV for uncertainty
design_cov = InitialDesign(
    output_dir='doe_cov_example',
    sampling_method='lhs'
)

# Add design variable with CoV
design_cov.add_design_variable(
    name='x',
    range_bounds=[-5, 5],
    cov=0.1,  # 10% coefficient of variation
    distribution='normal', 
    description='design variable with CoV-based uncertainty'
)

# Run sampling
results_cov = design_cov.run(
    objective_function=objective_func,
    num_samples=50
)

print(f"CoV example - Results range: [{min(results_cov['y']):.2f}, {max(results_cov['y']):.2f}]")
```

### Using Standard Deviation (Std)

```python
# Create design with standard deviation for uncertainty
design_std = InitialDesign(
    output_dir='doe_std_example',
    sampling_method='lhs'
)

# Add design variable with standard deviation
design_std.add_design_variable(
    name='x',
    range_bounds=[-5, 5],
    std=0.2,  # Fixed standard deviation of 0.2
    distribution='normal',
    description='design variable with std-based uncertainty'
)

# Run sampling
results_std = design_std.run(
    objective_function=objective_func,
    num_samples=50
)

print(f"Std example - Results range: [{min(results_std['y']):.2f}, {max(results_std['y']):.2f}]")
```

### Using Delta for Uniform Uncertainty

```python
# Create design with delta for uniform uncertainty
design_delta = InitialDesign(
    output_dir='doe_delta_example',
    sampling_method='lhs'
)

# Add design variable with uniform uncertainty
design_delta.add_design_variable(
    name='x',
    range_bounds=[-5, 5],
    delta=0.1,  # Half-width of 0.1 around design points
    distribution='uniform',
    description='design variable with uniform uncertainty'
)

# Run sampling
results_delta = design_delta.run(
    objective_function=objective_func,
    num_samples=50
)

print(f"Delta example - Results range: [{min(results_delta['y']):.2f}, {max(results_delta['y']):.2f}]")
```

### Handling Near-Zero Regions

```python
import numpy as np
from PyEGRO.doe import InitialDesign
import matplotlib.pyplot as plt

# Define an objective function that crosses zero
def sinusoidal_func(x):
    return np.sin(x[:, 0])

# Create designs with different uncertainty specifications
design_cov = InitialDesign(output_dir='doe_near_zero_cov', sampling_method='lhs')
design_std = InitialDesign(output_dir='doe_near_zero_std', sampling_method='lhs')

# Add design variables that will include zero in their range
design_cov.add_design_variable(
    name='x',
    range_bounds=[-3.14, 3.14],
    cov=0.1,  # CoV will have issues near zero
    distribution='normal'
)

design_std.add_design_variable(
    name='x',
    range_bounds=[-3.14, 3.14],
    std=0.2,  # Std works well even at zero
    distribution='normal'
)

# Run sampling
results_cov = design_cov.run(
    objective_function=sinusoidal_func,
    num_samples=200
)

results_std = design_std.run(
    objective_function=sinusoidal_func,
    num_samples=200
)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(results_cov['x'], results_cov['y'], alpha=0.7)
plt.title('Using CoV for Near-Zero Regions')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(results_std['x'], results_std['y'], alpha=0.7)
plt.title('Using Std for Near-Zero Regions')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 1D Test Function Examples

### Gaussian Mixture Function

This example demonstrates how to use the DOE module with a 1D Gaussian mixture function.

```python
import numpy as np
from PyEGRO.doe import InitialDesign
import matplotlib.pyplot as plt

# Define 1D Gaussian Mixture test function
def objective_func_1D(x):
    """1D test function with multiple Gaussian peaks"""
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x[:, 0]  # Extract first column if it's a 2D array
        
    term1 = 100 * np.exp(-((x - 10)**2) / 0.8)
    term2 = 80 * np.exp(-((x - 20)**2) / 50)
    term3 = 20 * np.exp(-((x - 30)**2) / 18)
    
    return -(term1 + term2 + term3 - 200)

# Create design with LHS sampling
design_1d = InitialDesign(
    output_dir='doe_1d_example',
    sampling_method='lhs',
    sampling_criterion='maximin'
)

# Add design variable using standard deviation instead of cov
design_1d.add_design_variable(
    name='x',
    range_bounds=[0, 40],
    std=0.5,  # Fixed standard deviation
    description='design parameter with std uncertainty'
)

# Run sampling
results_1d = design_1d.run(
    objective_function=objective_func_1D,
    num_samples=50
)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(results_1d['x'], results_1d['y'], color='red', alpha=0.7, label='Sampled Points')

# Generate the true function
x_true = np.linspace(0, 40, 1000)
y_true = objective_func_1D(x_true)
plt.plot(x_true, y_true, 'k-', label='True Function')

plt.title('1D Gaussian Mixture Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

## 2D Test Function Examples

### Gaussian Peaks Function with Delta Uncertainty

This example shows how to sample from a 2D function with two Gaussian peaks using uniform uncertainty.

```python
import numpy as np
from PyEGRO.doe import InitialDesign
import matplotlib.pyplot as plt

# Define 2D test function with two Gaussian peaks
def objective_func_2D(x):
    X1, X2 = x[:, 0], x[:, 1]
    # Parameters for two Gaussian peaks
    a1, x1, y1, sigma_x1, sigma_y1 = 100, 3, 2.1, 3, 3    # Lower and more sensitive
    a2, x2, y2, sigma_x2, sigma_y2 = 150, -1.5, -1.2, 1, 1  # Higher and more robust
    
    # Calculate negative sum of two Gaussian peaks
    f = -(
        a1 * np.exp(-((X1 - x1) ** 2 / (2 * sigma_x1 ** 2) + (X2 - y1) ** 2 / (2 * sigma_y1 ** 2))) +
        a2 * np.exp(-((X1 - x2) ** 2 / (2 * sigma_x2 ** 2) + (X2 - y2) ** 2 / (2 * sigma_y2 ** 2))) - 200
    )
    return f

# Create design with Sobol sequence and uniform uncertainty
design_2d = InitialDesign(
    output_dir='doe_2d_example_delta',
    sampling_method='sobol'
)

# Add design variables with delta for uniform uncertainty
design_2d.add_design_variable(
    name='x1',
    range_bounds=[-5, 5],
    delta=0.05,  # Half-width of 0.05 around design points
    distribution='uniform',
    description='first design variable with uniform uncertainty'
)

design_2d.add_design_variable(
    name='x2',
    range_bounds=[-6, 6],
    delta=0.05,  # Half-width of 0.05 around design points
    distribution='uniform',
    description='second design variable with uniform uncertainty'
)

# Run sampling
results_2d = design_2d.run(
    objective_function=objective_func_2D,
    num_samples=50
)

# Create a contour plot of the design space
plt.figure(figsize=(10, 8))

# Generate grid for visualization
x1_grid = np.linspace(-5, 5, 100)
x2_grid = np.linspace(-6, 6, 100)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
grid_points = np.vstack([X1.ravel(), X2.ravel()]).T
Z = objective_func_2D(grid_points).reshape(X1.shape)

# Create contour plot
plt.contourf(X1, X2, Z, 50, cmap='viridis')
plt.colorbar(label='Objective Value')
plt.scatter(results_2d['x1'], results_2d['x2'], c='red', s=50, alpha=0.7, edgecolors='white')

plt.title('2D Gaussian Peaks Function with Sobol Sampling (Uniform Uncertainty)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()
```

## Comparing Sampling Methods

This example compares different sampling methods for a 2D design space.

```python
import numpy as np
from PyEGRO.doe import InitialDesign
from PyEGRO.doe.sampling import AdaptiveDistributionSampler
import matplotlib.pyplot as plt

# Compare different sampling methods
sampling_methods = ['lhs', 'sobol', 'halton', 'random']
num_samples = 50

# Create figure
plt.figure(figsize=(16, 12))

for i, method in enumerate(sampling_methods):
    # Create design
    design = InitialDesign(
        output_dir=f'doe_compare_{method}',
        sampling_method=method,
        sampling_criterion='maximin' if method == 'lhs' else None
    )
    
    # Add design variables
    design.add_design_variable(
        name='x1',
        range_bounds=[-5, 5],
        std=0.2  # Using std instead of cov
    )
    
    design.add_design_variable(
        name='x2',
        range_bounds=[-5, 5],
        std=0.2
    )
    
    # Generate samples (without evaluating objective function)
    sampler = AdaptiveDistributionSampler()
    samples = sampler.generate_all_samples(
        design.variables,
        num_samples,
        method=method,
        criterion='maximin' if method == 'lhs' else None
    )
    
    # Plot
    plt.subplot(2, 2, i+1)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.7)
    plt.title(f"{method.upper()} Sampling ({num_samples} points)")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True)

plt.tight_layout()
plt.show()
```

## Environmental Variables

This example demonstrates how to incorporate environmental variables in your experiments with both min/max and std notation.

```python
import numpy as np
from PyEGRO.doe import InitialDesign
import matplotlib.pyplot as plt

# Define objective function with both design and environmental variables
def objective_with_env_vars(x):
    # Extract design and environmental variables
    design_var = x[:, 0]          # Design variable
    env_normal = x[:, 1]          # Normal distributed environmental variable
    env_uniform = x[:, 2]         # Uniform distributed environmental variable
    
    # Base function (using design variable)
    base_result = -design_var**2
    
    # Apply environmental effects
    # Normal env var: additive effect
    # Uniform env var: multiplicative effect
    result = (base_result + 5*env_normal) * (1 + env_uniform)
    
    return result

# Create design
design_with_env = InitialDesign(
    output_dir='doe_with_env_vars_new',
    sampling_method='lhs'
)

# Add design variable using std for uncertainty
design_with_env.add_design_variable(
    name='x',
    range_bounds=[-3, 3],
    std=0.2,  # Standard deviation (instead of cov)
    description='design parameter with std-based uncertainty'
)

# Add environmental variables with different distributions and notation
design_with_env.add_env_variable(
    name='noise_additive',
    distribution='normal',
    mean=0.0,
    std=0.5,  # Using std instead of cov
    description='additive noise (normal) with std'
)

design_with_env.add_env_variable(
    name