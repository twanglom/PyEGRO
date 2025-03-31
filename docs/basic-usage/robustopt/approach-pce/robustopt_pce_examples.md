# PyEGRO.robustopt.method_pce Usage Examples

This document provides examples of how to use the PyEGRO.robustopt.method_pce module for robust optimization using Polynomial Chaos Expansion (PCE) for uncertainty quantification.

## Table of Contents

1. [Quick Start](#quick-start)
   - [Variable Definition Methods](#variable-definition-methods)
   - [Basic Robust Optimization with PCE](#basic-robust-optimization-with-pce)
   - [Custom PCE Order and Sample Size](#custom-pce-order-and-sample-size)
2. [Example Functions](#example-functions)
   - [Gaussian Peaks Function](#gaussian-peaks-function)
   - [Aerospace Wing Design Example](#aerospace-wing-design-example)
   - [Manufacturing Process Optimization](#manufacturing-process-optimization)
3. [Using Metamodels](#using-metamodels)
   - [Using GPR Models](#using-gpr-models)
   - [Using Cokriging Models](#using-cokriging-models)

---

## Quick Start

### Variable Definition Methods

PCE supports two different methods for defining variables in robust optimization problems:

#### 1. Dictionary Format

```python
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [-5, 5],
            'cov': 0.05
        },
        {
            'name': 'e1',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': -1,
            'high': 1
        }
    ]
}
```

#### 2. Loading from JSON File

```python
import json

# Load existing problem definition
with open('data_info.json', 'r') as f:
    data_info = json.load(f)
```

### Basic Robust Optimization with PCE

This example demonstrates basic robust optimization using PCE for uncertainty quantification.

```python
import numpy as np
from PyEGRO.robustopt.method_pce import run_robust_optimization, save_optimization_results

# Define a test function for demonstration
def branin(X):
    """Simple branin function with interaction terms."""
    x1, x2 = X[:, 0], X[:, 1]

    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

# Define the problem with uncertain design variables
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'range_bounds': [-5, 10],
            'cov': 0.1,               # 10% coefficient of variation
            'distribution': 'normal'   # Normal distribution around design point
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'range_bounds': [0, 15],
            'cov': 0.05,              # 5% coefficient of variation
            'distribution': 'normal'
        }
    ]
}

# Run robust optimization with PCE
results = run_robust_optimization(
    data_info=data_info,
    true_func=branin,
    pce_samples=200,     # Number of PCE samples
    pce_order=3,         # Order of PCE expansion
    pop_size=50,         # Population size for NSGA-II
    n_gen=30,            # Number of generations
    show_info=True       # Display progress
)

# Save and visualize results
save_optimization_results(results, data_info, save_dir='RESULT_PCE_TEST')

# Print sample robust solution
pareto_set = results['pareto_set']
pareto_front = results['pareto_front']

# Get a balanced solution (mid-point of Pareto front)
idx = len(pareto_front) // 2
balanced_solution = pareto_set[idx]
mean_perf = pareto_front[idx, 0]
stddev = pareto_front[idx, 1]

print("\nBalanced Robust Solution:")
print(f"x1 = {balanced_solution[0]:.4f}, x2 = {balanced_solution[1]:.4f}")
print(f"Mean Performance = {mean_perf:.4f}, StdDev = {stddev:.4f}")
```

### Custom PCE Order and Sample Size

This example demonstrates how to customize the PCE order and sample size for different accuracy/performance tradeoffs.

```python
import numpy as np
from PyEGRO.robustopt.method_pce import run_robust_optimization, save_optimization_results

# Define a simple test function
def test_function(X):
    """Sphere function with environmental noise."""
    x = X[:, 0:2]  # Design variables
    e = X[:, 2]    # Environmental variable
    
    # Calculate base function
    f_base = np.sum(x**2, axis=1)
    
    # Add effect of environmental variable
    f = f_base + 0.2 * e * (x[:, 0] + x[:, 1])
    
    return f

# Define problem using dictionary format
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [-5, 5],
            'cov': 0.05
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [-5, 5],
            'cov': 0.05
        },
        {
            'name': 'e1',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': -1,
            'high': 1
        }
    ]
}

# Higher order PCE with more samples for better accuracy
results_high_accuracy = run_robust_optimization(
    data_info=data_info,
    true_func=test_function,
    pce_samples=500,    # More samples
    pce_order=5,        # Higher order expansion
    pop_size=50,
    n_gen=30,
    metric='hv'
)

save_optimization_results(results_high_accuracy, data_info, save_dir='RESULT_PCE_HIGH_ACCURACY')

# Lower order PCE with fewer samples for faster execution
results_fast = run_robust_optimization(
    data_info=data_info,
    true_func=test_function,
    pce_samples=100,    # Fewer samples
    pce_order=2,        # Lower order expansion
    pop_size=50,
    n_gen=30,
    metric='hv'
)

save_optimization_results(results_fast, data_info, save_dir='RESULT_PCE_FAST')

# Compare convergence and accuracy
print("\nPCE Configuration Comparison:")
print("-" * 40)
print(f"High Accuracy PCE (Order 5, 500 samples):")
print(f"  Final HV: {results_high_accuracy['convergence_history']['metric_values'][-1]:.6f}")
print(f"  Runtime: {results_high_accuracy['runtime']:.2f} seconds")

print(f"\nFast PCE (Order 2, 100 samples):")
print(f"  Final HV: {results_fast['convergence_history']['metric_values'][-1]:.6f}")
print(f"  Runtime: {results_fast['runtime']:.2f} seconds")
```

## Example Functions

### Gaussian Peaks Function

This example uses a 2D function with two Gaussian peaks, where one peak is higher but more sensitive to parameter variations.

```python
import numpy as np
from PyEGRO.robustopt.method_pce import run_robust_optimization, save_optimization_results

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

# Define problem with uncertain design variables
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [-5, 5],
            'cov': 0.2,  # Higher uncertainty to highlight robust solutions
            'description': 'First design variable'
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [-5, 5],
            'cov': 0.2,
            'description': 'Second design variable'
        }
    ]
}

# Run robust optimization with PCE
results = run_robust_optimization(
    data_info=data_info,
    true_func=objective_func_2D,
    pce_samples=300,
    pce_order=4,
    pop_size=100,
    n_gen=50,
    show_info=True
)

# Save results
save_optimization_results(results, data_info, save_dir='RESULT_PCE_GAUSSIAN_PEAKS')

# Analyze results - find best performing and most robust solutions
pareto_front = results['pareto_front']
pareto_set = results['pareto_set']

idx_best_perf = np.argmin(pareto_front[:, 0])
idx_most_robust = np.argmin(pareto_front[:, 1])

print("\nGaussian Peaks Optimization Results:")
print("-" * 50)
print(f"Best Performance Solution: x1={pareto_set[idx_best_perf, 0]:.4f}, x2={pareto_set[idx_best_perf, 1]:.4f}")
print(f"Mean={pareto_front[idx_best_perf, 0]:.4f}, StdDev={pareto_front[idx_best_perf, 1]:.4f}")
print(f"Most Robust Solution: x1={pareto_set[idx_most_robust, 0]:.4f}, x2={pareto_set[idx_most_robust, 1]:.4f}")
print(f"Mean={pareto_front[idx_most_robust, 0]:.4f}, StdDev={pareto_front[idx_most_robust, 1]:.4f}")
```

### Aerospace Wing Design Example

This example demonstrates a more complex application to aerospace wing design optimization using PCE.

```python
import numpy as np
from PyEGRO.robustopt.method_pce import run_robust_optimization, save_optimization_results

# Define a simplified wing performance model
def wing_model(X):
    """
    Simplified model for wing performance calculation.
    
    Returns a combined metric of drag and weight (lower is better).
    """
    # Design variables
    aspect_ratio = X[:, 0]       # Wing aspect ratio
    sweep_angle = X[:, 1]        # Sweep angle (degrees)
    thickness_ratio = X[:, 2]    # Thickness-to-chord ratio
    
    # Environmental variables
    mach_number = X[:, 3]        # Cruise Mach number
    altitude = X[:, 4]           # Cruise altitude (m)
    
    # Simplified physics calculations
    
    # Wave drag increases with Mach and thickness, decreases with sweep
    wave_drag = 0.01 * mach_number**2 * thickness_ratio / np.cos(np.radians(sweep_angle))
    
    # Induced drag decreases with aspect ratio
    induced_drag = 1.0 / (np.pi * aspect_ratio)
    
    # Profile drag increases with thickness
    profile_drag = 0.005 + 0.01 * thickness_ratio
    
    # Total drag
    total_drag = wave_drag + induced_drag + profile_drag
    
    # Structural weight increases with aspect ratio and decreases with sweep and thickness
    structural_weight = (1.5 * aspect_ratio**1.5) / (1 + 0.5*np.sin(np.radians(sweep_angle))) / thickness_ratio**0.5
    
    # Total performance metric (weighted sum of drag and weight)
    performance = 10 * total_drag + structural_weight
    
    return performance

# Define the robust wing design problem
data_info = {
    'variables': [
        # Design variables
        {
            'name': 'aspect_ratio',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [6, 12],
            'cov': 0.02,                # Manufacturing tolerance
            'description': 'Wing aspect ratio'
        },
        {
            'name': 'sweep_angle',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [20, 40],   # degrees
            'cov': 0.01,                # Manufacturing tolerance
            'description': 'Wing sweep angle'
        },
        {
            'name': 'thickness_ratio',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [0.08, 0.16],
            'cov': 0.03,                # Manufacturing tolerance
            'description': 'Thickness-to-chord ratio'
        },
        
        # Environmental variables (operating conditions)
        {
            'name': 'mach_number',
            'vars_type': 'env_vars',
            'distribution': 'triangular',
            'low': 0.75,      # Minimum Mach
            'high': 0.85,     # Maximum Mach
            'mode': 0.78,     # Most common Mach
            'description': 'Cruise Mach number'
        },
        {
            'name': 'altitude',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 11000,  # 11,000 m (typical cruise altitude)
            'cov': 0.1,      # CoV 10%
            'description': 'Cruise altitude'
        }
    ]
}

# Run robust optimization with PCE
results = run_robust_optimization(
    data_info=data_info,
    true_func=wing_model,
    pce_samples=400,
    pce_order=4,
    pop_size=100,
    n_gen=80
)

# Save results
save_optimization_results(results, data_info, save_dir='RESULT_PCE_WING_DESIGN')

# Print results summary
pareto_set = results['pareto_set']
pareto_front = results['pareto_front']

print("\nWing Design Optimization Results:")
print("-" * 50)

# Extract key solutions
idx_best_perf = np.argmin(pareto_front[:, 0])
idx_most_robust = np.argmin(pareto_front[:, 1])

print(f"{'Solution':<15} | {'Aspect Ratio':<12} | {'Sweep Angle':<12} | {'Thickness':<10} | {'Performance':<12} | {'StdDev':<10}")
print("-" * 85)

print(f"{'Best Performance':<15} | {pareto_set[idx_best_perf, 0]:<12.2f} | "
      f"{pareto_set[idx_best_perf, 1]:<12.2f} | {pareto_set[idx_best_perf, 2]:<10.4f} | "
      f"{pareto_front[idx_best_perf, 0]:<12.4f} | {pareto_front[idx_best_perf, 1]:<10.4f}")

print(f"{'Most Robust':<15} | {pareto_set[idx_most_robust, 0]:<12.2f} | "
      f"{pareto_set[idx_most_robust, 1]:<12.2f} | {pareto_set[idx_most_robust, 2]:<10.4f} | "
      f"{pareto_front[idx_most_robust, 0]:<12.4f} | {pareto_front[idx_most_robust, 1]:<10.4f}")
```

### Manufacturing Process Optimization

This example demonstrates optimizing a manufacturing process with multiple sources of uncertainty using PCE.

```python
import numpy as np
from PyEGRO.robustopt.method_pce import run_robust_optimization, save_optimization_results

# Define a manufacturing process model
def manufacturing_process(X):
    """
    Model of a manufacturing process with multiple parameters.
    
    Returns the total cost (to be minimized).
    """
    # Process parameters (design variables)
    temp = X[:, 0]           # Process temperature
    time = X[:, 1]           # Process time
    pressure = X[:, 2]       # Process pressure
    
    # Material properties (environmental variables)
    material_density = X[:, 3]
    material_purity = X[:, 4]
    
    # Machine variations (environmental variables)
    machine_bias = X[:, 5]
    
    # Quality calculation
    base_quality = (
        -0.1 * (temp - 350)**2 -     # Temperature effect
        0.05 * (time - 60)**2 -      # Time effect
        0.02 * (pressure - 50)**2     # Pressure effect
    )
    
    # Material effects on quality
    material_effect = 20 * material_purity - 0.001 * material_density
    
    # Machine bias effect
    machine_effect = -5 * machine_bias
    
    # Final quality (higher is better)
    quality = base_quality + material_effect + machine_effect
    
    # Cost calculation
    energy_cost = 0.01 * temp * time
    material_cost = 0.1 * material_density
    
    # Time-based costs
    labor_cost = 0.5 * time
    
    # Total cost
    total_cost = energy_cost + material_cost + labor_cost - 0.2 * quality
    
    return total_cost

# Define the robust optimization problem
data_info = {
    'variables': [
        # Process parameters (design variables)
        {
            'name': 'temperature',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [300, 400],  # degrees C
            'cov': 0.02,                 # 2% control uncertainty
            'description': 'Process temperature'
        },
        {
            'name': 'time',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [30, 90],    # minutes
            'cov': 0.03,                 # 3% control uncertainty
            'description': 'Process time'
        },
        {
            'name': 'pressure',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [30, 70],    # units
            'cov': 0.01,                 # 1% control uncertainty
            'description': 'Process pressure'
        },
        
        # Material properties (environmental variables)
        {
            'name': 'material_density',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 2000,
            'std': 50,
            'description': 'Material density'
        },
        {
            'name': 'material_purity',
            'vars_type': 'env_vars',
            'distribution': 'beta',
            'alpha': 9,     # Shape parameter
            'beta': 1,      # Shape parameter
            'low': 0.9,
            'high': 1.0,    # Range for scaling
            'description': 'Material purity'
        },
        
        # Machine variation (environmental variable)
        {
            'name': 'machine_bias',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 0,
            'std': 0.1,
            'description': 'Machine bias'
        }
    ]
}

# Run robust optimization with PCE
results = run_robust_optimization(
    data_info=data_info,
    true_func=manufacturing_process,
    pce_samples=300,
    pce_order=3,
    pop_size=100,
    n_gen=50
)

# Save results
save_optimization_results(results, data_info, save_dir='RESULT_PCE_MANUFACTURING')

# Print interesting solutions
print("\nManufacturing Process Optimization Results:")
print("-" * 60)

pareto_set = results['pareto_set']
pareto_front = results['pareto_front']

# Find solutions
idx_best_cost = np.argmin(pareto_front[:, 0])
idx_most_robust = np.argmin(pareto_front[:, 1])
idx_balanced = np.argmin(pareto_front[:, 0] * 0.7 + pareto_front[:, 1] * 0.3)  # Weighted balance

print(f"{'Solution':<15} | {'Temp (°C)':<10} | {'Time (min)':<10} | {'Pressure':<10} | {'Cost':<10} | {'StdDev':<10}")
print("-" * 72)

print(f"{'Lowest Cost':<15} | {pareto_set[idx_best_cost, 0]:<10.1f} | "
      f"{pareto_set[idx_best_cost, 1]:<10.1f} | {pareto_set[idx_best_cost, 2]:<10.1f} | "
      f"{pareto_front[idx_best_cost, 0]:<10.2f} | {pareto_front[idx_best_cost, 1]:<10.2f}")

print(f"{'Most Robust':<15} | {pareto_set[idx_most_robust, 0]:<10.1f} | "
      f"{pareto_set[idx_most_robust, 1]:<10.1f} | {pareto_set[idx_most_robust, 2]:<10.1f} | "
      f"{pareto_front[idx_most_robust, 0]:<10.2f} | {pareto_front[idx_most_robust, 1]:<10.2f}")

print(f"{'Balanced':<15} | {pareto_set[idx_balanced, 0]:<10.1f} | "
      f"{pareto_set[idx_balanced, 1]:<10.1f} | {pareto_set[idx_balanced, 2]:<10.1f} | "
      f"{pareto_front[idx_balanced, 0]:<10.2f} | {pareto_front[idx_balanced, 1]:<10.2f}")
```

## Using Metamodels

When evaluating the original function is computationally expensive, you can use pre-trained surrogate models for efficient robust optimization with PCE. PyEGRO supports multiple surrogate model types including Gaussian Process Regression (GPR) and Cokriging.

### Using GPR Models

```python
#==========================================================================
# Using Surrogate Model - GPR with PCE
#==========================================================================
from PyEGRO.meta.gpr import gpr_utils
from PyEGRO.robustopt.method_pce import run_robust_optimization, save_optimization_results
import json

print("\nUsing GPR Surrogate Model with PCE")
print("=" * 50)

# Load existing problem definition
with open('DATA_PREPARATION/data_info.json', 'r') as f:
    data_info = json.load(f)

# Initialize GPR handler (**GPR need to be trained)
gpr_handler = gpr_utils.DeviceAgnosticGPR(prefer_gpu=True)
gpr_handler.load_model('RESULT_MODEL_GPR')

# Run optimization with surrogate
results = run_robust_optimization(
    model_handler=gpr_handler,
    data_info=data_info,
    pce_samples=200,     # PCE samples per evaluation
    pce_order=3,         # PCE polynomial order
    pop_size=25,
    n_gen=100,
    metric='hv'
)

save_optimization_results(
    results=results,
    data_info=data_info,
    save_dir='PCE_RESULT_GPR'
)
```

### Using Cokriging Models

Cokriging models are particularly useful when you have multi-fidelity data or multiple correlated responses.

```python
#==========================================================================
# Using Surrogate Model - Cokriging with PCE
#==========================================================================
from PyEGRO.meta.cokriging import cokriging_utils
from PyEGRO.robustopt.method_pce import run_robust_optimization, save_optimization_results
import json

print("\nUsing Cokriging Surrogate Model with PCE")
print("=" * 50)

# Load existing problem definition
with open('DATA_PREPARATION/data_info.json', 'r') as f:
    data_info = json.load(f)

# Initialize Cokriging handler (**Cokriging model need to be trained)
cokriging_handler = cokriging_utils.DeviceAgnosticCoKriging(prefer_gpu=True)
cokriging_handler.load_model('RESULT_MODEL_COKRIGING')

# Run optimization with surrogate
results = run_robust_optimization(
    model_handler=cokriging_handler,
    data_info=data_info,
    pce_samples=200,     # PCE samples per evaluation
    pce_order=3,         # PCE polynomial order
    pop_size=25,
    n_gen=100,
    metric='hv',
    reference_point=[5, 5]  # Custom reference point for HV calculation
)

save_optimization_results(
    results=results,
    data_info=data_info,
    save_dir='PCE_RESULT_CK'
)
```

## Comparing PCE with MCS

This example demonstrates a comparison between PCE and MCS approaches for the same problem.

```python
import numpy as np
import time
from PyEGRO.robustopt.method_pce import run_robust_optimization as pce_optimization
from PyEGRO.robustopt.method_mcs import run_robust_optimization as mcs_optimization
from PyEGRO.robustopt.method_pce import save_optimization_results

# Define a test function
def test_function(X):
    """Test function for uncertainty quantification comparison."""
    x1, x2 = X[:, 0], X[:, 1]
    e1 = X[:, 2]   # Environmental variable
    
    # Base function with non-linearity
    f_base = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    
    # Add environmental impact
    f = f_base + 0.5 * e1 * (x1 + x2)
    
    return f

# Define problem with uncertain design variables
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [-5, 5],
            'cov': 0.1,
            'description': 'First design variable'
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [-5, 5],
            'cov': 0.1,
            'description': 'Second design variable'
        },
        {
            'name': 'e1',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 0,
            'std': 1,
            'description': 'Environmental variable'
        }
    ]
}

# Run PCE-based optimization
print("\nRunning PCE-based optimization...")
start_time_pce = time.time()
pce_results = pce_optimization(
    data_info=data_info,
    true_func=test_function,
    pce_samples=200,
    pce_order=3,
    pop_size=50,
    n_gen=30
)
pce_time = time.time() - start_time_pce

# Run MCS-based optimization
print("\nRunning MCS-based optimization...")
start_time_mcs = time.time()
mcs_results = mcs_optimization(
    data_info=data_info,
    true_func=test_function,
    mcs_samples=5000,  # Typically MCS needs more samples than PCE
    pop_size=50,
    n_gen=30
)
mcs_time = time.time() - start_time_mcs

# Save results
save_optimization_results(pce_results, data_info, save_dir='RESULT_PCE_COMPARISON')
save_optimization_results(mcs_results, data_info, save_dir='RESULT_MCS_COMPARISON')

# Compare results
print("\nMethod Comparison:")
print("-" * 50)
print(f"PCE approach (Order=3, Samples=200):")
print(f"  Runtime: {pce_time:.2f} seconds")
print(f"  Final HV: {pce_results['convergence_history']['metric_values'][-1]:.6f}")
print(f"  Number of total evaluations: {pce_results['convergence_history']['n_evals'][-1]}")

print(f"\nMCS approach (Samples=5000):")
print(f"  Runtime: {mcs_time:.2f} seconds")
print(f"  Final HV: {mcs_results['convergence_history']['metric_values'][-1]:.6f}")
print(f"  Number of total evaluations: {mcs_results['convergence_history']['n_evals'][-1]}")

# Extract best solutions from each approach
pce_idx_best = np.argmin(pce_results['pareto_front'][:, 0])
mcs_idx_best = np.argmin(mcs_results['pareto_front'][:, 0])

print("\nBest Solutions Found:")
print(f"PCE Best: x1={pce_results['pareto_set'][pce_idx_best, 0]:.4f}, "
      f"x2={pce_results['pareto_set'][pce_idx_best, 1]:.4f}")
print(f"MCS Best: x1={mcs_results['pareto_set'][mcs_idx_best, 0]:.4f}, "
      f"x2={mcs_results['pareto_set'][mcs_idx_best, 1]:.4f}")

# Compare performance vs. robustness tradeoffs
print("\nTradeoff Analysis:")
print(f"PCE Mean Range: [{np.min(pce_results['pareto_front'][:, 0]):.4f}, "
      f"{np.max(pce_results['pareto_front'][:, 0]):.4f}]")
print(f"PCE StdDev Range: [{np.min(pce_results['pareto_front'][:, 1]):.4f}, "
      f"{np.max(pce_results['pareto_front'][:, 1]):.4f}]")

print(f"MCS Mean Range: [{np.min(mcs_results['pareto_front'][:, 0]):.4f}, "
      f"{np.max(mcs_results['pareto_front'][:, 0]):.4f}]")
print(f"MCS StdDev Range: [{np.min(mcs_results['pareto_front'][:, 1]):.4f}, "
      f"{np.max(mcs_results['pareto_front'][:, 1]):.4f}]")
```

## PCE Parameter Study

This example demonstrates how different PCE parameters affect the accuracy and computational cost of the uncertainty quantification.

```python
import numpy as np
import time
from PyEGRO.robustopt.method_pce import PCESampler

# Define test function for parameter study
def nonlinear_function(x):
    """Non-linear test function for uncertainty analysis."""
    return x**3 + 2*x**2 - 5*x + 3

# Define range of PCE settings to test
pce_orders = [2, 3, 4, 5, 6]
pce_sample_sizes = [50, 100, 200, 400, 800]

# Analytical solution for normal with std=1
true_mean = 3  # For standard normal, the mean of x^3 + 2x^2 - 5x + 3
true_std = np.sqrt(15 + 4)  # Std of x^3 + 2x^2 - 5x + 3 for standard normal

# Track results
results = []

print(f"{'Order':<6} | {'Samples':<8} | {'Mean Error %':<12} | {'StdDev Error %':<14} | {'Time (ms)':<10}")
print("-" * 60)

for order in pce_orders:
    for samples in pce_sample_sizes:
        # Create PCE sampler
        sampler = PCESampler(n_samples=samples, order=order)
        
        # Time the evaluation
        start_time = time.time()
        
        # Generate samples
        x_samples = sampler.generate_samples(mean=0, std=1)
        
        # Evaluate function
        y_samples = nonlinear_function(x_samples)
        
        # Calculate statistics
        mean = np.mean(y_samples)
        std = np.std(y_samples)
        
        elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Calculate errors
        mean_error = abs(mean - true_mean) / true_mean * 100
        std_error = abs(std - true_std) / true_std * 100
        
        # Print result
        print(f"{order:<6} | {samples:<8} | {mean_error:<12.4f} | {std_error:<14.4f} | {elapsed:<10.2f}")
        
        # Store result
        results.append({
            'order': order,
            'samples': samples,
            'mean_error': mean_error,
            'std_error': std_error,
            'time_ms': elapsed
        })

# Summarize findings
print("\nRecommended Settings:")
print("-" * 30)

# Find best accuracy per computational budget
for samples in [50, 200, 800]:
    best_order = min([r for r in results if r['samples'] == samples], 
                     key=lambda x: x['mean_error'] + x['std_error'])
    
    print(f"For {samples} samples: Order {best_order['order']} "
          f"(Mean Error: {best_order['mean_error']:.4f}%, "
          f"StdDev Error: {best_order['std_error']:.4f}%)")

# Find minimum sample size for given accuracy target
target_accuracy = 1.0  # 1% error
for order in pce_orders:
    min_samples = min([r for r in results if r['order'] == order and 
                      (r['mean_error'] + r['std_error'])/2 < target_accuracy],
                      key=lambda x: x['samples'], default=None)
    
    if min_samples:
        print(f"For Order {order}: Minimum {min_samples['samples']} samples needed "
              f"for <{target_accuracy}% error")
```

This comprehensive guide demonstrates the various ways to use PCE for robust optimization in PyEGRO, from basic examples to more complex applications and comparisons with other methods.