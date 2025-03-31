# PyEGRO.robustopt.method_mcs Usage Examples

This document provides examples of how to use the PyEGRO.robustopt.method_mcs module for robust optimization using Monte Carlo Simulation (MCS) for uncertainty quantification.

## Table of Contents

1. [Quick Start](#quick-start)
   - [Variable Definition Methods](#variable-definition-methods)
   - [Basic Robust Optimization](#basic-robust-optimization)
   - [Custom Reference Point for HV](#custom-reference-point-for-hv)
2. [Example Functions](#example-functions)
   - [Gaussian Peaks Function](#gaussian-peaks-function)
   - [Aerospace Wing Design Example](#aerospace-wing-design-example)
   - [Manufacturing Process Optimization](#manufacturing-process-optimization)
   - [Problem with Mixed Variables](#problem-with-mixed-variables)
3. [Using Metamodels](#using-metamodels)

---

## Quick Start

### Variable Definition Methods

PyEGRO supports two different methods for defining variables in robust optimization problems:

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

### Basic Robust Optimization

In this example, we perform robust optimization on a simple test function with uncertain design variables.

```python
import numpy as np
from PyEGRO.robustopt.method_mcs import run_robust_optimization, save_optimization_results

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

# Run robust optimization
results = run_robust_optimization(
    data_info=data_info,
    true_func=branin,
    mcs_samples=5000,      # Number of Monte Carlo samples
    pop_size=50,           # Population size for NSGA-II
    n_gen=30,              # Number of generations
    show_info=True         # Display progress
)

# Save and visualize results
save_optimization_results(results, data_info, save_dir='RESULT_TEST_FUNCTION')

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

### Custom Reference Point for HV

This example demonstrates using a custom reference point for the hypervolume (HV) indicator.

```python
import numpy as np
from PyEGRO.robustopt.method_mcs import run_robust_optimization, save_optimization_results

# Define test function
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

# Set custom reference point based on preliminary analysis
# This should be worse than the worst expected solution in both objectives
reference_point = np.array([50.0, 10.0])

# Run optimization with custom reference point
results = run_robust_optimization(
    data_info=data_info,
    true_func=test_function,
    mcs_samples=5000,
    pop_size=50,
    n_gen=30,
    metric='hv',
    reference_point=reference_point
)

# Save results
save_optimization_results(results, data_info, save_dir='RESULT_CUSTOM_REFERENCE')

# Print hypervolume convergence
print("\nHypervolume Convergence:")
print("-" * 30)
hv_values = results['convergence_history']['metric_values']
iterations = range(1, len(hv_values) + 1)

print(f"{'Iteration':<10} | {'Hypervolume':<15}")
print("-" * 30)
for i, hv in zip(iterations[::5], hv_values[::5]):  # Print every 5th value
    print(f"{i:<10} | {hv:<15.6f}")

print(f"\nFinal hypervolume: {hv_values[-1]:.6f}")
print(f"Used reference point: {results['convergence_history']['reference_point']}")
```

## Example Functions

### Gaussian Peaks Function

This example uses a 2D function with two Gaussian peaks, where one peak is higher but more sensitive to parameter variations.

```python
import numpy as np
from PyEGRO.robustopt.method_mcs import run_robust_optimization, save_optimization_results

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

# Run robust optimization
results = run_robust_optimization(
    data_info=data_info,
    true_func=objective_func_2D,
    mcs_samples=8000,
    pop_size=100,
    n_gen=50,
    show_info=True
)

# Save results
save_optimization_results(results, data_info, save_dir='RESULT_GAUSSIAN_PEAKS')

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

This example demonstrates a more complex application to aerospace wing design optimization.

```python
import numpy as np
from PyEGRO.robustopt.method_mcs import run_robust_optimization, save_optimization_results

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

# Run robust optimization
results = run_robust_optimization(
    data_info=data_info,
    true_func=wing_model,
    mcs_samples=8000,
    pop_size=100,
    n_gen=80
)

# Save results
save_optimization_results(results, data_info, save_dir='RESULT_WING_DESIGN')

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

This example demonstrates optimizing a manufacturing process with multiple sources of uncertainty.

```python
import numpy as np
from PyEGRO.robustopt.method_mcs import run_robust_optimization, save_optimization_results

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

# Run robust optimization
results = run_robust_optimization(
    data_info=data_info,
    true_func=manufacturing_process,
    mcs_samples=5000,
    pop_size=100,
    n_gen=50
)

# Save results
save_optimization_results(results, data_info, save_dir='RESULT_MANUFACTURING')

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

### Problem with Mixed Variables

This example demonstrates a problem with both continuous and discrete variables.

```python
import numpy as np
from PyEGRO.robustopt.method_mcs import run_robust_optimization, save_optimization_results

# Define a mixed-variable optimization problem
def portfolio_problem(X):
    """
    Portfolio optimization problem with mixed variables.
    
    Returns the negative expected return (to be minimized).
    """
    # Investment allocations (continuous design variables)
    stock_allocation = X[:, 0]      # Allocation to stocks
    bond_allocation = X[:, 1]       # Allocation to bonds
    cash_allocation = X[:, 2]       # Allocation to cash
    
    # Asset selection (discrete design variables, but handled as continuous)
    # In real implementation, these would be rounded to integers
    stock_selection = X[:, 3]       # Stock selection (1-5)
    bond_selection = X[:, 4]        # Bond selection (1-3)
    
    # Market conditions (environmental variables)
    market_return = X[:, 5]         # Overall market return
    interest_rate = X[:, 6]         # Interest rate
    inflation = X[:, 7]             # Inflation rate
    
    # Expected returns based on selections (simplified model)
    stock_returns = [0.08, 0.09, 0.10, 0.12, 0.15]  # Expected returns for different stocks
    bond_returns = [0.03, 0.04, 0.05]               # Expected returns for different bonds
    
    # Get expected returns based on selections
    # Linear interpolation for continuous representation of discrete choices
    stock_idx = np.clip(stock_selection - 1, 0, 3.999)
    stock_idx_low = np.floor(stock_idx).astype(int)
    stock_idx_high = np.ceil(stock_idx).astype(int)
    stock_frac = stock_idx - stock_idx_low
    stock_return = (1 - stock_frac) * np.array([stock_returns[i] for i in stock_idx_low]) + \
                   stock_frac * np.array([stock_returns[i] for i in stock_idx_high])
    
    bond_idx = np.clip(bond_selection - 1, 0, 1.999)
    bond_idx_low = np.floor(bond_idx).astype(int)
    bond_idx_high = np.ceil(bond_idx).astype(int)
    bond_frac = bond_idx - bond_idx_low
    bond_return = (1 - bond_frac) * np.array([bond_returns[i] for i in bond_idx_low]) + \
                  bond_frac * np.array([bond_returns[i] for i in bond_idx_high])
    
    # Base return calculation
    cash_return = 0.01  # Fixed cash return
    
    # Market effects on returns
    stock_return = stock_return * (1 + 2 * (market_return - 0.05))  # Stocks highly affected by market
    bond_return = bond_return - 5 * (interest_rate - 0.03)          # Bonds affected by interest rates
    cash_return = cash_return - inflation                           # Cash affected by inflation
    
    # Total portfolio return
    portfolio_return = (
        stock_allocation * stock_return +
        bond_allocation * bond_return +
        cash_allocation * cash_return
    )
    
    # Return negative expected return (for minimization)
    return -portfolio_return

# Define the robust optimization problem
data_info = {
    'variables': [
        # Continuous allocation variables
        {
            'name': 'stock_allocation',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [0.0, 0.8],
            'cov': 0.01,                 # Small rebalancing drift
            'description': 'Stock allocation'
        },
        {
            'name': 'bond_allocation',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [0.0, 0.8],
            'cov': 0.01,
            'description': 'Bond allocation'
        },
        {
            'name': 'cash_allocation',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [0.0, 0.5],
            'cov': 0.005,
            'description': 'Cash allocation'
        },
        
        # Discrete selection variables (handled as continuous)
        {
            'name': 'stock_selection',
            'vars_type': 'design_vars',
            'distribution': 'fixed',
            'range_bounds': [1, 5],
            'cov': 0.0,                  # No uncertainty in selection
            'description': 'Stock selection'
        },
        {
            'name': 'bond_selection',
            'vars_type': 'design_vars',
            'distribution': 'fixed',
            'range_bounds': [1, 3],
            'cov': 0.0,                  # No uncertainty in selection
            'description': 'Bond selection'
        },
        
        # Market condition variables
        {
            'name': 'market_return',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 0.05,                # 5% average market return
            'std': 0.10,                 # 10% standard deviation
            'description': 'Market return'
        },
        {
            'name': 'interest_rate',
            'vars_type': 'env_vars',
            'distribution': 'triangular',
            'low': 0.01,                 # Minimum interest rate
            'high': 0.07,                # Maximum interest rate
            'mode': 0.03,                # Most likely interest rate
            'description': 'Interest rate'
        },
        {
            'name': 'inflation',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 0.025,               # 2.5% average inflation
            'std': 0.01,                 # 1% standard deviation
            'description': 'Inflation rate'
        }
    ]
}

# Run robust optimization
results = run_robust_optimization(
    data_info=data_info,
    true_func=portfolio_problem,
    mcs_samples=10000,
    pop_size=80,
    n_gen=50
)

# Save results
save_optimization_results(results, data_info, save_dir='RESULT_PORTFOLIO')

# Print key solutions
pareto_front = results['pareto_front']
pareto_set = results['pareto_set']

# Find key solutions
idx_best_return = np.argmin(pareto_front[:, 0])
idx_lowest_risk = np.argmin(pareto_front[:, 1])

# Post-process solutions for the best return and lowest risk portfolio
print("\nPortfolio Optimization Results:")
print("-" * 50)
print("Best Return Portfolio:")
print(f"  Stock Allocation: {pareto_set[idx_best_return, 0]:.2f}, Stock Type: {pareto_set[idx_best_return, 3]:.0f}")
print(f"  Bond Allocation: {pareto_set[idx_best_return, 1]:.2f}, Bond Type: {pareto_set[idx_best_return, 4]:.0f}")
print(f"  Cash Allocation: {pareto_set[idx_best_return, 2]:.2f}")
print(f"  Expected Return: {-pareto_front[idx_best_return, 0]:.2%}")
print(f"  Risk (StdDev): {pareto_front[idx_best_return, 1]:.2%}")

print("\nLowest Risk Portfolio:")
print(f"  Stock Allocation: {pareto_set[idx_lowest_risk, 0]:.2f}, Stock Type: {pareto_set[idx_lowest_risk, 3]:.0f}")
print(f"  Bond Allocation: {pareto_set[idx_lowest_risk, 1]:.2f}, Bond Type: {pareto_set[idx_lowest_risk, 4]:.0f}")
print(f"  Cash Allocation: {pareto_set[idx_lowest_risk, 2]:.2f}")
print(f"  Expected Return: {-pareto_front[idx_lowest_risk, 0]:.2%}")
print(f"  Risk (StdDev): {pareto_front[idx_lowest_risk, 1]:.2%}")
```

## Using Metamodels

When evaluating the original function is computationally expensive, you can use pre-trained surrogate models for efficient robust optimization. PyEGRO supports multiple surrogate model types including Gaussian Process Regression (GPR) and Cokriging.

### Using GPR Models

```python
#==========================================================================
# Using Surrogate Model - GPR
#==========================================================================
from PyEGRO.meta.gpr import gpr_utils
from PyEGRO.robustopt.method_mcs import run_robust_optimization, save_optimization_results
import json

print("\nUsing GPR Surrogate Model")
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
    mcs_samples=10000,
    pop_size=25,
    n_gen=100,
    metric='hv'
)

save_optimization_results(
    results=results,
    data_info=data_info,
    save_dir='MCS_RESULT_GPR'
)
```

### Using Cokriging Models

Cokriging models are particularly useful when you have multi-fidelity data or multiple correlated responses.

```python
#==========================================================================
# Using Surrogate Model - Cokriging
#==========================================================================
from PyEGRO.meta.cokriging import cokriging_utils
from PyEGRO.robustopt.method_mcs import run_robust_optimization, save_optimization_results
import json

print("\nUsing Cokriging Surrogate Model")
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
    mcs_samples=10000,
    pop_size=25,
    n_gen=100,
    metric='hv',
    reference_point=[5, 5]  # Custom reference point for HV calculation
)

save_optimization_results(
    results=results,
    data_info=data_info,
    save_dir='MCS_RESULT_CK'
)
```

The metamodel approach is particularly useful when:

1. The original function is computationally expensive
2. You need to run multiple robust optimizations with different settings
3. You already have a trained surrogate model from previous analyses
4. You want to leverage multi-fidelity data (especially for Cokriging)

First, the surrogate model needs to be trained separately using an appropriate DOE (Design of Experiments) and training algorithm. Once trained and saved, it can be loaded for robust optimization as shown in the examples above.

This approach significantly reduces computational time while still providing good approximations of the robustness metrics.