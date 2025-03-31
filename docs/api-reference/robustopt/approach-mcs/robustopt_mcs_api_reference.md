# PyEGRO.robustopt.method_mcs API Reference

This document provides detailed API documentation for the PyEGRO.robustopt.method_mcs module, a framework for robust optimization using Monte Carlo Simulation (MCS) for uncertainty quantification.

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Main Optimization Function](#main-optimization-function)
3. [Problem Class](#problem-class)
4. [Algorithm Setup](#algorithm-setup)
5. [Results Handling](#results-handling)

---

## Problem Definition

The robust optimization problem can be defined in three different ways:

1. Using a dictionary format with variables and their properties
2. Loading a pre-defined problem from a JSON file


### 1. Dictionary Format

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

### 2. Loading from JSON File

```python
import json

# Load existing problem definition
with open('data_info.json', 'r') as f:
    data_info = json.load(f)
```

### Variable Definition Format

#### Design Variables (`vars_type = 'design_vars'`)

Common parameters for design variables:
- `name`: Name of the variable (str)
- `vars_type`: Must be 'design_vars' (str)
- `distribution`: Distribution type (str)
- `range_bounds`: [lower_bound, upper_bound] (List[float])
- `cov`: Coefficient of variation (float, optional)
- `std`: Standard deviation (float, optional, alternative to cov)
- `description`: Description of the variable (str, optional)

Note:
- `cov > 0`: Uncertain design variable with specified coefficient of variation
- `cov = 0`: Deterministic design variable

#### Environmental Variables (`vars_type = 'env_vars'`)

Common parameters for environmental variables:
- `name`: Name of the variable (str)
- `vars_type`: Must be 'env_vars' (str)
- `distribution`: Distribution type (str)
- `description`: Description of the variable (str, optional)

Additional parameters depend on the distribution type.

### Supported Distribution Types

The following distributions are supported for both design and environmental variables:

#### 1. Uniform Distribution
```python
# Required parameters:
'distribution': 'uniform',
'low': float,  # Lower bound
'high': float  # Upper bound
```

#### 2. Normal Distribution
```python
# Required parameters:
'distribution': 'normal',
'mean': float,  # Mean value
# Plus one of the following:
'cov': float,   # Coefficient of variation
'std': float    # Standard deviation
# Optional:
'low': float,   # Lower bound (if truncated)
'high': float   # Upper bound (if truncated)
```

#### 3. Lognormal Distribution
```python
# Required parameters:
'distribution': 'lognormal',
'mean': float,  # Mean value
# Plus one of the following:
'cov': float,   # Coefficient of variation
'std': float    # Standard deviation
# Optional:
'low': float,   # Lower bound (if truncated)
'high': float   # Upper bound (if truncated)
```

## Main Optimization Function

### `run_robust_optimization`

Runs robust optimization using Monte Carlo Simulation for uncertainty quantification.

```python
from PyEGRO.robustopt.method_mcs.mcs import run_robust_optimization

results = run_robust_optimization(
    data_info: Dict,
    true_func: Optional[callable] = None,
    model_handler: Optional[Any] = None,
    mcs_samples: int = 10000,
    pop_size: int = 50,
    n_gen: int = 100,
    metric: str = 'hv',
    reference_point: Optional[np.ndarray] = None,
    show_info: bool = True,
    verbose: bool = False
)
```

#### Parameters

- `data_info` (Dict): Problem definition data containing variables
- `true_func` (callable, optional): True objective function for direct evaluation
- `model_handler` (Any, optional): Handler for surrogate model evaluations
- `mcs_samples` (int): Number of Monte Carlo samples per evaluation
- `pop_size` (int): Population size for NSGA-II
- `n_gen` (int): Number of generations
- `metric` (str): Performance metric ('hv' for hypervolume)
- `reference_point` (np.ndarray, optional): Reference point for hypervolume calculation
- `show_info` (bool): Whether to display information during optimization
- `verbose` (bool): Whether to print detailed progress

#### Returns

- `Dict`: Results dictionary containing:
  - `pareto_front`: Array of objective values (Mean, StdDev)
  - `pareto_set`: Array of decision variables
  - `convergence_history`: Dict with metric history
  - `runtime`: Total optimization time
  - `success`: Bool indicating successful completion

#### Notes

- Either `true_func` or `model_handler` must be provided
- The optimization minimizes both mean performance and standard deviation
- Hypervolume (HV) metric is used to track convergence

---

## Problem Class

### `RobustOptimizationProblemMCS`

Multi-objective optimization problem using Monte Carlo Sampling for uncertainty quantification.

```python
from PyEGRO.robustopt.method_mcs.mcs import RobustOptimizationProblemMCS

problem = RobustOptimizationProblemMCS(
    variables: List[Dict] or List[Variable],
    true_func: Optional[callable] = None,
    model_handler: Optional[Any] = None,
    num_mcs_samples: int = 100000,
    batch_size: int = 1000
)
```

#### Parameters

- `variables` (List): List of variable definitions (dicts or Variable objects)
- `true_func` (callable, optional): True objective function for direct evaluation
- `model_handler` (Any, optional): Handler for surrogate model evaluations
- `num_mcs_samples` (int): Number of Monte Carlo samples per evaluation
- `batch_size` (int): Batch size for model evaluations

#### Methods

##### `_evaluate_samples(samples: np.ndarray) -> np.ndarray`

Evaluates samples using either the true function or surrogate model.

##### `_generate_samples(design_point: np.ndarray) -> np.ndarray`

Generates Monte Carlo samples for a given design point.

##### `_evaluate(X: np.ndarray, out: Dict, *args, **kwargs)`

Evaluates objectives using Monte Carlo simulation.

---

## Algorithm Setup

### `setup_algorithm`

Sets up the NSGA-II algorithm with standard parameters.

```python
from PyEGRO.robustopt.method_mcs.mcs import setup_algorithm

algorithm = setup_algorithm(pop_size: int)
```

#### Parameters

- `pop_size` (int): Population size for NSGA-II

#### Returns

- `NSGA2`: Configured NSGA-II algorithm

---

## Results Handling

### `save_optimization_results`

Saves optimization results and creates visualizations.

```python
from PyEGRO.robustopt.method_mcs.mcs import save_optimization_results

save_optimization_results(
    results: Dict,
    data_info: Dict,
    save_dir: str = 'RESULT_PARETO_FRONT_MCS'
)
```

#### Parameters

- `results` (Dict): Optimization results dictionary
- `data_info` (Dict): Problem definition data
- `save_dir` (str): Directory to save results

#### Saved Outputs

- `pareto_solutions.csv`: CSV file with Pareto front solutions
- `optimization_summary.txt`: Text summary of optimization results
- `convergence.csv`: CSV file with convergence history
- Various visualization plots:
  - Pareto front plot
  - Convergence plot
  - Parameter correlation plots
  - Trade-off analysis

#### Notes

- Creates specified directory if it doesn't exist
- Automatically determines appropriate visualizations based on problem dimension