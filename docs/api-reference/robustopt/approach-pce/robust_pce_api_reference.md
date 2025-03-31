# PyEGRO.robustopt.method_pce API Reference

This document provides detailed API documentation for the PyEGRO.robustopt.method_pce module, a framework for robust optimization using Polynomial Chaos Expansion (PCE) for uncertainty quantification.

## Table of Contents

1. [Overview](#overview)
2. [Problem Definition](#problem-definition)
3. [Main Optimization Function](#main-optimization-function)
4. [PCE Classes](#pce-classes)
5. [Algorithm Setup](#algorithm-setup)
6. [Results Handling](#results-handling)

---

## Overview

Polynomial Chaos Expansion (PCE) is a powerful spectral method for uncertainty quantification that represents stochastic responses using a series of orthogonal polynomials. Compared to Monte Carlo Simulation (MCS), PCE often requires fewer samples to achieve accurate statistical moments and offers rapid convergence for smooth responses.

The PyEGRO.robustopt.method_pce module implements PCE-based uncertainty quantification within a robust optimization framework. The optimization is performed using the NSGA-II algorithm to find solutions that balance performance (mean) and robustness (standard deviation).

---

## Problem Definition

The robust optimization problem can be defined in three different ways, just as with the MCS method:

1. Using a dictionary format with variables and their properties
2. Loading a pre-defined problem from a JSON file

For detailed variable definition formats and supported distributions, please refer to the [PyEGRO.robustopt.method_mcs API Reference](PyEGRO.robustopt.method_mcs.md).

---

## Main Optimization Function

### `run_robust_optimization`

Runs robust optimization using Polynomial Chaos Expansion for uncertainty quantification.

```python
from PyEGRO.robustopt.method_pce import run_robust_optimization

results = run_robust_optimization(
    data_info: Dict,
    true_func: Optional[callable] = None,
    model_handler: Optional[Any] = None,
    pce_samples: int = 200,
    pce_order: int = 3,
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
- `pce_samples` (int): Number of PCE samples per evaluation
- `pce_order` (int): Order of the PCE polynomial expansion
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
- PCE parameters (`pce_samples` and `pce_order`) control the accuracy and computational cost of the uncertainty quantification
- Higher PCE orders provide more accurate approximations but require more samples

---

## PCE Classes

### `PCESampler`

Class for generating samples using Polynomial Chaos Expansion techniques.

```python
from PyEGRO.robustopt.method_pce import PCESampler

sampler = PCESampler(n_samples=200, order=3)
```

#### Parameters

- `n_samples` (int): Number of samples to generate
- `order` (int): Order of the polynomial expansion

#### Methods

##### `generate_samples(mean: float, std: float, bounds: Optional[tuple] = None) -> np.ndarray`

Generate PCE samples with improved uncertainty quantification.

**Parameters:**
- `mean` (float): Mean value
- `std` (float): Standard deviation
- `bounds` (tuple, optional): Tuple containing (lower, upper) bounds

**Returns:** numpy.ndarray of samples

### `RobustOptimizationProblemPCE`

Multi-objective optimization problem using Polynomial Chaos Expansion for uncertainty quantification.

```python
from PyEGRO.robustopt.method_pce import RobustOptimizationProblemPCE

problem = RobustOptimizationProblemPCE(
    variables: List[Dict],
    true_func: Optional[callable] = None,
    model_handler: Optional[Any] = None,
    num_pce_samples: int = 200,
    pce_order: int = 3,
    batch_size: int = 1000
)
```

#### Parameters

- `variables` (List): List of variable definitions (dicts or Variable objects)
- `true_func` (callable, optional): True objective function for direct evaluation
- `model_handler` (Any, optional): Handler for surrogate model evaluations
- `num_pce_samples` (int): Number of PCE samples per evaluation
- `pce_order` (int): Order of the PCE polynomial expansion
- `batch_size` (int): Batch size for model evaluations

#### Methods

##### `_generate_samples(design_point: np.ndarray) -> np.ndarray`

Generate PCE samples for a given design point.

##### `_evaluate_samples(samples: np.ndarray) -> np.ndarray`

Evaluate samples using either the true function or surrogate model.

##### `_evaluate(X: np.ndarray, out: Dict, *args, **kwargs)`

Evaluate objectives using PCE-based uncertainty quantification.

---

## Algorithm Setup

### `setup_algorithm`

Sets up the NSGA-II algorithm with standard parameters.

```python
from PyEGRO.robustopt.method_pce import setup_algorithm

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
from PyEGRO.robustopt.method_pce import save_optimization_results

save_optimization_results(
    results: Dict,
    data_info: Dict,
    save_dir: str = 'RESULT_PARETO_FRONT_PCE'
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

- The summary output includes PCE-specific information such as PCE evaluations
- Creates specified directory if it doesn't exist
- Automatically determines appropriate visualizations based on problem dimension