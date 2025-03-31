# PyEGRO.sensitivity.SApce API Reference

This document provides detailed API documentation for the PyEGRO.sensitivity.SApce module, a framework for sensitivity analysis using Polynomial Chaos Expansion (PCE).

## Table of Contents

1. [Overview](#overview)
2. [Main Classes](#main-classes)
   - [PCESensitivityAnalysis](#pcesensitivityanalysis)
   - [SensitivityVisualization](#sensitivityvisualization)
3. [Main Function](#main-function)
   - [run_sensitivity_analysis](#run_sensitivity_analysis)
4. [Usage Notes](#usage-notes)

---

## Overview

The PyEGRO.sensitivity.SApce module provides tools for global sensitivity analysis using Polynomial Chaos Expansion (PCE). This approach offers an efficient alternative to sampling-based methods for analyzing how different input variables affect the output of a model or function. The module focuses on calculating first-order and total-order sensitivity indices.

Key features:
- Implementation of sensitivity analysis using Polynomial Chaos Expansion
- Support for both true function evaluation and surrogate models
- Automatic visualization of sensitivity indices
- Comprehensive reporting of results

---

## Main Classes

### PCESensitivityAnalysis

Main class for performing sensitivity analysis using Polynomial Chaos Expansion.

```python
from PyEGRO.sensitivity.SApce import PCESensitivityAnalysis

analyzer = PCESensitivityAnalysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    true_func=your_objective_function,  # or use model_handler
    output_dir="RESULT_SA",
    show_variables_info=True,
    random_seed=42
)

results_df = analyzer.run_analysis(
    polynomial_order=3,
    num_samples=1000,
    show_progress=True
)
```

#### Constructor Parameters

- `data_info_path` (str): Path to the data configuration file
- `model_handler` (Any, optional): Model handler for surrogate model (if using surrogate)
- `true_func` (callable, optional): True objective function (if using direct evaluation)
- `model_path` (str, optional): Path to trained GPR model (if using surrogate)
- `output_dir` (str): Directory for saving results
- `show_variables_info` (bool): Whether to display variable information
- `random_seed` (int): Random seed for reproducibility

#### Alternative Constructor

```python
analyzer = PCESensitivityAnalysis.from_initial_design(
    design=your_design_object,  # InitialDesign instance
    true_func=None,  # Optional override for objective function
    output_dir="RESULT_SA",
    show_variables_info=True,
    random_seed=42
)
```

#### Methods

##### `run_analysis(polynomial_order: int = 3, num_samples: int = 1000, show_progress: bool = True) -> pd.DataFrame`

Run the PCE-based sensitivity analysis.

**Parameters:**
- `polynomial_order` (int): Order of the polynomial expansion
- `num_samples` (int): Number of samples for PCE fitting
- `show_progress` (bool): Whether to show progress information

**Returns:**
- `pd.DataFrame`: DataFrame containing the sensitivity indices

##### `display_variable_info()`

Display detailed information about all variables.

### SensitivityVisualization

Handles visualization of sensitivity analysis results.

```python
from PyEGRO.sensitivity.SApce import SensitivityVisualization

visualizer = SensitivityVisualization(output_dir="RESULT_SA")
visualizer.create_visualization(results_df)
```

#### Constructor Parameters

- `output_dir` (str, optional): Directory for saving visualization outputs.

#### Methods

##### `create_visualization(sensitivity_df: pd.DataFrame)`

Create visualization for sensitivity indices.

**Parameters:**
- `sensitivity_df` (pd.DataFrame): DataFrame containing sensitivity analysis results.

---

## Main Function

### run_sensitivity_analysis

Convenience function to run PCE-based sensitivity analysis.

```python
from PyEGRO.sensitivity.SApce import run_sensitivity_analysis

results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    true_func=your_objective_function,  # or model_handler
    polynomial_order=3,
    num_samples=1000,
    output_dir="RESULT_SA",
    random_seed=42,
    show_progress=True
)
```

#### Parameters

- `data_info_path` (str): Path to the data configuration file
- `true_func` (callable, optional): True objective function (if using direct evaluation)
- `model_handler` (Any, optional): Model handler for surrogate model
- `model_path` (str, optional): Path to trained GPR model (if using surrogate)
- `polynomial_order` (int): Order of the polynomial expansion
- `num_samples` (int): Number of samples for PCE fitting
- `output_dir` (str): Directory for saving results
- `random_seed` (int): Random seed for reproducibility
- `show_progress` (bool): Whether to show progress bar

#### Returns

- `pd.DataFrame`: DataFrame containing the sensitivity indices

---

## Usage Notes

### Polynomial Order Selection

The polynomial order controls the complexity of the PCE model:
- Higher order captures more complex variable interactions but requires more samples
- Lower order is more efficient but may miss higher-order interactions
- Typical values range from 2-5 depending on problem complexity

### Sample Size Considerations

PCE generally requires fewer samples than Monte Carlo methods, but the required number depends on:
- The polynomial order (higher order requires more samples)
- The number of variables (more variables require more samples)
- A general rule of thumb: use at least 10 × P samples, where P is the number of PCE terms

### Outputs

The analysis produces several outputs in the specified output directory:
- `sensitivity_analysis_results.csv`: CSV file with all sensitivity indices
- `sensitivity_indices.png`: Visualization of first-order and total-order indices
- `sensitivity_indices.fig.pkl`: Pickled matplotlib figure for further customization
- `analysis_summary.json`: Summary statistics including most influential parameters and interaction strengths

### Computational Efficiency

PCE is generally more computationally efficient than traditional Monte Carlo methods:
- Requires fewer function evaluations for comparable accuracy
- Provides a smooth surrogate model that can be used for additional analyses
- Most suitable for smooth response surfaces with moderate dimensionality