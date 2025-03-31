# PyEGRO.sensitivity.SAmcs API Reference

This document provides detailed API documentation for the PyEGRO.sensitivity.SAmcs module, a framework for sensitivity analysis using Monte Carlo Simulation (MCS) with the Sobol method based on Saltelli's sampling approach.

## Table of Contents

1. [Overview](#overview)
2. [Main Classes](#main-classes)
   - [MCSSensitivityAnalysis](#mcssensitivityanalysis)
   - [SensitivityVisualization](#sensitivityvisualization)
3. [Main Function](#main-function)
   - [run_sensitivity_analysis](#run_sensitivity_analysis)
4. [Usage Notes](#usage-notes)

---

## Overview

The PyEGRO.SAmcs module provides tools for global sensitivity analysis using Monte Carlo Simulation with the Sobol method. This approach offers insights into how different input variables affect the output of a model or function. The module focuses on calculating first-order and total-order sensitivity indices.

Key features:
- Implementation of Sobol method using Saltelli's sampling approach
- Support for both true function evaluation and surrogate models
- Automatic visualization of sensitivity indices
- Comprehensive reporting of results

---

## Main Classes

### MCSSensitivityAnalysis

Main class for performing sensitivity analysis using the Monte Carlo Simulation with Sobol method.

```python
from PyEGRO.sensitivity.SAmcs import MCSSensitivityAnalysis

analyzer = MCSSensitivityAnalysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    true_func=your_objective_function,  # or use model_handler
    output_dir="RESULT_SA",
    show_variables_info=True,
    random_seed=42
)

results_df = analyzer.run_analysis(
    num_samples=1024,
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
analyzer = MCSSensitivityAnalysis.from_initial_design(
    design=your_design_object,  # InitialDesign instance
    true_func=None,  # Optional override for objective function
    output_dir="RESULT_SA",
    show_variables_info=True,
    random_seed=42
)
```

#### Methods

##### `run_analysis(num_samples: int = 1024, show_progress: bool = True) -> pd.DataFrame`

Run the Monte Carlo-based sensitivity analysis using Sobol method.

**Parameters:**
- `num_samples` (int): Base sample size for Sobol sequence generation. Actual evaluations will be N*(D+2).
- `show_progress` (bool): Whether to show progress information.

**Returns:**
- `pd.DataFrame`: DataFrame containing the sensitivity indices.

##### `display_variable_info()`

Display detailed information about all variables.

### SensitivityVisualization

Handles visualization of sensitivity analysis results.

```python
from PyEGRO.sensitivity.SAmcs import SensitivityVisualization

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

Convenience function to run Monte Carlo-based sensitivity analysis with Sobol method.

```python
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis

results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    true_func=your_objective_function,  # or model_handler
    num_samples=1024,
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
- `num_samples` (int): Base sample size for Sobol sequence generation
- `output_dir` (str): Directory for saving results
- `random_seed` (int): Random seed for reproducibility
- `show_progress` (bool): Whether to show progress bar

#### Returns

- `pd.DataFrame`: DataFrame containing the sensitivity indices

---

## Usage Notes

### Sample Size Considerations

The total number of model evaluations will be `num_samples * (D + 2)`, where:
- `num_samples` is the base sample size
- `D` is the number of variables

This is due to the Saltelli sampling method's requirements.

### Outputs

The analysis produces several outputs in the specified output directory:
- `sensitivity_analysis_results.csv`: CSV file with all sensitivity indices
- `sensitivity_indices.png`: Visualization of first-order and total-order indices
- `sensitivity_indices.fig.pkl`: Pickled matplotlib figure for further customization
- `analysis_summary.json`: Summary statistics including most influential parameters and interaction strengths

### Computational Considerations

For computationally expensive models, consider:
1. Using surrogate models with the `model_handler` parameter
2. Starting with a smaller `num_samples` and gradually increasing for convergence
3. Using `show_progress=True` to monitor the analysis progress