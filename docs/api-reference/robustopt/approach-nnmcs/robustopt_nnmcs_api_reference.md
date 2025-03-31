# PyEGRO.robustopt.method_nnmcs API Reference

This document provides detailed API documentation for the PyEGRO.robustopt.method_nnmcs module, a two-stage approach for robust optimization that combines Neural Networks (NN) with Monte Carlo Simulation (MCS) for efficient uncertainty quantification.

## Table of Contents

1. [Overview](#overview)
2. [Configuration Classes](#configuration-classes)
3. [Main Optimization Class](#main-optimization-class)
4. [Pipeline Stages](#pipeline-stages)
5. [Usage Notes](#usage-notes)

---

## Overview

The Two-Stage Robust Optimization approach addresses the computational challenge in traditional robust optimization by using neural networks to accelerate the optimization process. It consists of the following main steps:

1. **Uncertainty Propagation**: Sample design points and perform low-fidelity MCS to generate training data
2. **Hyperparameter Optimization**: Find optimal neural network architectures for predicting mean and standard deviation
3. **Neural Network Training**: Train surrogate models to predict statistical moments
4. **Two-Stage Optimization**: Use neural networks in optimization loop with verification by high-fidelity MCS

This approach provides a balance between computational efficiency and solution accuracy.

---

## Configuration Classes

### `ANNConfig`

Configuration class for neural network hyperparameters and training settings.

```python
from PyEGRO.robustopt.method_nnmcs import ANNConfig

ann_config = ANNConfig(
    n_trials=50,
    n_jobs=-1,
    hyperparameter_ranges={
        'n_layers': (2, 4),
        'n_units': (64, 256),
        'learning_rate': (1e-4, 1e-2),
        'batch_size': (32, 64),
        'max_epochs': (100, 300),
        'activations': ['ReLU', 'LeakyReLU'],
        'optimizers': ['Adam'],
        'loss_functions': ['MSELoss']
    }
)
```

#### Parameters

- `n_trials` (int): Number of hyperparameter optimization trials
- `n_jobs` (int): Number of parallel jobs (-1 for all available cores)
- `hyperparameter_ranges` (Dict): Dictionary of hyperparameter ranges for optimization

### `SamplingConfig`

Configuration class for sampling parameters in the pipeline.

```python
from PyEGRO.robustopt.method_nnmcs import SamplingConfig

sampling_config = SamplingConfig(
    n_lhs_samples=2000,
    n_mcs_samples_low=10000,
    n_mcs_samples_high=1000000,
    random_seed=42
)
```

#### Parameters

- `n_lhs_samples` (int): Number of LHS design samples for initial sampling
- `n_mcs_samples_low` (int): Number of MCS samples for low-fidelity uncertainty propagation (training)
- `n_mcs_samples_high` (int): Number of MCS samples for high-fidelity verification
- `random_seed` (int): Random seed for reproducibility

### `OptimizationConfig`

Configuration class for optimization parameters.

```python
from PyEGRO.robustopt.method_nnmcs import OptimizationConfig

opt_config = OptimizationConfig(
    pop_size=50,
    n_gen=100,
    prefer_gpu=True,
    verbose=False,
    random_seed=42,
    metric='hv',
    reference_point=np.array([5.0, 5.0])
)
```

#### Parameters

- `pop_size` (int): Population size for NSGA-II
- `n_gen` (int): Number of generations for optimization
- `prefer_gpu` (bool): Whether to use GPU if available
- `verbose` (bool): Whether to print detailed information
- `random_seed` (int): Random seed for reproducibility
- `metric` (str): Performance metric ('hv' for hypervolume)
- `reference_point` (np.ndarray, optional): Reference point for hypervolume calculation

### `PathConfig`

Configuration class for file paths used in the pipeline.

```python
from PyEGRO.robustopt.method_nnmcs import PathConfig

path_config = PathConfig(
    model_type='gpr',
    model_dir='RESULT_MODEL_GPR',
    ann_model_dir='RESULT_MODEL_ANN',
    data_info_path='DATA_PREPARATION/data_info.json',
    results_dir='RESULT_PARETO_FRONT_TWOSTAGE',
    qoi_dir='RESULT_QOI'
)
```

#### Parameters

- `model_type` (str): Type of metamodel ('gpr', 'cokriging', etc.)
- `model_dir` (str): Directory for surrogate model
- `ann_model_dir` (str): Directory for neural network models
- `data_info_path` (str): Path to data_info.json file
- `results_dir` (str): Directory for optimization results
- `qoi_dir` (str): Directory for uncertainty quantification results

---

## Main Optimization Class

### `RobustOptimization`

Main class that orchestrates the two-stage robust optimization process.

```python
from PyEGRO.robustopt.method_nnmcs import RobustOptimization

optimizer = RobustOptimization(
    ann_config=ann_config,
    sampling_config=sampling_config,
    optimization_config=opt_config,
    path_config=path_config
)

pareto_set, pareto_front = optimizer.run()
```

#### Constructor Parameters

- `ann_config` (ANNConfig): Neural network configuration
- `sampling_config` (SamplingConfig): Sampling configuration
- `optimization_config` (OptimizationConfig): Optimization configuration
- `path_config` (PathConfig): Path configuration

#### Methods

##### `run() -> Tuple[np.ndarray, np.ndarray]`

Run the complete two-stage robust optimization pipeline.

**Returns:** Tuple containing the Pareto set and Pareto front

---

## Pipeline Stages

The two-stage robust optimization pipeline consists of four main stages:

### 1. Uncertainty Propagation

Samples design points using Latin Hypercube Sampling (LHS) and performs low-fidelity Monte Carlo Simulation (MCS) to generate training data for the neural networks. This stage produces a dataset of design points and their corresponding statistical moments (mean and standard deviation).

### 2. Hyperparameter Optimization

Uses Optuna to find optimal neural network architectures for predicting the mean and standard deviation. This stage searches through the hyperparameter space defined in `ANNConfig` to find configurations that minimize validation loss.

### 3. Neural Network Training

Trains two neural networks using the best hyperparameters:
- One for predicting the mean of the objective function
- One for predicting the standard deviation of the objective function

These neural networks serve as fast surrogate models for the statistical moments.

### 4. Two-Stage Optimization

Performs multi-objective optimization using the trained neural networks for fast evaluations. The optimization seeks to find the Pareto front balancing performance (mean) and robustness (standard deviation). Final solutions are verified using high-fidelity MCS.

---

## Usage Notes

### GPU Acceleration

The pipeline supports GPU acceleration for both neural network training and surrogate model evaluations. Set `prefer_gpu=True` in `OptimizationConfig` to use GPU if available.

### File Structure

The pipeline creates several directories to store intermediate and final results:
- `qoi_dir`: Contains uncertainty propagation results
- `ann_model_dir`: Contains trained neural network models
- `results_dir`: Contains final optimization results

### Model Types

The pipeline supports different types of surrogate models through the `model_type` parameter in `PathConfig`:
- 'gpr': Gaussian Process Regression
- 'cokriging': Cokriging (multi-fidelity) models

### Customization

The pipeline can be customized through the configuration classes:
- Modify `hyperparameter_ranges` in `ANNConfig` to explore different neural network architectures
- Adjust sample sizes in `SamplingConfig` to balance accuracy and computational cost
- Change optimization parameters in `OptimizationConfig` to control convergence behavior