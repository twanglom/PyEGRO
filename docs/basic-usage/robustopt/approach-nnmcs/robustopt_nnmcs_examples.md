# PyEGRO.robustopt.method_nnmcs Usage Examples

This document provides examples of how to use the PyEGRO.robustopt.method_nnmcs module for robust optimization using a two-stage approach that combines Neural Networks (NN) with Monte Carlo Simulation (MCS).

## Table of Contents

1. [Quick Start](#quick-start)
   - [Basic Configuration](#basic-configuration)
   - [Running the Pipeline](#running-the-pipeline)
2. [Customizing Neural Network Training](#customizing-neural-network-training)
   - [Hyperparameter Ranges](#hyperparameter-ranges)
   - [Adjusting Training Settings](#adjusting-training-settings)
3. [Sampling Strategy](#sampling-strategy)
   - [Balancing Accuracy and Speed](#balancing-accuracy-and-speed)
4. [Using Different Surrogate Models](#using-different-surrogate-models)
   - [With GPR Models](#with-gpr-models)
   - [With Cokriging Models](#with-cokriging-models)
5. [Full Pipeline Example](#full-pipeline-example)

---

## Quick Start

### Basic Configuration

The first step is to set up the configuration objects for the pipeline:

```python
import numpy as np
from PyEGRO.robustopt.method_nnmcs import (
    RobustOptimization,
    ANNConfig,
    SamplingConfig,
    OptimizationConfig,
    PathConfig
)

# Basic configurations
ann_config = ANNConfig()  # Default values
sampling_config = SamplingConfig()  # Default values
opt_config = OptimizationConfig()  # Default values
path_config = PathConfig()  # Default values

# Initialize the optimizer
optimizer = RobustOptimization(
    ann_config=ann_config,
    sampling_config=sampling_config,
    optimization_config=opt_config,
    path_config=path_config
)
```

### Running the Pipeline

Once configured, you can run the complete pipeline:

```python
# Run the full pipeline
pareto_set, pareto_front = optimizer.run()

# Access optimization results
print(f"Number of Pareto-optimal solutions: {len(pareto_set)}")
print("\nSample Solutions:")
for i in range(min(3, len(pareto_set))):
    print(f"Solution {i+1}:")
    print(f"  Design variables: {pareto_set[i]}")
    print(f"  Objectives (Mean, StdDev): {pareto_front[i]}")
```

## Customizing Neural Network Training

### Hyperparameter Ranges

You can customize the neural network architecture search space:

```python
# Custom ANN configuration
ann_config = ANNConfig(
    n_trials=100,  # More trials for better hyperparameter optimization
    n_jobs=-1,     # Use all available cores
    hyperparameter_ranges={
        'n_layers': (2, 5),            # Network depth
        'n_units': (64, 512),          # Units per layer
        'learning_rate': (1e-5, 1e-2), # Learning rate range
        'batch_size': (16, 128),       # Batch size range
        'max_epochs': (100, 500),      # Maximum training epochs
        'activations': ['ReLU', 'LeakyReLU', 'Tanh', 'SiLU'],  # Activation functions
        'optimizers': ['Adam', 'AdamW', 'RMSprop'],             # Optimizers
        'loss_functions': ['MSELoss', 'HuberLoss', 'L1Loss']    # Loss functions
    }
)
```

### Adjusting Training Settings

You can control the training process:

```python
# ANN configuration focused on training stability
ann_config = ANNConfig(
    n_trials=50,
    n_jobs=4,  # Limit to 4 parallel jobs
    hyperparameter_ranges={
        'n_layers': (1, 3),            # Simpler networks
        'n_units': (32, 128),          # Fewer units
        'learning_rate': (1e-4, 1e-3), # More conservative learning rates
        'batch_size': (32, 64),        # Medium batch sizes
        'max_epochs': (200, 300),      # Moderate training duration
        'activations': ['ReLU'],       # Stick to ReLU
        'optimizers': ['Adam'],        # Use only Adam
        'loss_functions': ['MSELoss']  # Use only MSE loss
    }
)
```

## Sampling Strategy

### Balancing Accuracy and Speed

You can adjust the sampling configuration to balance computational cost and accuracy:

```python
# Fast exploration configuration
fast_sampling = SamplingConfig(
    n_lhs_samples=500,       # Fewer design samples
    n_mcs_samples_low=1000,  # Low-fidelity MCS for training
    n_mcs_samples_high=10000 # Medium-fidelity verification
)

# High accuracy configuration
accurate_sampling = SamplingConfig(
    n_lhs_samples=5000,        # More design samples
    n_mcs_samples_low=10000,   # Higher-fidelity training
    n_mcs_samples_high=1000000 # Very high-fidelity verification
)

# Balanced configuration
balanced_sampling = SamplingConfig(
    n_lhs_samples=2000,
    n_mcs_samples_low=5000,
    n_mcs_samples_high=100000
)
```

## Using Different Metamodels

### With GPR Models

```python
# Configuration for GPR models
path_config_gpr = PathConfig(
    model_type='gpr',
    model_dir='RESULT_MODEL_GPR',
    ann_model_dir='RESULT_MODEL_ANN_GPR',
    results_dir='RESULT_PARETO_FRONT_TWOSTAGE_GPR'
)

# Initialize optimizer with GPR
optimizer_gpr = RobustOptimization(
    ann_config=ann_config,
    sampling_config=sampling_config,
    optimization_config=opt_config,
    path_config=path_config_gpr
)

# Run optimization with GPR
pareto_set_gpr, pareto_front_gpr = optimizer_gpr.run()
```

### With Cokriging Models

```python
# Configuration for Cokriging models
path_config_ck = PathConfig(
    model_type='cokriging',
    model_dir='RESULT_MODEL_COKRIGING',
    ann_model_dir='RESULT_MODEL_ANN_CK',
    results_dir='RESULT_PARETO_FRONT_TWOSTAGE_CK'
)

# Initialize optimizer with Cokriging
optimizer_ck = RobustOptimization(
    ann_config=ann_config,
    sampling_config=sampling_config,
    optimization_config=opt_config,
    path_config=path_config_ck
)

# Run optimization with Cokriging
pareto_set_ck, pareto_front_ck = optimizer_ck.run()
```

## Full Pipeline Example

Here's a complete example that demonstrates the entire pipeline with custom configurations:

```python
import numpy as np
from PyEGRO.robustopt.method_nnmcs import (
    RobustOptimization, 
    ANNConfig, 
    SamplingConfig, 
    OptimizationConfig, 
    PathConfig
)

# 1. Define Neural Network Configuration
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

# 2. Define Sampling Configuration
sampling_config = SamplingConfig(
    n_lhs_samples=2000,
    n_mcs_samples_low=10000,
    n_mcs_samples_high=500000,
    random_seed=42
)

# 3. Define Optimization Configuration
opt_config = OptimizationConfig(
    pop_size=50,
    n_gen=100,
    prefer_gpu=True,
    verbose=True,
    random_seed=42,
    metric='hv',
    reference_point=np.array([10.0, 5.0])
)

# 4. Define Paths Configuration
path_config = PathConfig(
    model_type='gpr',
    model_dir='RESULT_MODEL_GPR',
    ann_model_dir='RESULT_MODEL_ANN',
    data_info_path='DATA_PREPARATION/data_info.json',
    results_dir='RESULT_PARETO_FRONT_TWOSTAGE',
    qoi_dir='RESULT_QOI'
)

# 5. Initialize the Optimizer
optimizer = RobustOptimization(
    ann_config=ann_config,
    sampling_config=sampling_config,
    optimization_config=opt_config,
    path_config=path_config
)

# 6. Run the Optimization
print("Starting Two-Stage Robust Optimization Pipeline...")
pareto_set, pareto_front = optimizer.run()

# 7. Process and Display Results
print("\nOptimization Results:")
print("-" * 50)
print(f"Number of Pareto-optimal solutions: {len(pareto_set)}")

# Get the best mean solution
idx_best_mean = np.argmin(pareto_front[:, 0])
print(f"\nBest Mean Solution:")
print(f"  Design Variables: {pareto_set[idx_best_mean]}")
print(f"  Mean: {pareto_front[idx_best_mean, 0]:.4f}")
print(f"  StdDev: {pareto_front[idx_best_mean, 1]:.4f}")

# Get the most robust solution
idx_best_std = np.argmin(pareto_front[:, 1])
print(f"\nMost Robust Solution:")
print(f"  Design Variables: {pareto_set[idx_best_std]}")
print(f"  Mean: {pareto_front[idx_best_std, 0]:.4f}")
print(f"  StdDev: {pareto_front[idx_best_std, 1]:.4f}")

# Get a balanced solution
idx_balanced = np.argmin(0.5 * pareto_front[:, 0] + 0.5 * pareto_front[:, 1])
print(f"\nBalanced Solution:")
print(f"  Design Variables: {pareto_set[idx_balanced]}")
print(f"  Mean: {pareto_front[idx_balanced, 0]:.4f}")
print(f"  StdDev: {pareto_front[idx_balanced, 1]:.4f}")
```

This example demonstrates the complete workflow for the two-stage robust optimization approach, from configuring the pipeline to analyzing the final results. The approach significantly reduces computational time compared to traditional MCS-based robust optimization while maintaining high accuracy in the final solutions.