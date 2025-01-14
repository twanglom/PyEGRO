<figure markdown>
  ![PyEGRO Logo](assets/logo.png){ width="400" }
</figure>

# PyEGRO: Python Efficient Global Robust Optimization

## Overview
**PyEGRO** is a Python library designed for solving complex engineering problems with efficient global robust optimization. It provides tools for initial design sampling, surrogate modeling, sensitivity analysis, and robust optimization.

## 📚 Features

### Design of Experiments 
- Advanced sampling methods like Latin Hypercube, Sobol, and Halton
- Support for design and environmental variables
- Customizable sampling criteria
- Multi-dimensional support

### Efficient Global Optimization
- Multiple acquisition functions available:
   -- Expected Improvement (EI) and Boosting the Exploration term (𝜁-EI)
   -- Probability of Improvement (PI)
   -- Lower Confidence Bound (LCB)
   -- E3I (Exploration Enhanced Expected Improvement )
   -- EIGF (Expected Improvement for Global Fit)
   -- CRI3 (Distance-Enhanced Gradient)

- Comprehensive training configuration
- Built-in visualization tools
- Parallel evaluation support

### Surrogate Model Training
- Gaussian Process Regression (GPR) training
- Hardware optimization (CPU/GPU)
- Progress tracking and visualization
- Hyperparameter optimization

### Robust Optimization
- Monte Carlo Simulation (MCS)
- Polynomial Chaos Expansion (PCE)
- Support for both direct and surrogate evaluation
- Multi-objective Pareto solution

### Sensitivity Analysis
- Sobol indices calculation
- Support for true functions and surrogates
- Analysis and visualization

### Uncertainty Quantification
- Uncertainty Propagation
- Distribution Analysis
- Moment estimation
- PDF/CDF estimation
