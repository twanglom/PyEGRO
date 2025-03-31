# PyEGRO.meta.egocokriging API Reference

This document provides detailed API documentation for the PyEGRO.meta.egocokriging module, a framework for multi-fidelity surrogate modeling and optimization using Co-Kriging.

## Table of Contents

1. [Main Optimizer](#main-optimizer)
2. [Multi-Fidelity Model](#multi-fidelity-model)
3. [Configuration](#configuration)
4. [Acquisition Functions](#acquisition-functions)
5. [Utilities](#utilities)
6. [Visualization](#visualization)

---

## Main Optimizer

### `EfficientGlobalOptimization`

Main class for Efficient Global Optimization with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import EfficientGlobalOptimization
```

#### Constructor

```python
EfficientGlobalOptimization(
    objective_func,
    bounds: np.ndarray,
    variable_names: list,
    config: TrainingConfig = None,
    initial_data: pd.DataFrame = None,
    low_fidelity_func = None
)
```

**Parameters:**
- `objective_func` (callable): High-fidelity objective function to be optimized
- `bounds` (np.ndarray): Bounds for each variable, shape (n_dimensions, 2)
- `variable_names` (list): Names of input variables
- `config` (TrainingConfig, optional): Configuration for the optimization process
- `initial_data` (pd.DataFrame, optional): Initial data points with optional 'fidelity' column
- `low_fidelity_func` (callable, optional): Low-fidelity objective function (faster but less accurate)

#### Methods

##### `run()`

Run the optimization process with multi-fidelity support.

```python
history = optimizer.run()
```

**Returns:**
- `dict`: History of the optimization process containing iterations, performance metrics, rho values, and chosen points

---

## Multi-Fidelity Model

### `CoKrigingKernel`

Co-Kriging kernel for multi-fidelity Gaussian process modeling.

```python
from PyEGRO.meta.egocokriging import CoKrigingKernel
```

#### Constructor

```python
CoKrigingKernel(
    base_kernel,
    num_dims: int,
    active_dims: Optional[Tuple[int, ...]] = None
)
```

**Parameters:**
- `base_kernel` (gpytorch.kernels.Kernel): Base kernel to use for both kernel_c and kernel_d
- `num_dims` (int): Number of input dimensions
- `active_dims` (Tuple[int, ...], optional): Dimensions to apply kernel to

#### Properties

##### `rho`

The scaling parameter that controls correlation between fidelity levels.

**Returns:**
- `torch.Tensor`: Transformed rho parameter value

#### Methods

##### `forward(x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params)`

Forward pass to compute kernel matrix.

**Parameters:**
- `x1` (torch.Tensor): First input tensor with fidelity indicator in last column
- `x2` (torch.Tensor): Second input tensor with fidelity indicator in last column
- `diag` (bool, optional): Whether to return only diagonal elements

**Returns:**
- `torch.Tensor`: Kernel matrix

### `CoKrigingModel`

Co-Kriging Gaussian Process Model for multi-fidelity optimization.

```python
from PyEGRO.meta.egocokriging import CoKrigingModel
```

#### Constructor

```python
CoKrigingModel(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    likelihood: gpytorch.likelihoods.Likelihood,
    kernel: str = 'matern15'
)
```

**Parameters:**
- `train_x` (torch.Tensor): Training input data with fidelity indicator in last column
- `train_y` (torch.Tensor): Training target data
- `likelihood` (gpytorch.likelihoods.Likelihood): GPyTorch likelihood
- `kernel` (str, optional): Kernel type ('matern25', 'matern15', 'matern05', 'rbf')

#### Methods

##### `forward(x: torch.Tensor)`

Forward pass of Co-Kriging model.

**Parameters:**
- `x` (torch.Tensor): Input data with fidelity indicator in last column

**Returns:**
- `gpytorch.distributions.MultivariateNormal`: Distribution representing GP predictions

##### `get_hyperparameters()`

Get current hyperparameter values including both kernel components and rho.

**Returns:**
- `dict`: Dictionary containing the hyperparameters

##### `predict(x: torch.Tensor)`

Make predictions with the Co-Kriging model.

**Parameters:**
- `x` (torch.Tensor): Input points with fidelity indicator in last column

**Returns:**
- `tuple`: (mean predictions, standard deviations)

---

## Configuration

### `TrainingConfig`

Configuration class for EGO training parameters with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import TrainingConfig
```

#### Constructor

```python
TrainingConfig(
    max_iterations: int = 100,
    rmse_threshold: float = 0.001,
    rmse_patience: int = 10,
    relative_improvement: float = 0.01,
    acquisition_name: str = "ei",
    acquisition_params: Dict[str, Any] = field(default_factory=dict),
    training_iter: int = 100,
    verbose: bool = False,
    show_summary: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "RESULT_MODEL_COKRIGING",
    learning_rate: float = 0.01,
    early_stopping_patience: int = 20,
    jitter: float = 1e-3,
    kernel: str = "matern15",
    multi_fidelity: bool = True,
    rho_threshold: float = 0.05,
    rho_patience: int = 3
)
```

**Parameters:**
- `max_iterations` (int): Maximum number of optimization iterations
- `rmse_threshold` (float): RMSE threshold for early stopping
- `rmse_patience` (int): Number of iterations without improvement before stopping
- `relative_improvement` (float): Minimum relative improvement required
- `acquisition_name` (str): Name of acquisition function
- `acquisition_params` (Dict[str, Any]): Parameters for the acquisition function
- `training_iter` (int): Number of training iterations for GP model
- `verbose` (bool): Whether to print detailed training information
- `show_summary` (bool): Whether to show parameter summary before training
- `device` (str): Device to use for training ('cpu' or 'cuda')
- `save_dir` (str): Directory to save results and models
- `learning_rate` (float): Learning rate for model optimization
- `early_stopping_patience` (int): Patience for early stopping during model training
- `jitter` (float): Jitter value for numerical stability
- `kernel` (str): Kernel type for base kernel
- `multi_fidelity` (bool): Whether to use multi-fidelity modeling
- `rho_threshold` (float): Convergence threshold for rho parameter (as percentage)
- `rho_patience` (int): Number of iterations without improvement in rho before stopping

#### Methods

##### `to_dict()`

Convert configuration to dictionary.

**Returns:**
- `dict`: Dictionary containing all configuration parameters

---

## Acquisition Functions

### Base Class

#### `AcquisitionFunction`

Base class for all acquisition functions with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging.acquisition import AcquisitionFunction
```

#### Methods

##### `evaluate(X)`

Evaluate the acquisition function at points X.

**Parameters:**
- `X` (np.ndarray): Points to evaluate

**Returns:**
- `np.ndarray`: Acquisition function values (negated for minimization)

##### `optimize()`

Optimize the acquisition function to find the next sampling point.

**Returns:**
- `np.ndarray`: Next point to evaluate

##### `_prepare_input_with_fidelity(X, fidelity=1)`

Add fidelity indicator to input if the model is multi-fidelity.

**Parameters:**
- `X` (torch.Tensor): Input data without fidelity indicator
- `fidelity` (int): Fidelity level (0 for low, 1 for high)

**Returns:**
- `torch.Tensor`: Input data suitable for the model

### Specific Acquisition Functions

#### `ExpectedImprovement`

Expected Improvement acquisition function with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import ExpectedImprovement
```

#### `LowerConfidenceBound`

Lower Confidence Bound acquisition function with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import LowerConfidenceBound
```

#### `PredictiveVariance`

Predictive Variance acquisition function with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import PredictiveVariance
```

#### `ProbabilityImprovement`

Probability of Improvement acquisition function with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import ProbabilityImprovement
```

#### `ExpectedImprovementGlobalFit`

Expected Improvement for Global Fit acquisition function with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import ExpectedImprovementGlobalFit
```

#### `Criterion3`

Implementation of Criterion 3 acquisition function with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import Criterion3
```

#### `ExplorationEnhancedEI`

Exploration Enhanced Expected Improvement (E³I) acquisition function with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import ExplorationEnhancedEI
```

### Factory Functions

#### `create_acquisition_function`

Create an acquisition function by name with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import create_acquisition_function

acquisition = create_acquisition_function(
    name: str,
    model,
    likelihood,
    bounds,
    scaler_x,
    scaler_y,
    y_train=None,
    **kwargs
)
```

#### `propose_location`

Find the next location to sample using a specified acquisition function.

```python
from PyEGRO.meta.egocokriging import propose_location

X_next = propose_location(
    acquisition_name: str,
    model,
    likelihood,
    y_train,
    bounds,
    scaler_x,
    scaler_y,
    **kwargs
)
```

---

## Utilities

### Training Utilities

#### `train_ck_model`

Train a Co-Kriging model with optimization settings.

```python
from PyEGRO.meta.egocokriging import train_ck_model

rho_value = train_ck_model(
    model,
    likelihood,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainingConfig
)
```

**Returns:**
- `float`: Current rho value after training

#### `check_rho_convergence`

Check if rho parameter has converged.

```python
from PyEGRO.meta.egocokriging import check_rho_convergence

has_converged = check_rho_convergence(
    rho_history: List[float],
    config: TrainingConfig
)
```

**Returns:**
- `bool`: True if rho has converged, False otherwise

### Model Storage Utilities

#### `save_model_data`

Save model, scalers, hyperparameters and metadata with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import save_model_data

save_model_data(
    model,
    likelihood,
    scalers: Dict,
    metadata: Dict,
    save_dir: str
)
```

#### `load_model_data`

Load saved model data.

```python
from PyEGRO.meta.egocokriging import load_model_data

data = load_model_data(save_dir: str)
```

### Evaluation Utilities

#### `evaluate_model_performance`

Calculate model performance metrics using LOOCV.

```python
from PyEGRO.meta.egocokriging.utils import evaluate_model_performance

metrics = evaluate_model_performance(
    model,
    likelihood,
    X_train: torch.Tensor,
    y_train: torch.Tensor
)
```

### Data Preparation Utilities

#### `prepare_data_with_fidelity`

Prepare input data with fidelity indicator.

```python
from PyEGRO.meta.egocokriging import prepare_data_with_fidelity

X_with_fidelity = prepare_data_with_fidelity(
    X: np.ndarray,
    fidelity: int
)
```

**Parameters:**
- `X` (np.ndarray): Input data without fidelity indicator
- `fidelity` (int): Fidelity level (0 for low, 1 for high)

**Returns:**
- `np.ndarray`: Input data with fidelity indicator in the last column

### Directory Utilities

#### `setup_directories`

Create necessary directories for saving results.

```python
from PyEGRO.meta.egocokriging import setup_directories

setup_directories(save_dir: str, create_plots: bool = True)
```

---

## Visualization

### `EGOAnimator`

Class for creating animations of the optimization process with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import EGOAnimator
```

#### Constructor

```python
EGOAnimator(
    save_dir: str = 'RESULT_MODEL_COKRIGING/animation',
    frame_duration: int = 500
)
```

#### Methods

##### `save_1D_frame`

Save a 1D visualization frame with multi-fidelity support.

```python
animator.save_1D_frame(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler_x: object,
    scaler_y: object,
    bounds: np.ndarray,
    iteration: int,
    variable_names: List[str],
    device: torch.device = None,
    true_function: Optional[callable] = None,
    batch_size: int = 1000,
    fidelities: Optional[np.ndarray] = None
)
```

##### `save_2D_frame`

Save a 2D visualization frame with multi-fidelity support.

```python
animator.save_2D_frame(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler_x: object,
    scaler_y: object,
    bounds: np.ndarray,
    iteration: int,
    variable_names: List[str],
    n_initial_samples: int,
    n_points: int = 50,
    batch_size: int = 1000,
    device: torch.device = None,
    fidelities: Optional[np.ndarray] = None
)
```

##### `create_gif`

Create GIF animation from saved frames.

```python
animator.create_gif(duration: Optional[int] = None)
```

### `ModelVisualizer`

Class for creating final model visualizations with multi-fidelity support.

```python
from PyEGRO.meta.egocokriging import ModelVisualizer
```

#### Constructor

```python
ModelVisualizer(
    model,
    likelihood,
    scaler_x,
    scaler_y,
    bounds,
    variable_names,
    history,
    device,
    save_dir='RESULT_MODEL_COKRIGING',
    multi_fidelity=False,
    rho_history=None
)
```

#### Methods

##### `plot_rho_evolution`

Plot the evolution of rho parameter over iterations.

```python
visualizer.plot_rho_evolution()
```

##### `plot_final_prediction`

Create final prediction plots with multi-fidelity support.

```python
visualizer.plot_final_prediction(
    X_train,
    y_train,
    true_function=None,
    fidelities=None
)
```

##### `plot_convergence_metrics`

Plot optimization metrics over iterations including rho values.

```python
visualizer.plot_convergence_metrics(history: dict)
```

##### `plot_error_analysis`

Plot prediction errors and residuals with separate plots for high and low fidelity data.

```python
visualizer.plot_error_analysis(
    X_train,
    y_train,
    fidelities=None
)
```