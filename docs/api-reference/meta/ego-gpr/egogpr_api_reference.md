# PyEGRO.meta.egogpr API Reference

This document provides detailed API documentation for the PyEGRO.meta.egogpr module, a framework for building highly accurate surrogate models through intelligent adaptive sampling strategies.

## Table of Contents

1. [Main Optimizer](#main-optimizer)
2. [Model](#model)
3. [Configuration](#configuration)
4. [Acquisition Functions](#acquisition-functions)
5. [Utilities](#utilities)
6. [Visualization](#visualization)

---

## Main Optimizer

### `EfficientGlobalOptimization`

Main class for performing Efficient Global Optimization.

```python
from PyEGRO.meta.egogpr import EfficientGlobalOptimization
```

#### Constructor

```python
EfficientGlobalOptimization(
    objective_func,
    bounds: np.ndarray,
    variable_names: list,
    config: TrainingConfig = None,
    initial_data: pd.DataFrame = None
)
```

**Parameters:**
- `objective_func` (callable): Function to be optimized
- `bounds` (np.ndarray): Bounds for each variable, shape (n_dimensions, 2)
- `variable_names` (list): Names of input variables
- `config` (TrainingConfig, optional): Configuration for the optimization process
- `initial_data` (pd.DataFrame, optional): Initial data points (if already available)

#### Methods

##### `run()`

Run the optimization process.

```python
history = optimizer.run()
```

**Returns:**
- `dict`: History of the optimization process containing iterations, performance metrics, and chosen points

---

## Model

### `GPRegressionModel`

Gaussian Process Regression Model with configurable kernel.

```python
from PyEGRO.meta.egogpr import GPRegressionModel
```

#### Constructor

```python
GPRegressionModel(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    likelihood: gpytorch.likelihoods.Likelihood,
    kernel: str = 'matern25'
)
```

**Parameters:**
- `train_x` (torch.Tensor): Training input data
- `train_y` (torch.Tensor): Training target data
- `likelihood` (gpytorch.likelihoods.Likelihood): GPyTorch likelihood
- `kernel` (str, optional): Kernel type ('matern25', 'matern15', 'matern05', 'rbf', 'linear')

#### Methods

##### `forward(x: torch.Tensor)`

Forward pass of GP model.

**Parameters:**
- `x` (torch.Tensor): Input data

**Returns:**
- `gpytorch.distributions.MultivariateNormal`: Distribution representing GP predictions

##### `get_hyperparameters()`

Get current hyperparameter values.

**Returns:**
- `dict`: Dictionary containing the hyperparameters

##### `predict(x: torch.Tensor)`

Make predictions with the model.

**Parameters:**
- `x` (torch.Tensor): Input points

**Returns:**
- `tuple`: (mean predictions, standard deviations)

---

## Configuration

### `TrainingConfig`

Configuration class for EGO training parameters.

```python
from PyEGRO.meta.egogpr import TrainingConfig
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
    save_dir: str = "RESULT_MODEL_GPR",
    learning_rate: float = 0.01,
    early_stopping_patience: int = 20,
    jitter: float = 1e-3,
    kernel: str = "matern25"
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
- `kernel` (str): Kernel type for GPR model

#### Methods

##### `to_dict()`

Convert configuration to dictionary.

**Returns:**
- `dict`: Dictionary containing all configuration parameters

---

## Acquisition Functions

### Base Class

#### `AcquisitionFunction`

Base class for all acquisition functions.

```python
from egogpr.acquisition import AcquisitionFunction
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

### Specific Acquisition Functions

#### `ExpectedImprovement`

Expected Improvement acquisition function.

```python
from PyEGRO.meta.egogpr import ExpectedImprovement
```

#### `LowerConfidenceBound`

Lower Confidence Bound acquisition function.

```python
from PyEGRO.meta.egogpr import LowerConfidenceBound
```

#### `PredictiveVariance`

Predictive Variance acquisition function.

```python
from PyEGRO.meta.egogpr import PredictiveVariance
```

#### `ProbabilityImprovement`

Probability of Improvement acquisition function.

```python
from PyEGRO.meta.egogpr import ProbabilityImprovement
```

#### `ExpectedImprovementGlobalFit`

Expected Improvement for Global Fit acquisition function.

```python
from PyEGRO.meta.egogpr import ExpectedImprovementGlobalFit
```

#### `Criterion3`

Implementation of Criterion 3 acquisition function.

```python
from PyEGRO.meta.egogpr import Criterion3
```

#### `ExplorationEnhancedEI`

Exploration Enhanced Expected Improvement (E³I) acquisition function.

```python
from PyEGRO.meta.egogpr import ExplorationEnhancedEI
```

### Factory Functions

#### `create_acquisition_function`

Create an acquisition function by name.

```python
from PyEGRO.meta.egogpr import create_acquisition_function

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
from PyEGRO.meta.egogpr import propose_location

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

#### `train_gp_model`

Train a GP model with optimization settings.

```python
from PyEGRO.meta.egogpr.utils import train_gp_model

train_gp_model(
    model,
    likelihood,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainingConfig
)
```

### Model Storage Utilities

#### `save_model_data`

Save model, scalers, hyperparameters and metadata.

```python
from PyEGRO.meta.egogpr import save_model_data

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
from PyEGRO.meta.egogpr import load_model_data

data = load_model_data(save_dir: str)
```

### Evaluation Utilities

#### `evaluate_model_performance`

Calculate model performance metrics using LOOCV.

```python
from PyEGRO.meta.egogpr.utils import evaluate_model_performance

metrics = evaluate_model_performance(
    model,
    likelihood,
    X_train: torch.Tensor,
    y_train: torch.Tensor
)
```

### Directory Utilities

#### `setup_directories`

Create necessary directories for saving results.

```python
from PyEGRO.meta.egogpr import setup_directories

setup_directories(save_dir: str, create_plots: bool = True)
```

---

## Visualization

### `EGOAnimator`

Class for creating animations of the optimization process.

```python
from PyEGRO.meta.egogpr import EGOAnimator
```

#### Constructor

```python
EGOAnimator(
    save_dir: str = 'RESULT_MODEL_GPR/animation',
    frame_duration: int = 500
)
```

#### Methods

##### `save_1D_frame`

Save a 1D visualization frame.

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
    batch_size: int = 1000
)
```

##### `save_2D_frame`

Save a 2D visualization frame.

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
    device: torch.device = None
)
```

##### `create_gif`

Create GIF animation from saved frames.

```python
animator.create_gif(duration: Optional[int] = None)
```

### `ModelVisualizer`

Class for creating final model visualizations.

```python
from PyEGRO.meta.egogpr import ModelVisualizer
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
    save_dir='RESULT_MODEL_GPR'
)
```

#### Methods

##### `plot_final_prediction`

Create final prediction plots.

```python
visualizer.plot_final_prediction(
    X_train,
    y_train,
    true_function=None
)
```

##### `plot_convergence_metrics`

Plot optimization metrics over iterations.

```python
visualizer.plot_convergence_metrics(history: dict)
```

##### `plot_error_analysis`

Plot prediction errors and residuals.

```python
visualizer.plot_error_analysis(X_train, y_train)
```