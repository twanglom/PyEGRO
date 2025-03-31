# PyEGRO Co-Kriging Module API Reference

This document provides detailed API reference for the Co-Kriging (multi-fidelity) module in the PyEGRO package.

## Table of Contents
- [MetaTrainingCoKriging Class](#metatrainingcokriging-class)
- [CoKrigingModel Class](#cokrigingmodel-class)
- [CoKrigingKernel Class](#cokrigingkernel-class)
- [DeviceAgnosticCoKriging Class](#deviceagnosticcokriging-class)
- [Visualization Functions](#visualization-functions)

## MetaTrainingCoKriging Class

The `MetaTrainingCoKriging` class provides a high-level interface for training and managing Co-Kriging models that combine low and high-fidelity data sources.

### Constructor

```python
MetaTrainingCoKriging(
    num_iterations=1000,
    prefer_gpu=True,
    show_progress=True,
    show_hardware_info=True,
    show_model_info=True,
    output_dir='RESULT_MODEL_COKRIGING',
    kernel='matern25',
    learning_rate=0.01,
    patience=50
)
```

#### Parameters

- `num_iterations` (int, optional): Number of training iterations. Default: `1000`
- `prefer_gpu` (bool, optional): Whether to use GPU if available. Default: `True`
- `show_progress` (bool, optional): Whether to show detailed progress. Default: `True`
- `show_hardware_info` (bool, optional): Whether to show system hardware info. Default: `True`
- `show_model_info` (bool, optional): Whether to show model architecture info. Default: `True`
- `output_dir` (str, optional): Directory for saving results. Default: `'RESULT_MODEL_COKRIGING'`
- `kernel` (str, optional): Kernel to use for base GP model. Options: `'matern25'`, `'matern15'`, `'matern05'`, `'rbf'`. Default: `'matern25'`
- `learning_rate` (float, optional): Learning rate for optimizer. Default: `0.01`
- `patience` (int, optional): Number of iterations to wait for improvement before early stopping. Default: `50`

### Methods

#### train

Train the Co-Kriging model with low and high-fidelity data.

```python
train(X_low, y_low, X_high, y_high, X_test=None, y_test=None, feature_names=None)
```

##### Parameters

- `X_low` (numpy.ndarray or pandas.DataFrame): Low-fidelity input features
- `y_low` (numpy.ndarray, pandas.DataFrame, or pandas.Series): Low-fidelity target values
- `X_high` (numpy.ndarray or pandas.DataFrame): High-fidelity input features
- `y_high` (numpy.ndarray, pandas.DataFrame, or pandas.Series): High-fidelity target values
- `X_test` (numpy.ndarray or pandas.DataFrame, optional): Test features. Default: `None`
- `y_test` (numpy.ndarray, pandas.DataFrame, or pandas.Series, optional): Test targets. Default: `None`
- `feature_names` (list of str, optional): List of feature names. Default: `None`

##### Returns

- `model` (CoKrigingModel): Trained model
- `scaler_X` (StandardScaler): Feature scaler
- `scaler_y` (StandardScaler): Target scaler

#### predict

Make predictions with the trained Co-Kriging model.

```python
predict(X, fidelity='high')
```

##### Parameters

- `X` (numpy.ndarray or pandas.DataFrame): Input features for prediction
- `fidelity` (str, optional): 'high' or 'low' to specify which fidelity level to predict. Default: `'high'`

##### Returns

- `mean` (numpy.ndarray): Mean predictions
- `std` (numpy.ndarray): Standard deviations of predictions

#### load_model

Load a trained model from disk.

```python
load_model(model_path=None)
```

##### Parameters

- `model_path` (str, optional): Path to the saved model file. If None, uses the default path. Default: `None`

##### Returns

- `model` (CoKrigingModel): Loaded model

#### print_hyperparameters

Print the learned hyperparameters of the model.

```python
print_hyperparameters()
```

## CoKrigingModel Class

The `CoKrigingModel` class implements a Gaussian Process model for Co-Kriging/multi-fidelity modeling using the Kennedy & O'Hagan (2000) approach.

### Constructor

```python
CoKrigingModel(train_x, train_y, likelihood, kernel='matern15')
```

#### Parameters

- `train_x` (torch.Tensor): Training input data (with fidelity indicator as last column)
- `train_y` (torch.Tensor): Training target data
- `likelihood` (gpytorch.likelihoods.Likelihood): GP likelihood function
- `kernel` (str, optional): Kernel type. Options: `'matern25'`, `'matern15'`, `'matern05'`, `'rbf'`. Default: `'matern15'`

### Methods

#### forward

Computes the mean and covariance of the GP posterior.

```python
forward(x)
```

##### Parameters

- `x` (torch.Tensor): Input data (with fidelity indicator as last column)

##### Returns

- `gpytorch.distributions.MultivariateNormal`: Distribution with predicted mean and covariance

## CoKrigingKernel Class

The `CoKrigingKernel` class implements a custom kernel for Co-Kriging that handles multi-fidelity data following the auto-regressive structure of the Kennedy & O'Hagan model.

### Constructor

```python
CoKrigingKernel(base_kernel, num_dims, active_dims=None)
```

#### Parameters

- `base_kernel` (gpytorch.kernels.Kernel): Base kernel to use for both low-fidelity and difference components
- `num_dims` (int): Number of dimensions including the fidelity indicator
- `active_dims` (tuple, optional): Dimensions that the kernel operates on. Default: `None`

### Properties

#### rho

Returns the scaling parameter ρ that controls the correlation between fidelity levels.

```python
rho
```

### Methods

#### forward

Compute the covariance matrix between inputs x1 and x2 based on the Kennedy & O'Hagan multi-fidelity formulation.

```python
forward(x1, x2, diag=False, **params)
```

##### Parameters

- `x1` (torch.Tensor): First input
- `x2` (torch.Tensor): Second input
- `diag` (bool, optional): Whether to return just the diagonal of the covariance matrix. Default: `False`
- `**params`: Additional parameters for the base kernels

##### Returns

- `torch.Tensor`: Covariance matrix

## DeviceAgnosticCoKriging Class

The `DeviceAgnosticCoKriging` class provides a device-agnostic handler for Co-Kriging models.

### Constructor

```python
DeviceAgnosticCoKriging(prefer_gpu=False)
```

#### Parameters

- `prefer_gpu` (bool, optional): Whether to use GPU if available. Default: `False`

### Methods

#### load_model

Load model using state dict approach.

```python
load_model(model_dir='RESULT_MODEL_COKRIGING')
```

##### Parameters

- `model_dir` (str, optional): Directory containing the model files. Default: `'RESULT_MODEL_COKRIGING'`

##### Returns

- `bool`: True if model was loaded successfully, False otherwise

#### predict

Make predictions with the loaded Co-Kriging model.

```python
predict(X, fidelity='high', batch_size=1000)
```

##### Parameters

- `X` (numpy.ndarray): Input features (n_samples, n_features)
- `fidelity` (str, optional): 'high' or 'low' to specify which fidelity level to predict. Default: `'high'`
- `batch_size` (int, optional): Batch size for processing large datasets. Default: `1000`

##### Returns

- Tuple of `(mean_predictions, std_predictions)` as numpy arrays

## Visualization Functions

### visualize_cokriging

Create comprehensive visualizations for Co-Kriging model performance.

```python
visualize_cokriging(
    meta,
    X_low,
    y_low,
    X_high,
    y_high,
    X_test=None,
    y_test=None, 
    variable_names=None,
    bounds=None,
    savefig=False,
    output_dir=None
)
```

#### Parameters

- `meta` (MetaTrainingCoKriging): Trained MetaTrainingCoKriging instance
- `X_low` (numpy.ndarray or pandas.DataFrame): Low-fidelity inputs
- `y_low` (numpy.ndarray, pandas.DataFrame, or pandas.Series): Low-fidelity targets
- `X_high` (numpy.ndarray or pandas.DataFrame): High-fidelity inputs
- `y_high` (numpy.ndarray, pandas.DataFrame, or pandas.Series): High-fidelity targets
- `X_test` (numpy.ndarray or pandas.DataFrame, optional): Test inputs. Default: `None`
- `y_test` (numpy.ndarray, pandas.DataFrame, or pandas.Series, optional): Test targets. Default: `None`
- `variable_names` (list of str, optional): Names of input variables. Default: `None`
- `bounds` (numpy.ndarray, optional): Bounds of input variables for sampling. Default: `None`
- `savefig` (bool, optional): Whether to save figures to disk. Default: `False`
- `output_dir` (str, optional): Directory to save figures. Default: `None`, uses meta.output_dir

#### Returns

- `figures` (dict): Dictionary of figure handles