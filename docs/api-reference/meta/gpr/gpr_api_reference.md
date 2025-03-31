# PyEGRO GPR Module API Reference

This document provides detailed API reference for the Gaussian Process Regression (GPR) module in the PyEGRO package.

## Table of Contents
- [MetaTraining Class](#metatraining-class)
- [GPRegressionModel Class](#gpregressionmodel-class)
- [DeviceAgnosticGPR Class](#deviceagnosticgpr-class)
- [Visualization Functions](#visualization-functions)

## MetaTraining Class

The `MetaTraining` class provides a high-level interface for training and managing Gaussian Process Regression models.

### Constructor

```python
MetaTraining(
    test_size=0.3,
    num_iterations=1000,
    prefer_gpu=True,
    show_progress=True,
    show_hardware_info=True,
    show_model_info=True,
    output_dir='RESULT_MODEL_GPR',
    data_dir='DATA_PREPARATION',
    data_info_file=None,
    data_training_file=None,
    kernel='matern15',
    learning_rate=0.01,
    patience=50
)
```

#### Parameters

- `test_size` (float, optional): Fraction of data to use for testing if no test data is provided. Default: `0.3`
- `num_iterations` (int, optional): Number of training iterations. Default: `1000`
- `prefer_gpu` (bool, optional): Whether to use GPU if available. Default: `True`
- `show_progress` (bool, optional): Whether to show detailed progress. Default: `True`
- `show_hardware_info` (bool, optional): Whether to show system hardware info. Default: `True`
- `show_model_info` (bool, optional): Whether to show model architecture info. Default: `True`
- `output_dir` (str, optional): Directory for saving results. Default: `'RESULT_MODEL_GPR'`
- `data_dir` (str, optional): Directory containing input data. Default: `'DATA_PREPARATION'`
- `data_info_file` (str, optional): Path to data info JSON file. Default: `None`
- `data_training_file` (str, optional): Path to training data CSV file. Default: `None`
- `kernel` (str, optional): Kernel to use for GPR model. Options: `'matern25'`, `'matern15'`, `'matern05'`, `'rbf'`, `'linear'`. Default: `'matern15'`
- `learning_rate` (float, optional): Learning rate for optimizer. Default: `0.01`
- `patience` (int, optional): Number of iterations to wait for improvement before early stopping. Default: `50`

### Methods

#### train

Train the GPR model with more flexible data options.

```python
train(X=None, y=None, X_test=None, y_test=None, feature_names=None, custom_data=False)
```

##### Parameters

- `X` (numpy.ndarray, pandas.DataFrame, or str, optional): Training features or path to training data CSV. Default: `None`
- `y` (numpy.ndarray, pandas.DataFrame, or pandas.Series, optional): Training targets (only needed if custom_data=True). Default: `None`
- `X_test` (numpy.ndarray or pandas.DataFrame, optional): Test features. Default: `None`
- `y_test` (numpy.ndarray, pandas.DataFrame, or pandas.Series, optional): Test targets. Default: `None`
- `feature_names` (list of str, optional): List of feature names. Default: `None`
- `custom_data` (bool, optional): Whether using custom data instead of loading from files. Default: `False`

##### Returns

- `model` (GPRegressionModel): Trained model
- `scaler_X` (StandardScaler): Feature scaler
- `scaler_y` (StandardScaler): Target scaler

#### predict

Make predictions using the trained model.

```python
predict(X)
```

##### Parameters

- `X` (numpy.ndarray or pandas.DataFrame): Input features

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

- `model` (GPRegressionModel): Loaded model

#### print_hyperparameters

Print the learned hyperparameters of the model.

```python
print_hyperparameters()
```

## GPRegressionModel Class

The `GPRegressionModel` class implements a Gaussian Process Regression model with configurable kernels.

### Constructor

```python
GPRegressionModel(train_x, train_y, likelihood, kernel='matern15')
```

#### Parameters

- `train_x` (torch.Tensor): Training input data
- `train_y` (torch.Tensor): Training target data
- `likelihood` (gpytorch.likelihoods.Likelihood): GP likelihood function
- `kernel` (str, optional): Kernel type. Options: `'matern25'`, `'matern15'`, `'matern05'`, `'rbf'`, `'linear'`. Default: `'matern15'`

### Methods

#### forward

Computes the mean and covariance of the GP posterior.

```python
forward(x)
```

##### Parameters

- `x` (torch.Tensor): Input data

##### Returns

- `gpytorch.distributions.MultivariateNormal`: Distribution with predicted mean and covariance

## DeviceAgnosticGPR Class

The `DeviceAgnosticGPR` class provides a device-agnostic handler for GPR models, making it easier to use models on both CPU and GPU.

### Constructor

```python
DeviceAgnosticGPR(prefer_gpu=False)
```

#### Parameters

- `prefer_gpu` (bool, optional): Whether to use GPU if available. Default: `False`

### Methods

#### load_model

Load model using state dict approach.

```python
load_model(model_dir='RESULT_MODEL_GPR')
```

##### Parameters

- `model_dir` (str, optional): Directory containing the model files. Default: `'RESULT_MODEL_GPR'`

##### Returns

- `bool`: True if model was loaded successfully, False otherwise

#### predict

Make predictions with the loaded GPR model.

```python
predict(X, batch_size=1000)
```

##### Parameters

- `X` (numpy.ndarray): Input features (n_samples, n_features)
- `batch_size` (int, optional): Batch size for processing large datasets. Default: `1000`

##### Returns

- Tuple of `(mean_predictions, std_predictions)` as numpy arrays

## Visualization Functions

### visualize_gpr

Create comprehensive visualizations for GPR model performance.

```python
visualize_gpr(
    meta,
    X_train,
    y_train,
    X_test=None,
    y_test=None, 
    variable_names=None,
    bounds=None,
    savefig=False,
    output_dir=None
)
```

#### Parameters

- `meta` (MetaTraining): Trained MetaTraining instance
- `X_train` (numpy.ndarray or pandas.DataFrame): Training inputs
- `y_train` (numpy.ndarray, pandas.DataFrame, or pandas.Series): Training targets
- `X_test` (numpy.ndarray or pandas.DataFrame, optional): Test inputs. Default: `None`
- `y_test` (numpy.ndarray, pandas.DataFrame, or pandas.Series, optional): Test targets. Default: `None`
- `variable_names` (list of str, optional): Names of input variables. Default: `None`
- `bounds` (numpy.ndarray, optional): Bounds of input variables for sampling. Default: `None`
- `savefig` (bool, optional): Whether to save figures to disk. Default: `False`
- `output_dir` (str, optional): Directory to save figures. Default: `None`, uses meta.output_dir

#### Returns

- `figures` (dict): Dictionary of figure handles