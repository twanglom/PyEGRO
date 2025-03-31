# PyEGRO ModelTesting API Reference

This document provides detailed API reference for the Model Testing module in the PyEGRO package, which allows evaluation of trained GPR and Co-Kriging models on unseen data.

## Table of Contents
- [ModelTester Class](#modeltester-class)
- [Utility Functions](#utility-functions)

## ModelTester Class

The `ModelTester` class provides functionality for loading trained models and evaluating their performance on test data.

### Constructor

```python
ModelTester(
    model_dir='RESULT_MODEL_GPR',
    model_name=None,
    model_path=None,
    logger=None
)
```

#### Parameters

- `model_dir` (str, optional): Directory containing trained model files. Default: `'RESULT_MODEL_GPR'`
- `model_name` (str, optional): Base name of the model file without extension. Default: `None` (will be inferred from directory or file path)
- `model_path` (str, optional): Direct path to the model file. Default: `None`
- `logger` (logging.Logger, optional): Logger object for logging messages. Default: `None`

### Methods

#### load_model

Load the trained model and scalers from disk.

```python
load_model()
```

##### Returns

- `self`: Returns the ModelTester instance for method chaining

#### load_test_data

Load test data from CSV file or generate synthetic test data.

```python
load_test_data(data_path=None, feature_cols=None, target_col='y', n_samples=100, n_features=2)
```

##### Parameters

- `data_path` (str, optional): Path to the CSV file containing test data. Default: `None`
- `feature_cols` (list of str, optional): List of feature column names. Default: `None` (all columns except target)
- `target_col` (str, optional): Target column name. Default: `'y'`
- `n_samples` (int, optional): Number of samples for synthetic data if no data_path provided. Default: `100`
- `n_features` (int, optional): Number of features for synthetic data if no data_path provided. Default: `2`

##### Returns

- Tuple of `(X_test, y_test)`: Test features and targets as numpy arrays

#### evaluate

Evaluate the model performance on test data.

```python
evaluate(X_test, y_test)
```

##### Parameters

- `X_test` (numpy.ndarray): Test features
- `y_test` (numpy.ndarray): Test targets

##### Returns

- `self`: Returns the ModelTester instance for method chaining with updated test_results property

#### save_results

Save test results and generate plots.

```python
save_results(output_dir='test_results')
```

##### Parameters

- `output_dir` (str, optional): Directory to save results. Default: `'test_results'`

##### Returns

- `self`: Returns the ModelTester instance for method chaining

#### plot_results

Generate and optionally save plots of test results.

```python
plot_results(output_dir='test_results', show_plots=True, save_plots=True, smooth=True, smooth_window=11)
```

##### Parameters

- `output_dir` (str, optional): Directory to save plots. Default: `'test_results'`
- `show_plots` (bool, optional): Whether to display plots. Default: `True`
- `save_plots` (bool, optional): Whether to save plots to disk. Default: `True`
- `smooth` (bool, optional): Whether to apply smoothing to uncertainty plots. Default: `True`
- `smooth_window` (int, optional): Window size for smoothing filter. Default: `11`

##### Returns

- `figures` (dict): Dictionary of matplotlib figure objects

### Properties

- `test_results` (dict): Contains test metrics and predictions after running `evaluate`. Keys include:
  - `r2`: R² score
  - `mse`: Mean squared error
  - `rmse`: Root mean squared error
  - `mae`: Mean absolute error
  - `y_test`: Test targets
  - `y_pred`: Model predictions
  - `std_dev`: Standard deviations of predictions
  - `model_type`: Type of model ('gpr' or 'cokriging')

## Utility Functions

### load_and_test_model

A convenience function to load a model and test it in one call.

```python
load_and_test_model(
    data_path=None,
    model_dir=None,
    model_name=None,
    model_path=None,
    output_dir='test_results',
    feature_cols=None,
    target_col='y',
    logger=None,
    show_plots=True,
    smooth=True,
    smooth_window=11
)
```

#### Parameters

- `data_path` (str, optional): Path to the CSV file containing test data. Default: `None`
- `model_dir` (str, optional): Directory containing trained model files. Default: `None`
- `model_name` (str, optional): Base name of the model file without extension. Default: `None`
- `model_path` (str, optional): Direct path to the model file. Default: `None`
- `output_dir` (str, optional): Directory to save results. Default: `'test_results'`
- `feature_cols` (list of str, optional): List of feature column names. Default: `None`
- `target_col` (str, optional): Target column name. Default: `'y'`
- `logger` (logging.Logger, optional): Logger object for logging messages. Default: `None`
- `show_plots` (bool, optional): Whether to display plots. Default: `True`
- `smooth` (bool, optional): Whether to apply smoothing to uncertainty plots. Default: `True`
- `smooth_window` (int, optional): Window size for smoothing filter. Default: `11`

#### Returns

- `test_results` (dict): Dictionary containing test metrics and predictions