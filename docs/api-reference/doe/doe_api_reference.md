# PyEGRO DOE Module API Reference

## Overview

The Design of Experiments (DOE) module in PyEGRO provides tools for generating initial design samples with support for multiple sampling methods and variable types. This reference details the classes, methods, and parameters available in the module.

## Classes

### `InitialDesign`

The main class for creating and managing design samples.

#### Constructor

```python
InitialDesign(
    output_dir='DATA_PREPARATION',
    show_progress=True,
    show_result=True,
    sampling_method='lhs',
    sampling_criterion='maximin',
    results_filename='training_data'
)
```

##### Parameters

- `output_dir` (str, optional): Directory for saving files. Default: 'DATA_PREPARATION'
- `show_progress` (bool, optional): Whether to show progress bars during execution. Default: True
- `show_result` (bool, optional): Whether to show result summary after execution. Default: True
- `sampling_method` (str, optional): Sampling method to use ('lhs', 'sobol', 'halton', or 'random'). Default: 'lhs'
- `sampling_criterion` (str, optional): Criterion for LHS (only used when `sampling_method` is 'lhs'). Default: 'maximin'
- `results_filename` (str, optional): Base filename for saving results CSV. Default: 'training_data'

#### Methods

##### `add_design_variable`

Add a design variable to the experiment.

```python
add_design_variable(name, range_bounds, cov=None, std=None, delta=None, distribution='normal', description='')
```

###### Parameters

- `name` (str): Name of the variable
- `range_bounds` (List[float]): List containing [min, max] bounds
- `cov` (float, optional): Coefficient of variation
- `std` (float, optional): Standard deviation
- `delta` (float, optional): -+ Uniform bounds
- `distribution` (str, optional): Distribution type ('normal' or 'lognormal'). Default: 'normal'
- `description` (str, optional): Optional description of the variable. Default: ''

###### Returns

- None

##### `add_env_variable`

Add an environmental variable to the experiment.

```python
add_env_variable(name, distribution, description='', mean=None, cov=None, std=None, low=None, high=None)
```

###### Parameters

- `name` (str): Name of the variable
- `distribution` (str): Distribution type (see Supported Distributions)
- `description` (str, optional): Optional description of the variable. Default: ''
- `mean` (float, optional): Mean value (for normal, lognormal, etc.). Default: None
- `cov` (float, optional): Coefficient of variation (alternative to std). Default: None
- `std` (float, optional): Standard deviation (alternative to cov). Default: None
- `low` (float, optional): Lower bound (for uniform, triangular, etc.). Default: None
- `high` (float, optional): Upper bound (for uniform, triangular, etc.). Default: None

###### Returns

- None

##### `save`

Save design configuration to a JSON file.

```python
save(filename='data_info')
```

###### Parameters

- `filename` (str, optional): Base name for the JSON file (without extension). Default: 'data_info'

###### Returns

- None

##### `load`

Load design configuration from a previously saved JSON file.

```python
load(filename)
```

###### Parameters

- `filename` (str): Base name of the JSON file to load (without extension)

###### Returns

- None

##### `run`

Run the sampling process and evaluate the objective function.

```python
run(objective_function, num_samples, save_results=True)
```

###### Parameters

- `objective_function` (Callable): Function to evaluate samples
- `num_samples` (int): Number of samples to generate
- `save_results` (bool, optional): Whether to save results to file. Default: True

###### Returns

- pandas.DataFrame containing samples and objective values

##### `print_summary`

Print summary statistics of the sampling results.

```python
print_summary(df)
```

###### Parameters

- `df` (pandas.DataFrame): DataFrame containing the sampling results

###### Returns

- None

### `AdaptiveDistributionSampler`

A class for generating samples from different distributions.

#### Class Attributes

- `SAMPLING_METHODS` (list): List of supported sampling methods: ['random', 'lhs', 'sobol', 'halton']

#### Methods

##### `generate_samples`

Generate samples for a given distribution type.

```python
generate_samples(distribution, mean=None, cov=None, std=None, lower=None, upper=None, size=1000, **kwargs)
```

###### Parameters

- `distribution` (str): Distribution type (see Supported Distributions)
- `mean` (float, optional): Mean value. Default: None
- `cov` (float, optional): Coefficient of variation. Default: None
- `std` (float, optional): Standard deviation (alternative to cov). Default: None
- `lower` (float, optional): Lower bound. Default: None
- `upper` (float, optional): Upper bound. Default: None
- `size` (int, optional): Number of samples to generate. Default: 1000
- `**kwargs` (dict): Additional parameters for specific distributions

###### Returns

- numpy.ndarray of samples

##### `generate_design_samples`

Generate samples for design variables using specified method.

```python
generate_design_samples(design_vars, num_samples, method='lhs', criterion='maximin')
```

###### Parameters

- `design_vars` (List[Variable]): List of design variables
- `num_samples` (int): Number of samples to generate
- `method` (str, optional): Sampling method. Default: 'lhs'
- `criterion` (str, optional): Criterion for LHS. Default: 'maximin'

###### Returns

- numpy.ndarray of samples scaled to variable bounds

##### `generate_env_samples`

Generate samples for an environmental variable.

```python
generate_env_samples(var, num_samples)
```

###### Parameters

- `var` (Variable or dict): Environmental variable to sample
- `num_samples` (int): Number of samples to generate

###### Returns

- numpy.ndarray of samples

##### `generate_all_samples`

Generate samples for all variables.

```python
generate_all_samples(variables, num_samples, method='lhs', criterion=None)
```

###### Parameters

- `variables` (List[Variable]): List of Variable objects
- `num_samples` (int): Number of samples to generate
- `method` (str, optional): Sampling method. Default: 'lhs'
- `criterion` (str, optional): Criterion for LHS. Default: None

###### Returns

- numpy.ndarray of samples

##### `calculate_input_bounds`

Calculate input bounds for all variables.

```python
calculate_input_bounds(variables)
```

###### Parameters

- `variables` (List[Variable]): List of variables

###### Returns

- List[List[float]] containing [min, max] bounds for each variable

### `Variable`

A dataclass representing a variable in the design.

#### Attributes

- `name` (str): Name of the variable
- `vars_type` (str): Variable type ('design_vars' or 'env_vars')
- `distribution` (str): Distribution type (see Supported Distributions)
- `description` (str): Optional description
- `range_bounds` (List[float], optional): List containing [min, max] bounds (for design variables)
- `low` (float, optional): Lower bound (for various distributions)
- `high` (float, optional): Upper bound (for various distributions)
- `mean` (float, optional): Mean value (for various distributions)
- `cov` (float, optional): Coefficient of variation
- `std` (float, optional): Standard deviation (alternative to cov)

## Supported Distributions

The following distributions are supported for environment variables:

1. **Uniform**: Constant probability within range
   - Required parameters: `low`, `high`

2. **Normal**: Gaussian distribution
   - Required parameters: `mean` plus either `cov` or `std` 
   - Optional parameters: `low`, `high` (for truncation)

3. **Lognormal**: Natural logarithm follows normal distribution
   - Required parameters: `mean` plus either `cov` or `std`
   - Optional parameters: `low`, `high` (for truncation)

## Module Functions

### `get_version`

Return the current version of the module.

```python
get_version()
```

#### Returns

- str - Version string

### `get_citation`

Return citation information for the module.

```python
get_citation()
```

#### Returns

- str - Citation text

### `print_usage`

Print detailed usage instructions for the module.

```python
print_usage()
```

#### Returns

- None

## Examples

### Basic Example

```python
import numpy as np
from PyEGRO.doe import InitialDesign

# Define objective function
def objective_func_2D(x):
    X1, X2 = x[:, 0], x[:, 1]
    return -((X1 - 2)**2 + (X2 - 3)**2)

# Create initial design
design = InitialDesign(
    output_dir='my_experiment',
    sampling_method='lhs',
    sampling_criterion='maximin'
)

# Add variables
design.add_design_variable(name='x1', range_bounds=[-5, 5], cov=0.1)
design.add_design_variable(name='x2', range_bounds=[-5, 5], cov=0.1)

# Run sampling
results = design.run(objective_function=objective_func_2D, num_samples=50)
```

### With Multiple Distribution Types

```python
# Create design
design = InitialDesign(sampling_method='sobol')

# Add design variables
design.add_design_variable(name='x1', range_bounds=[-5, 5], cov=0.1)

# Add environmental variables with different distributions
design.add_env_variable(
    name='normal_var',
    distribution='normal',
    mean=5,
    std=2.0  # Using std instead of cov
)

design.add_env_variable(
    name='uniform_var',
    distribution='uniform',
    low=0.0,
    high=1.0
)

design.add_env_variable(
    name='lognormal_var',
    distribution='lognormal',
    mean=10,
    std=2.0  
)

# Define objective function that uses all variables
def objective_func(x):
    design_var = x[:, 0]
    normal_var = x[:, 1]
    uniform_var = x[:, 2]
    lognormal_var = x[:, 3]
    
    return design_var**2 + 10*normal_var + 5*uniform_var + lognormal_var

# Run sampling
results = design.run(objective_function=objective_func, num_samples=100)
```

## Output File Formats

### data_info.json

Contains configuration information including:
- Variable definitions (name, type, distribution, bounds, etc.)
- Sampling configuration (method, criterion)
- Metadata (creation time, variable counts)

### training_data.csv

Contains the sample data:
- Columns for each variable (design and environmental)
- Final column 'y' with objective function values