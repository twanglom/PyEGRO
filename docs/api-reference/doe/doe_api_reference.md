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
add_design_variable(
    name,
    range_bounds,
    cov=None,
    std=None,
    delta=None,
    distribution='normal',
    description=''
)
```

###### Parameters

- `name` (str): Name of the variable
- `range_bounds` (List[float]): List containing [min, max] bounds
- `cov` (float, optional): Coefficient of variation (uncertainty as a proportion of the mean)
- `std` (float, optional): Standard deviation (fixed uncertainty value)
- `delta` (float, optional): Half-width for uniform uncertainty around design points
- `distribution` (str, optional): Distribution type ('normal', 'uniform', 'lognormal'). Default: 'normal'
- `description` (str, optional): Optional description of the variable. Default: ''

###### Notes

- Specify exactly one of `cov`, `std`, or `delta` for uncertain variables.
- For deterministic variables, don't specify any uncertainty parameter.
- If `delta` is specified, using 'uniform' distribution is recommended.

###### Returns

- None

##### `add_env_variable`

Add an environmental variable to the experiment.

```python
add_env_variable(
    name,
    distribution,
    description='',
    mean=None,
    cov=None,
    std=None,
    low=None,
    high=None,
    min=None,
    max=None
)
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
- `min` (float, optional): Lower bound (alternative to low). Default: None
- `max` (float, optional): Upper bound (alternative to high). Default: None

###### Notes

- For normal/lognormal distributions, specify `mean` and exactly one of `cov` or `std`.
- For uniform distributions, specify either `low`/`high` or `min`/`max`.
- The parameters `min`/`max` are provided for backward compatibility with `low`/`high`.

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
generate_samples(
    distribution,
    mean=None,
    cov=None,
    std=None,
    delta=None,
    lower=None,
    upper=None,
    size=1000,
    **kwargs
)
```

###### Parameters

- `distribution` (str): Distribution type (see Supported Distributions)
- `mean` (float, optional): Mean value. Default: None
- `cov` (float, optional): Coefficient of variation. Default: None
- `std` (float, optional): Standard deviation (alternative to cov). Default: None
- `delta` (float, optional): Half-width for uniform distribution. Default: None
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

#### Attributes 

- `cov` (float, optional): Coefficient of variation
- `std` (float, optional): Standard deviation (alternative to cov)
- `delta` (float, optional): Half-width for uniform uncertainty
- `low` (float, optional): Lower bound (for various distributions)
- `high` (float, optional): Upper bound (for various distributions)
- `min` (float, optional): Lower bound (alternative to low)
- `max` (float, optional): Upper bound (alternative to high)

## Supported Distributions

The following distributions are supported for design and environment variables:

**Uniform**: Constant probability within interval

- Required parameters  

    env variables: `low`/`high` or `min`/`max`

    design variables: `delta` 

**Normal**: Gaussian distribution

- Required parameters 
   
    env variables: `mean` plus either `cov` or `std` 

    design variables: `cov` or `std` 

    Optional parameters: `lower`, `upper` (for truncation)

**Lognormal**: Natural logarithm follows normal distribution

- Required parameters 

    env variables: `mean` plus either `cov` or `std` 

    design variables: `cov` or `std` 

    Optional parameters: `lower`, `upper` (for truncation)


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

### Using Standard Deviation Instead of CoV

```python
# Create design
design = InitialDesign(sampling_method='lhs')

# Add design variables with standard deviation
design.add_design_variable(
    name='x1', 
    range_bounds=[-5, 5], 
    std=0.2  # Fixed standard deviation
)

design.add_design_variable(
    name='x2', 
    range_bounds=[-5, 5], 
    std=0.3
)

# Run sampling
results = design.run(objective_function=objective_func_2D, num_samples=50)
```

### Using Delta for Uniform Uncertainty

```python
# Create design
design = InitialDesign(sampling_method='sobol')

# Add design variables with uniform uncertainty
design.add_design_variable(
    name='x1', 
    range_bounds=[-5, 5], 
    delta=0.1,  # Half-width of uniform distribution
    distribution='uniform'
)

design.add_design_variable(
    name='x2', 
    range_bounds=[-5, 5], 
    delta=0.1,
    distribution='uniform'
)

# Run sampling
results = design.run(objective_function=objective_func_2D, num_samples=50)
```

### With Multiple Distribution Types

```python
# Create design
design = InitialDesign(sampling_method='sobol')

# Add design variables
design.add_design_variable(name='x1', range_bounds=[-5, 5], std=0.2)

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

Example structure:
```json
{
  "variables": [
    {
      "name": "x1",
      "vars_type": "design_vars",
      "range_bounds": [-5, 5],
      "std": 0.2,
      "distribution": "normal",
      "description": "Design variable with std uncertainty"
    },
    {
      "name": "noise",
      "vars_type": "env_vars",
      "distribution": "normal",
      "mean": 0,
      "std": 0.5,
      "description": "Environmental noise"
    }
  ],
  "variable_names": ["x1", "noise"],
  "input_bound": [[-5, 5], [-1.5, 1.5]],
  "sampling_config": {
    "method": "lhs",
    "criterion": "maximin"
  },
  "metadata": {
    "created_at": "2025-03-31T12:34:56.789Z",
    "total_variables": 2
  }
}
```

### training_data.csv

Contains the sample data:
- Columns for each variable (design and environmental)
- Final column 'y' with objective function values

Example:
```
x1,noise,y
-3.2,0.12,-10.24
2.1,-0.42,-4.41
4.5,0.31,-20.25
-1.8,0.05,-3.24
...
```

## Best Practices

### When to use each uncertainty specification method

- **Use `cov`**: When uncertainty is proportional to the design value (e.g., manufacturing tolerance as a percentage)
- **Use `std`**: When uncertainty is fixed regardless of design value or near zero regions exist
- **Use `delta`**: When uncertainty follows a uniform distribution with fixed width

### Choosing a sampling method

- **LHS**: Good for general-purpose sampling with even coverage
- **Sobol/Halton**: Good for progressive sampling where you might need more points later
- **Random**: Only use when randomness is specifically required

### Handling environmental variables

- Use `std` instead of `cov` for normal environmental variables when the mean is near zero
- For uniform distributions, both `low`/`high` and `min`/`max` are supported for backward compatibility
