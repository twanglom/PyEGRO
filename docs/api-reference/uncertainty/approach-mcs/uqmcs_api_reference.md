# PyEGRO UQmcs API Reference

This document provides detailed information about the API for PyEGRO's Uncertainty Quantification Monte Carlo Simulation (UQmcs) module.

## Table of Contents
1. [UncertaintyPropagation Class](#uncertaintypropagation-class)
   * [Constructor Parameters](#constructor-parameters)
   * [Methods](#methods)
2. [Convenience Functions](#convenience-functions)
   * [run_uncertainty_analysis](#run_uncertainty_analysis)
   * [analyze_specific_point](#analyze_specific_point)
3. [InformationVisualization Class](#informationvisualization-class)
   * [Constructor Parameters](#constructor-parameters-1)
   * [Methods](#methods-1)
4. [Data Configuration Format](#data-configuration-format)
   * [Design Variables](#design-variables)
   * [Environmental Variables](#environmental-variables)
   * [Example Configuration](#example-configuration)

## UncertaintyPropagation Class

The main class for uncertainty propagation analysis in PyEGRO.

### Constructor Parameters

```python
UncertaintyPropagation(
    data_info_path: str = "DATA_PREPARATION/data_info.json",
    model_handler: Optional[Any] = None,
    true_func: Optional[callable] = None,
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    output_dir: str = "RESULT_QOI",
    show_variables_info: bool = True,
    random_seed: int = 42
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| data_info_path | str | Path to the data configuration JSON file | "DATA_PREPARATION/data_info.json" |
| model_handler | Optional[Any] | Model handler for surrogate model | None |
| true_func | Optional[callable] | True objective function for direct evaluation | None |
| model_path | Optional[str] | Path to trained GPR model (if using surrogate) | None |
| use_gpu | bool | Whether to use GPU if available for surrogate model | True |
| output_dir | str | Directory for saving results | "RESULT_QOI" |
| show_variables_info | bool | Whether to display variable information | True |
| random_seed | int | Random seed for reproducibility | 42 |

**Note**: At least one of `true_func`, `model_handler`, or `model_path` must be provided.

### Methods

#### from_initial_design (class method)

Creates a UncertaintyPropagation instance directly from an InitialDesign object.

```python
@classmethod
def from_initial_design(cls, 
                        design=None,
                        design_infos=None,
                        true_func=None, 
                        output_dir="RESULT_QOI",
                        use_gpu=True,
                        show_variables_info=True,
                        random_seed=42)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| design | InitialDesign | InitialDesign instance | None |
| design_infos | InitialDesign | Alternative name for design (for backward compatibility) | None |
| true_func | Optional[callable] | Optional override for objective function | None |
| output_dir | str | Directory for saving results | "RESULT_QOI" |
| use_gpu | bool | Whether to use GPU if available | True |
| show_variables_info | bool | Whether to display variable information | True |
| random_seed | int | Random seed for reproducibility | 42 |

#### display_variable_info

Displays detailed information about all variables.

```python
def display_variable_info(self)
```

#### run_analysis

Runs the uncertainty propagation analysis.

```python
def run_analysis(self,
                true_func: Optional[callable] = None,
                num_design_samples: int = 1000,
                num_mcs_samples: int = 100000,
                show_progress: bool = True) -> pd.DataFrame
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| true_func | Optional[callable] | Optional override for the true function | None |
| num_design_samples | int | Number of design points to evaluate | 1000 |
| num_mcs_samples | int | Number of Monte Carlo samples per design point | 100000 |
| show_progress | bool | Whether to show progress bar | True |

**Returns**: pandas.DataFrame containing the analysis results.

#### analyze_specific_point

Analyzes uncertainty at a specific design point with customized environmental variables.

```python
def analyze_specific_point(self,
                         design_point: Dict[str, float],
                         env_vars_override: Optional[Dict[str, Dict[str, float]]] = None,
                         num_mcs_samples: int = 100000,
                         create_pdf: bool = True,
                         create_reliability: bool = True) -> Dict[str, Any]
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| design_point | Dict[str, float] | Dictionary of design variable values {var_name: value} | Required |
| env_vars_override | Optional[Dict[str, Dict[str, float]]] | Dictionary to override env variable distributions | None |
| num_mcs_samples | int | Number of Monte Carlo samples | 100000 |
| create_pdf | bool | Whether to create PDF visualization | True |
| create_reliability | bool | Whether to create reliability plot | True |

**Returns**: Dictionary containing analysis results.

## Convenience Functions

### run_uncertainty_analysis

Convenience function to run uncertainty propagation analysis.

```python
def run_uncertainty_analysis(
    data_info_path: str = "DATA_PREPARATION/data_info.json",
    true_func: Optional[callable] = None,
    model_handler: Optional[Any] = None,
    model_path: Optional[str] = None,
    num_design_samples: int = 1000,
    num_mcs_samples: int = 100000,
    use_gpu: bool = True,
    output_dir: str = "RESULT_QOI",
    random_seed: int = 42,
    show_progress: bool = True
) -> pd.DataFrame
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| data_info_path | str | Path to the data configuration file | "DATA_PREPARATION/data_info.json" |
| true_func | Optional[callable] | True objective function (if using direct evaluation) | None |
| model_handler | Optional[Any] | Model handler for surrogate model | None |
| model_path | Optional[str] | Path to trained GPR model (if using surrogate) | None |
| num_design_samples | int | Number of design points to evaluate | 1000 |
| num_mcs_samples | int | Number of Monte Carlo samples per design point | 100000 |
| use_gpu | bool | Whether to use GPU if available for surrogate model | True |
| output_dir | str | Directory for saving results | "RESULT_QOI" |
| random_seed | int | Random seed for reproducibility | 42 |
| show_progress | bool | Whether to show progress bar | True |

**Returns**: pandas.DataFrame containing the analysis results.

### analyze_specific_point

Convenience function to analyze uncertainty at a specific design point.

```python
def analyze_specific_point(
    data_info_path: str,
    design_point: Dict[str, float],
    true_func: Optional[callable] = None,
    model_handler: Optional[Any] = None,
    env_vars_override: Optional[Dict[str, Dict[str, float]]] = None,
    num_mcs_samples: int = 100000,
    output_dir: str = "RESULT_QOI_POINT",
    random_seed: int = 42,
    create_pdf: bool = True,
    create_reliability: bool = True
) -> Dict[str, Any]
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| data_info_path | str | Path to the data configuration file | Required |
| design_point | Dict[str, float] | Dictionary of design variable values {var_name: value} | Required |
| true_func | Optional[callable] | True objective function (if using direct evaluation) | None |
| model_handler | Optional[Any] | Model handler for surrogate model | None |
| env_vars_override | Optional[Dict[str, Dict[str, float]]] | Optional dictionary to override env variable distributions | None |
| num_mcs_samples | int | Number of Monte Carlo samples | 100000 |
| output_dir | str | Directory for saving results | "RESULT_QOI_POINT" |
| random_seed | int | Random seed for reproducibility | 42 |
| create_pdf | bool | Whether to create PDF visualization | True |
| create_reliability | bool | Whether to create reliability plot | True |

**Returns**: Dictionary containing analysis results.

## InformationVisualization Class

Handles visualization of uncertainty analysis results.

### Constructor Parameters

```python
InformationVisualization(output_dir=None)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| output_dir | str or Path | Directory for saving visualizations | None (defaults to 'RESULT_QOI') |

### Methods

#### create_visualization

Creates appropriate visualization based on number of design variables.

```python
def create_visualization(self, results_df: pd.DataFrame, design_vars: List[Dict[str, Any]])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| results_df | pd.DataFrame | DataFrame containing the analysis results |
| design_vars | List[Dict[str, Any]] | List of design variable dictionaries |

#### create_pdf_visualization

Creates PDF visualization for a specific design point.

```python
def create_pdf_visualization(self, 
                            design_point: Dict[str, Any], 
                            samples: np.ndarray, 
                            title: str = "Probability Density Function at Design Point",
                            filename: str = "pdf_visualization.png") -> plt.Figure
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| design_point | Dict[str, Any] | Dictionary of design variable values | Required |
| samples | np.ndarray | Array of samples from Monte Carlo simulation | Required |
| title | str | Title for the plot | "Probability Density Function at Design Point" |
| filename | str | Filename for saving the plot | "pdf_visualization.png" |

**Returns**: Matplotlib figure object.

#### create_reliability_plot

Creates reliability plot showing probability of exceedance vs threshold value.

```python
def create_reliability_plot(self, 
                          samples: np.ndarray, 
                          threshold_values: np.ndarray, 
                          threshold_type: str = 'upper',
                          title: str = "Reliability Analysis",
                          filename: str = "reliability_analysis.png") -> plt.Figure
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| samples | np.ndarray | Array of samples from Monte Carlo simulation | Required |
| threshold_values | np.ndarray | Array of threshold values to evaluate | Required |
| threshold_type | str | 'upper' for P(X > threshold) or 'lower' for P(X < threshold) | 'upper' |
| title | str | Title for the plot | "Reliability Analysis" |
| filename | str | Filename for saving the plot | "reliability_analysis.png" |

**Returns**: Matplotlib figure object.

#### create_joint_pdf_visualization

Creates a joint PDF visualization showing the relationship between an input variable and the output response.

```python
def create_joint_pdf_visualization(self,
                                 design_point: Dict[str, Any],
                                 samples: np.ndarray,
                                 target_variable: str,
                                 input_variable: str,
                                 title: str = "Joint Probability Distribution",
                                 filename: str = "joint_pdf_visualization.png") -> plt.Figure
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| design_point | Dict[str, Any] | Dictionary of design variable values | Required |
| samples | np.ndarray | 2D array of samples - first column is input variable, second is output | Required |
| target_variable | str | Name of the target/output variable | Required |
| input_variable | str | Name of the input variable to analyze | Required |
| title | str | Title for the plot | "Joint Probability Distribution" |
| filename | str | Filename for saving the plot | "joint_pdf_visualization.png" |

**Returns**: Matplotlib figure object.

## Data Configuration Format

The data configuration is stored in a JSON file with specific structure for design and environmental variables.

### Design Variables

Design variables are controllable inputs with the following attributes:

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| name | str | Variable name | Yes |
| vars_type | str | Must be 'design_vars' | Yes |
| range_bounds | [float, float] | Lower and upper bounds for the variable | Yes |
| cov | float | Coefficient of Variation (uncertainty) | No (default = 0, deterministic) |
| distribution | str | Probability distribution (e.g., 'normal') | No (required if cov > 0) |
| description | str | Description of the variable | No |

### Environmental Variables

Environmental variables are uncontrollable inputs or noise factors:

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| name | str | Variable name | Yes |
| vars_type | str | Must be 'env_vars' | Yes |
| distribution | str | Probability distribution (e.g., 'normal', 'uniform') | Yes |
| mean | float | Mean value (for 'normal' distribution) | Yes (for 'normal') |
| std | float | Standard deviation (for 'normal' distribution) | Yes (if 'cov' not specified) |
| cov | float | Coefficient of variation (for 'normal' distribution) | Yes (if 'std' not specified) |
| low | float | Lower bound (for 'uniform' distribution) | Yes (for 'uniform') |
| high | float | Upper bound (for 'uniform' distribution) | Yes (for 'uniform') |
| description | str | Description of the variable | No |

### Example Configuration

```json
{
  "variables": [
    {
      "name": "x1",
      "vars_type": "design_vars",
      "range_bounds": [0, 10],
      "cov": 0.05,
      "distribution": "normal",
      "description": "First design variable"
    },
    {
      "name": "x2",
      "vars_type": "design_vars",
      "range_bounds": [-5, 5],
      "cov": 0,
      "description": "Deterministic design variable"
    },
    {
      "name": "noise1",
      "vars_type": "env_vars",
      "distribution": "normal",
      "mean": 0,
      "std": 0.1,
      "description": "Measurement noise"
    },
    {
      "name": "temp",
      "vars_type": "env_vars",
      "distribution": "uniform",
      "low": 20,
      "high": 30,
      "description": "Operating temperature"
    }
  ]
}
```