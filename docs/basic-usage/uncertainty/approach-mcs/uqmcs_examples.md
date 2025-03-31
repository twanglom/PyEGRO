# PyEGRO Uncertainty Quantification Module Usage Guide
(PyEGRO.uncertainty.UQmcs)

This guide provides examples and usage patterns for PyEGRO's Uncertainty Quantification module using Monte Carlo Simulation (UQmcs). The module helps you analyze how uncertainty in input variables propagates to output responses, making it useful for robust design and risk assessment.

## Table of Contents

#### [Quick Start](#quick-start-section)
   * [Basic Setup](#basic-setup)
   * [Running the Analysis](#running-the-analysis)
#### [Uncertainty Specification Methods](#uncertainty-specification-methods)
   * [Using Coefficient of Variation (CoV)](#using-coefficient-of-variation-cov)
   * [Using Standard Deviation (Std)](#using-standard-deviation-std)
   * [Using Delta for Uniform Uncertainty](#using-delta-for-uniform-uncertainty)
   * [Handling Near-Zero Regions](#handling-near-zero-regions)
#### [Working with Different Models](#working-with-different-models)
   * [Using a Direct Function](#using-a-direct-function)
   * [Using a Metamodel](#using-a-metamodel)
#### [Example Applications](#example-applications)
   * [1D Multi-modal Function](#1d-multi-modal-function)
   * [2D Multi-modal Function](#2d-multi-modal-function)
   * [Ishigami Function](#ishigami-function)
   * [Beam Analysis with Material Density Uncertainty](#beam-analysis-with-material-density-uncertainty)
   * [Borehole Function](#borehole-function)
   * [Chemical Process Model](#chemical-process-model)
   * [Circuit Design Problem](#circuit-design-problem)
#### [Advance Usage](#advance-usage)


## 1. Quick Start {#quick-start-section}

* Basic Setup

Before running any analysis, you need to define your variables (design variables and environmental variables) in a configuration file. Here's how to set up a simple configuration:

```python
import numpy as np
import json
import os

# Create a dictionary defining your variables
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'range_bounds': [0, 10],
            'cov': 0.05,  # 5% coefficient of variation
            'distribution': 'normal',
            'description': 'First design variable'
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'range_bounds': [-5, 5],
            'std': 0.2,  # Using standard deviation instead of CoV
            'distribution': 'normal',
            'description': 'Second design variable'
        },
        {
            'name': 'noise',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 0,
            'std': 0.1,
            'description': 'Measurement noise'
        }
    ]
}

# Save the configuration to a JSON file
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/data_info.json", 'w') as f:
    json.dump(data_info, f)
```

### Running the Analysis

Once your configuration is ready, you can run the uncertainty propagation analysis:

```python
from PyEGRO.uncertainty.UQmcs import run_uncertainty_analysis

# Define your objective function
def simple_function(X):
    return X[:, 0]**2 + X[:, 1] + 0.5*X[:, 2]

# Run the analysis
results = run_uncertainty_analysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    true_func=simple_function,
    num_design_samples=500,    # Number of design points to evaluate
    num_mcs_samples=10000,     # Number of Monte Carlo samples per design point
    output_dir="RESULT_QOI",   # Directory to save results
    show_progress=True
)

# The results are also available as a pandas DataFrame
print(results.head())
```

## 2. Uncertainty Specification Methods {#uncertainty-specification-methods}

The UQmcs module supports multiple ways to specify uncertainty for design and environmental variables.

### Using Coefficient of Variation (CoV)

Coefficient of Variation (CoV) is the ratio of the standard deviation to the mean, expressed as a decimal or percentage. It's useful when the uncertainty scales with the magnitude of the variable.

```python
import numpy as np
import json
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Define a simple quadratic function
def quadratic_func(X):
    return X[:, 0]**2

# Setup with CoV
data_info = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [1, 10],  # Positive range (good for CoV)
            'cov': 0.1,  # 10% coefficient of variation
            'distribution': 'normal',
            'description': 'Design variable with CoV uncertainty'
        }
    ]
}

# Save and run analysis
with open("DATA_PREPARATION/cov_example.json", 'w') as f:
    json.dump(data_info, f)

uq_cov = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/cov_example.json",
    true_func=quadratic_func,
    output_dir="RESULT_COV"
)

results_cov = uq_cov.run_analysis(num_design_samples=100, num_mcs_samples=10000)
```

### Using Standard Deviation (Std)

Standard deviation specifies the absolute width of the uncertainty distribution. This is preferable when the variable range crosses zero or when you want a constant uncertainty regardless of the mean value.

```python
# Setup with standard deviation
data_info = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [-5, 5],  # Range crosses zero (better to use std)
            'std': 0.3,  # Fixed standard deviation of 0.3
            'distribution': 'normal',
            'description': 'Design variable with std uncertainty'
        }
    ]
}

# Save and run analysis
with open("DATA_PREPARATION/std_example.json", 'w') as f:
    json.dump(data_info, f)

uq_std = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/std_example.json",
    true_func=quadratic_func,
    output_dir="RESULT_STD"
)

results_std = uq_std.run_analysis(num_design_samples=100, num_mcs_samples=10000)
```

### Using Delta for Uniform Uncertainty

The `delta` parameter allows you to specify uniform uncertainty around design points, which is useful when you prefer a uniform distribution rather than normal.

```python
# Setup with delta for uniform uncertainty
data_info = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [0, 10],
            'delta': 0.5,  # Half-width of 0.5 around design points
            'distribution': 'uniform',
            'description': 'Design variable with uniform uncertainty'
        }
    ]
}

# Save and run analysis
with open("DATA_PREPARATION/delta_example.json", 'w') as f:
    json.dump(data_info, f)

uq_delta = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/delta_example.json",
    true_func=quadratic_func,
    output_dir="RESULT_DELTA"
)

results_delta = uq_delta.run_analysis(num_design_samples=100, num_mcs_samples=10000)
```

### Handling Near-Zero Regions

When variables cross near zero and you use CoV, issues can arise because CoV is undefined at zero. The updated code handles this automatically, but it's best to use `std` for such cases.

```python
import numpy as np
import json
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation
import matplotlib.pyplot as plt

# Define a function that's sensitive to the sign of x
def sine_func(X):
    return np.sin(X[:, 0])

# Compare CoV and Std for a variable that crosses zero
data_info_cov = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [-3.14, 3.14],  # Crosses zero
            'cov': 0.1,  # CoV will be handled specially near zero
            'distribution': 'normal',
            'description': 'Design variable with CoV (crossing zero)'
        }
    ]
}

data_info_std = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [-3.14, 3.14],  # Crosses zero
            'std': 0.2,  # Better approach for variables crossing zero
            'distribution': 'normal',
            'description': 'Design variable with std (crossing zero)'
        }
    ]
}

# Save configurations
with open("DATA_PREPARATION/near_zero_cov.json", 'w') as f:
    json.dump(data_info_cov, f)
    
with open("DATA_PREPARATION/near_zero_std.json", 'w') as f:
    json.dump(data_info_std, f)

# Run analyses
uq_cov = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/near_zero_cov.json",
    true_func=sine_func,
    output_dir="RESULT_NEAR_ZERO_COV"
)

uq_std = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/near_zero_std.json",
    true_func=sine_func,
    output_dir="RESULT_NEAR_ZERO_STD"
)

results_cov = uq_cov.run_analysis(num_design_samples=200, num_mcs_samples=5000)
results_std = uq_std.run_analysis(num_design_samples=200, num_mcs_samples=5000)

# Plot to compare
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(results_cov['x'], results_cov['Mean'], c=results_cov['StdDev'], cmap='viridis')
plt.colorbar(label='Standard Deviation')
plt.title('Using CoV for Variable Crossing Zero')
plt.xlabel('x')
plt.ylabel('Mean Response')

plt.subplot(1, 2, 2)
plt.scatter(results_std['x'], results_std['Mean'], c=results_std['StdDev'], cmap='viridis')
plt.colorbar(label='Standard Deviation')
plt.title('Using Std for Variable Crossing Zero')
plt.xlabel('x')
plt.ylabel('Mean Response')

plt.tight_layout()
plt.savefig("near_zero_comparison.png")
plt.show()
```

## 3. Working with Different Models {#working-with-different-models}

### Using a Direct Function

For analytical functions or computationally inexpensive simulations, you can use the objective function directly:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import run_uncertainty_analysis

# Define a non-linear function with multiple inputs
def complex_function(X):
    """
    Non-linear function with multiple inputs and complex behavior.
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    
    return x1**2 * np.sin(x2) + x3 * np.exp(-0.5 * (x1 - x2)**2)

# Define data_info with various uncertainty specifications
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'range_bounds': [0, 5],
            'cov': 0.1,  # Using CoV for positive range
            'distribution': 'normal',
            'description': 'First design variable with CoV'
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'range_bounds': [0, 2*np.pi],
            'std': 0.2,  # Using std for more direct control
            'distribution': 'normal',
            'description': 'Second design variable with std'
        },
        {
            'name': 'x3',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'min': 0,  # Using min/max notation
            'max': 10,
            'description': 'Environmental variable with uniform distribution'
        }
    ]
}

# Save data_info
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/complex_data_info.json", 'w') as f:
    json.dump(data_info, f)

# Run analysis using the convenience function
results_df = run_uncertainty_analysis(
    data_info_path="DATA_PREPARATION/complex_data_info.json",
    true_func=complex_function,
    num_design_samples=500,
    num_mcs_samples=10000,
    output_dir="RESULT_QOI_COMPLEX",
    show_progress=True
)
```

### Using a Metamodel

For computationally expensive models, you can use a surrogate model:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import run_uncertainty_analysis
from PyEGRO.meta.gpr.gpr_utils import DeviceAgnosticGPR

# Load a pre-trained surrogate model
model_handler = DeviceAgnosticGPR(prefer_gpu=True)
model_handler.load_model('RESULT_MODEL_GPR')

# Run analysis with the surrogate model
results_df = run_uncertainty_analysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    model_handler=model_handler,
    num_design_samples=2000,  # Can use more samples since evaluations are cheap
    num_mcs_samples=100000,  # Higher accuracy with more MC samples
    output_dir="RESULT_QOI_SURROGATE",
    show_progress=True
)
```


## 4. Example Applications {#example-applications}

### 1D Multi-modal Function with Standard Deviation

Analyzing a function with multiple modes to study how uncertainty affects global behavior, using standard deviation for better handling of values crossing zero:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Define a multi-modal function
def multimodal_1d(X):
    x = X[:, 0]
    return np.sin(5*x) * np.exp(-0.5*x**2) + 0.2*np.sin(15*x)

# Setup data configuration using std instead of cov since range crosses zero
data_info = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [-3, 3],
            'std': 0.1,  # Using standard deviation instead of cov for zero-crossing range
            'distribution': 'normal',
            'description': 'Input variable with std-based uncertainty'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/multimodal_1d.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/multimodal_1d.json",
    true_func=multimodal_1d,
    output_dir="RESULT_MULTIMODAL_1D"
)

# Run general analysis
results = uq.run_analysis(num_design_samples=100, num_mcs_samples=10000)

# Analyze specific points of interest
interest_points = [
    {'x': -2.0},  # Local minimum
    {'x': 0.0},   # Global maximum
    {'x': 2.0}    # Local minimum
]

for i, point in enumerate(interest_points):
    print(f"Analyzing point {i+1}: {point}")
    result = uq.analyze_specific_point(
        design_point=point,
        num_mcs_samples=50000,
        create_pdf=True,
        create_reliability=True
    )
    print(f"Mean response: {result['statistics']['mean']}")
    print(f"95% CI: [{result['statistics']['percentiles']['2.5']}, {result['statistics']['percentiles']['97.5']}]")
    print()
```

### 1D Multi-modal Function (case 2)

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation
import matplotlib.pyplot as plt

# Define a multi-modal function with multiple local minima
def multimodal_peaks(X):
    """
    Multi-modal test function with multiple local minima.
    
    f(x) = -(100*exp(-(x-10)²/0.8) + 80*exp(-(x-20)²/50) + 20*exp(-(x-30)²/18) - 200)
    
    Has distinct peaks at x=10, x=20, and x=30 with different widths.
    """
    x = X[:, 0]
    term1 = 100 * np.exp(-((x - 10)**2) / 0.8)
    term2 = 80 * np.exp(-((x - 20)**2) / 50)
    term3 = 20 * np.exp(-((x - 30)**2) / 18)
    
    return -(term1 + term2 + term3 - 200)

# Setup data configuration using cov (appropriate here since values are all positive)
data_info = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [0, 40],
            'cov': 0.05,  # 5% coefficient of variation (works well for positive range)
            'distribution': 'normal',
            'description': 'Design variable with CoV-based uncertainty'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/peaks_function.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/peaks_function.json",
    true_func=multimodal_peaks,
    output_dir="RESULT_PEAKS"
)

# Run general analysis with more samples for better resolution
results = uq.run_analysis(num_design_samples=2000, num_mcs_samples=50000)

# Analyze only the second peak at x=20
point = {'x': 20.0}  # Second peak (wide, medium sensitivity)
print(f"Analyzing point: {point}")
result = uq.analyze_specific_point(
    design_point=point,
    num_mcs_samples=50000,
    create_pdf=True,
    create_reliability=True
)
print(f"Mean response: {result['statistics']['mean']:.4f}")
print(f"Standard deviation: {result['statistics']['std']:.4f}")
print(f"Coefficient of variation: {result['statistics']['cov']:.4f}")
print(f"95% CI: [{result['statistics']['percentiles']['2.5']:.4f}, {result['statistics']['percentiles']['97.5']:.4f}]")

# Add probability of failure analysis
# Define a performance requirement (e.g., response must be better than 115)
performance_threshold = 115.0
samples = result['statistics']['samples'] if 'samples' in result['statistics'] else None

if samples is None:
    # If samples aren't available in the results, generate them again
    x_samples = np.random.normal(20.0, 20.0 * 0.05, 50000)  # Using point value and CoV
    x_samples = x_samples.reshape(-1, 1)
    samples = multimodal_peaks(x_samples)

# Calculate probability of failure
pf = np.mean(samples < performance_threshold)
print(f"\nProbability of Failure Analysis:")
print(f"Performance threshold: {performance_threshold:.4f}")
print(f"Probability that response < threshold: {pf:.6f}")
print(f"Reliability: {1-pf:.6f}")
```

### 2D Multi-modal Function with Uniform Uncertainty

Analyzing a 2D function with uniform uncertainty around design points using the delta parameter:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Define a 2D multi-modal function (modified Himmelblau's function)
def multimodal_2d(X):
    x1, x2 = X[:, 0], X[:, 1]
    return -((x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2) + 200

# Setup data configuration with delta for uniform uncertainty
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'range_bounds': [-5, 5],
            'delta': 0.2,  # Half-width of 0.2 around design points
            'distribution': 'uniform',
            'description': 'First design variable with uniform uncertainty'
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'range_bounds': [-5, 5],
            'delta': 0.2,  # Half-width of 0.2 around design points
            'distribution': 'uniform',
            'description': 'Second design variable with uniform uncertainty'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/multimodal_2d.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance and run analysis
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/multimodal_2d.json",
    true_func=multimodal_2d,
    output_dir="RESULT_MULTIMODAL_2D"
)

# Run analysis
results = uq.run_analysis(
    num_design_samples=400,  # Higher resolution for 2D visualization
    num_mcs_samples=10000
)

# Analyze specific points (known optima of Himmelblau's function)
optima = [
    {'x1': 3.0, 'x2': 2.0},
    {'x1': -2.805118, 'x2': 3.131312},
    {'x1': -3.779310, 'x2': -3.283186},
    {'x1': 3.584428, 'x2': -1.848126}
]

for i, point in enumerate(optima):
    print(f"Analyzing optimum {i+1}: {point}")
    result = uq.analyze_specific_point(
        design_point=point,
        num_mcs_samples=25000,
        create_pdf=True,
        create_reliability=True
    )
    print(f"Mean response: {result['statistics']['mean']}")
    print(f"Standard deviation: {result['statistics']['std']}")
```




### Ishigami function

Sets up all three input variables (X1, X2, X3) as environmental variables with uniform distributions between -π and π

```python
import numpy as np
import json
import os
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Define the Ishigami function
def ishigami_function(X):
    """
    Ishigami function: f(X1, X2, X3) = sin(X1) + a*(sin(X2))^2 + b*X3^4*sin(X1)
    
    Parameters:
    X[:, 0] = X1
    X[:, 1] = X2
    X[:, 2] = X3
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    
    # Parameters for Ishigami function
    a = 7
    b = 0.1
    
    # Calculate Ishigami function
    return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

# Define data configuration
data_info = {
    'variables': [
        {
            'name': 'X1',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'min': -np.pi,
            'max': np.pi,
            'description': 'Random variable 1 (uniform in [-π, π])'
        },
        {
            'name': 'X2',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'min': -np.pi,
            'max': np.pi,
            'description': 'Random variable 2 (uniform in [-π, π])'
        },
        {
            'name': 'X3',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'min': -np.pi,
            'max': np.pi,
            'description': 'Random variable 3 (uniform in [-π, π])'
        }
    ]
}

# Create directory and save configuration
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/ishigami_function.json", 'w') as f:
    json.dump(data_info, f)


uq_cov = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/ishigami_function.json",
    true_func=ishigami_function,
    output_dir="RESULT_ISHIGAMI"
)

# Since we only have environmental variables, we need an empty dictionary
# for the design_point parameter (no design variables to specify)
empty_design = {}

# Analyze with default environmental conditions and create visualizations
result = uq_cov.analyze_specific_point(
    design_point=empty_design,
    num_mcs_samples=100000,
    create_pdf=True,
    create_reliability=True
)
```









### Beam Analysis with Material Density Uncertainty

This example analyzes a structural beam with uncertainty in the material properties:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Beam deflection model (simplified)
def beam_model(X):
    """
    Calculate maximum beam deflection with uncertain parameters
    X[:, 0] = Length (m)
    X[:, 1] = Width (m)  
    X[:, 2] = Height (m)
    X[:, 3] = Load (N)
    X[:, 4] = Elastic modulus (Pa)
    X[:, 5] = Density (kg/m^3)
    """
    L, b, h, P, E, rho = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    
    # Calculate moment of inertia
    I = b * h**3 / 12
    
    # Maximum deflection for a simply supported beam with point load at center
    deflection = P * L**3 / (48 * E * I)
    
    # Weight of the beam
    weight = L * b * h * rho
    
    # Return a weighted sum of deflection and weight (multi-objective)
    return deflection + 0.01 * weight

# Setup data configuration
data_info = {
    'variables': [
        {
            'name': 'Length',
            'vars_type': 'design_vars',
            'range_bounds': [1.0, 3.0],
            'std': 0.05,  # Fixed standard deviation of 5cm
            'distribution': 'normal',
            'description': 'Beam length (m) with std-based tolerance'
        },
        {
            'name': 'Width',
            'vars_type': 'design_vars',
            'range_bounds': [0.05, 0.2],
            'delta': 0.005,  # Uniform uncertainty of ±5mm
            'distribution': 'uniform',
            'description': 'Beam width (m) with uniform tolerance'
        },
        {
            'name': 'Height',
            'vars_type': 'design_vars',
            'range_bounds': [0.1, 0.5],
            'cov': 0.03,  # 3% manufacturing tolerance
            'distribution': 'normal',
            'description': 'Beam height (m)'
        },
        {
            'name': 'Load',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 1000,  # 1000 N
            'std': 200,    # 200 N standard deviation
            'description': 'Point load at center (N)'
        },
        {
            'name': 'Elastic_Modulus',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 70e9,   # Aluminum (Pa)
            'std': 3.5e9,   # Standard deviation of 3.5 GPa
            'description': 'Elastic modulus (Pa) with std-based variability'
        },
        {
            'name': 'Density',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 2700,   # Aluminum (kg/m^3)
            'cov': 0.03,    # 3% variability (using CoV for demonstration)
            'description': 'Material density (kg/m^3) with CoV-based variability'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/beam_model.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance and run analysis
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/beam_model.json",
    true_func=beam_model,
    output_dir="RESULT_BEAM"
)

# Run analysis
results = uq.run_analysis(
    num_design_samples=200,
    num_mcs_samples=5000
)

# Analyze a specific design point
nominal_design = {
    'Length': 2.0,
    'Width': 0.1,
    'Height': 0.3
}


# Analyze the nominal design with default environmental conditions
result = uq.analyze_specific_point(
    design_point=nominal_design,
    num_mcs_samples=100000,
    create_pdf=True,
    create_reliability=True
)

print("Nominal design analysis:")
print(f"Mean deflection: {result['statistics']['mean']:.6f} m")
print(f"Std deviation: {result['statistics']['std']:.6f} m")
print(f"95% CI: [{result['statistics']['percentiles']['2.5']:.6f}, {result['statistics']['percentiles']['97.5']:.6f}] m")

# Analyze with modified environmental conditions (higher load variability)
env_override = {
    'Load': {'mean': 1000, 'std': 400}  # Double the standard deviation
}

result_high_load = uq.analyze_specific_point(
    design_point=nominal_design,
    env_vars_override=env_override,
    num_mcs_samples=10000,
    create_pdf=True,
    create_reliability=True
)

print("\nDesign with higher load variability:")
print(f"Mean deflection: {result_high_load['statistics']['mean']:.6f} m")
print(f"Std deviation: {result_high_load['statistics']['std']:.6f} m")
print(f"95% CI: [{result_high_load['statistics']['percentiles']['2.5']:.6f}, {result_high_load['statistics']['percentiles']['97.5']:.6f}] m")
```

### Borehole Function

This example uses the borehole function, a common test function in the uncertainty quantification literature:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Borehole function
def borehole(X):
    """
    Borehole function with 8 inputs:
    X[:, 0] = rw (radius of borehole)
    X[:, 1] = r  (radius of influence)
    X[:, 2] = Tu (transmissivity of upper aquifer)
    X[:, 3] = Hu (potentiometric head of upper aquifer)
    X[:, 4] = Tl (transmissivity of lower aquifer)
    X[:, 5] = Hl (potentiometric head of lower aquifer)
    X[:, 6] = L  (length of borehole)
    X[:, 7] = Kw (hydraulic conductivity of borehole)
    """
    rw, r, Tu, Hu, Tl, Hl, L, Kw = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    
    frac1 = 2 * np.pi * Tu * (Hu - Hl)
    frac2a = 2*L*Tu / (np.log(r/rw) * rw**2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r/rw) * (1 + frac2a + frac2b)
    
    return frac1 / frac2

# Setup data configuration
data_info = {
    'variables': [
        {
            'name': 'rw',
            'vars_type': 'design_vars',
            'range_bounds': [0.05, 0.15],
            'std': 0.005,  # Standard deviation of 5mm
            'distribution': 'normal',
            'description': 'Radius of borehole (m) with std-based uncertainty'
        },
        {
            'name': 'r',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'min': 100,     # Using min instead of low
            'max': 50000,   # Using max instead of high
            'description': 'Radius of influence (m) using min/max notation'
        },
        {
            'name': 'Tu',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 63070,
            'high': 115600,
            'description': 'Transmissivity of upper aquifer (m^2/yr)'
        },
        {
            'name': 'Hu',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 990,
            'high': 1110,
            'description': 'Potentiometric head of upper aquifer (m)'
        },
        {
            'name': 'Tl',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 63.1,
            'high': 116,
            'description': 'Transmissivity of lower aquifer (m^2/yr)'
        },
        {
            'name': 'Hl',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 700,
            'high': 820,
            'description': 'Potentiometric head of lower aquifer (m)'
        },
        {
            'name': 'L',
            'vars_type': 'design_vars',
            'range_bounds': [1120, 1680],
            'cov': 0.05,
            'distribution': 'normal',
            'description': 'Length of borehole (m)'
        },
        {
            'name': 'Kw',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 9855,
            'high': 12045,
            'description': 'Hydraulic conductivity of borehole (m/yr)'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/borehole.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance and run analysis
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/borehole.json",
    true_func=borehole,
    output_dir="RESULT_BOREHOLE"
)

# Run analysis
results = uq.run_analysis(
    num_design_samples=100,
    num_mcs_samples=5000
)

# Analyze specific design points
designs_to_analyze = [
    {'rw': 0.10, 'L': 1400},  # Nominal design
    {'rw': 0.05, 'L': 1120},  # Minimum dimensions
    {'rw': 0.15, 'L': 1680}   # Maximum dimensions
]

for i, design in enumerate(designs_to_analyze):
    print(f"\nAnalyzing design {i+1}: {design}")
    result = uq.analyze_specific_point(
        design_point=design,
        num_mcs_samples=10000,
        create_pdf=True,
        create_reliability=True
    )
    print(f"Mean flow rate: {result['statistics']['mean']:.2f} m^3/yr")
    print(f"Coefficient of variation: {result['statistics']['cov']:.4f}")
    print(f"95% CI: [{result['statistics']['percentiles']['2.5']:.2f}, {result['statistics']['percentiles']['97.5']:.2f}] m^3/yr")
```

### Chemical Process Model

This example analyzes a chemical reactor model with uncertainties in reaction parameters:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Simplified Chemical Reactor Model
def reactor_model(X):
    """
    Simple model of a chemical reactor with Arrhenius kinetics
    X[:, 0] = Temperature (K)
    X[:, 1] = Residence time (min)
    X[:, 2] = Initial concentration (mol/L)
    X[:, 3] = Pre-exponential factor (1/min)
    X[:, 4] = Activation energy (J/mol)
    """
    T, t, C0, A, Ea = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    
    # Gas constant in J/(mol·K)
    R = 8.314
    
    # Reaction rate constant (Arrhenius equation)
    k = A * np.exp(-Ea/(R*T))
    
    # Conversion for first-order reaction
    conversion = 1 - np.exp(-k * t)
    
    # Yield calculation (with a penalty for high temperature due to side reactions)
    yield_factor = conversion * (1 - 0.1 * np.maximum(0, (T - 600)/100)**2)
    
    # Productivity (throughput)
    productivity = yield_factor * C0
    
    # Energy cost factor
    energy_cost = 0.01 * T * t
    
    # Overall objective (maximize productivity, minimize energy)
    return productivity - energy_cost

# Setup data configuration
data_info = {
    'variables': [
        {
            'name': 'Temperature',
            'vars_type': 'design_vars',
            'range_bounds': [500, 700],  # Kelvin
            'std': 10.0,  # Standard deviation of 10K
            'distribution': 'normal',
            'description': 'Reactor temperature (K) with std-based control accuracy'
        },
        {
            'name': 'ResidenceTime',
            'vars_type': 'design_vars',
            'range_bounds': [5, 30],  # minutes
            'delta': 1.0,  # Uniform uncertainty of ±1 minute
            'distribution': 'uniform',
            'description': 'Residence time (min) with uniform uncertainty'
        },
        {
            'name': 'InitialConcentration',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 2.0,  # mol/L
            'std': 0.1,   # standard deviation
            'description': 'Initial reactant concentration (mol/L)'
        },
        {
            'name': 'PreExponentialFactor',
            'vars_type': 'env_vars',
            'distribution': 'lognormal',
            'mean': np.log(1e6),  # log space
            'std': 0.2,           # log space
            'description': 'Reaction rate pre-exponential factor (1/min)'
        },
        {
            'name': 'ActivationEnergy',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 80000,  # 80 kJ/mol
            'std': 4000,    # 4 kJ/mol
            'description': 'Reaction activation energy (J/mol)'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/reactor.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance and run analysis
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/reactor.json",
    true_func=reactor_model,
    output_dir="RESULT_REACTOR"
)

# Run analysis
results = uq.run_analysis(
    num_design_samples=200,
    num_mcs_samples=5000
)

# Find the design point with highest mean response
best_idx = results['Mean'].idxmax()
best_design = {
    'Temperature': results.loc[best_idx, 'Temperature'],
    'ResidenceTime': results.loc[best_idx, 'ResidenceTime']
}

print(f"Best design found: {best_design}")
print(f"Mean performance: {results.loc[best_idx, 'Mean']:.4f}")
print(f"Standard deviation: {results.loc[best_idx, 'StdDev']:.4f}")

# Analyze this optimal point in more detail
result_optimal = uq.analyze_specific_point(
    design_point=best_design,
    num_mcs_samples=10000,
    create_pdf=True,
    create_reliability=True
)

# Analyze a robust design point (may not have highest mean, but lower variability)
# Find design with highest ratio of mean to standard deviation
robustness_metric = results['Mean'] / results['StdDev']
robust_idx = robustness_metric.idxmax()
robust_design = {
    'Temperature': results.loc[robust_idx, 'Temperature'],
    'ResidenceTime': results.loc[robust_idx, 'ResidenceTime']
}

print(f"\nMost robust design: {robust_design}")
print(f"Mean performance: {results.loc[robust_idx, 'Mean']:.4f}")
print(f"Standard deviation: {results.loc[robust_idx, 'StdDev']:.4f}")
print(f"Coefficient of variation: {results.loc[robust_idx, 'StdDev']/results.loc[robust_idx, 'Mean']:.4f}")

# Analyze this robust point in more detail
result_robust = uq.analyze_specific_point(
    design_point=robust_design,
    num_mcs_samples=10000,
    create_pdf=True,
    create_reliability=True
)
```

### Circuit Design Problem

This example analyzes an electronic circuit with tolerance uncertainties in components:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Circuit performance model (voltage divider with passive components)
def circuit_model(X):
    """
    Simple circuit performance model for a low-pass RC filter circuit
    X[:, 0] = R1 (kΩ) - First resistor
    X[:, 1] = R2 (kΩ) - Second resistor
    X[:, 2] = C (μF) - Capacitor
    X[:, 3] = Vin (V) - Input voltage
    X[:, 4] = f (kHz) - Frequency
    X[:, 5] = T (°C) - Temperature
    """
    R1, R2, C, Vin, f, T = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    
    # Convert units
    R1 = R1 * 1e3  # kΩ to Ω
    R2 = R2 * 1e3  # kΩ to Ω
    C = C * 1e-6   # μF to F
    f = f * 1e3    # kHz to Hz
    
    # Temperature effects on resistors (typical TCR for carbon resistors)
    TCR = 1000e-6  # ppm/°C
    R1_T = R1 * (1 + TCR * (T - 25))
    R2_T = R2 * (1 + TCR * (T - 25))
    
    # Angular frequency
    omega = 2 * np.pi * f
    
    # Calculate impedance of the capacitor
    Zc = 1 / (1j * omega * C)
    
    # Voltage divider with capacitor in parallel with R2
    Z_parallel = (R2_T * Zc) / (R2_T + Zc)
    Vout = Vin * (Z_parallel / (R1_T + Z_parallel))
    
    # Get magnitude and phase
    magnitude = np.abs(Vout)
    phase = np.angle(Vout, deg=True)
    
    # Create a combined performance metric (maximize magnitude at target frequency,
    # minimize deviation from target phase)
    target_phase = -45  # desired phase shift
    performance = magnitude - 0.1 * np.abs(phase - target_phase)
    
    return performance

# Setup data configuration
data_info = {
    'variables': [
        {
            'name': 'R1',
            'vars_type': 'design_vars',
            'range_bounds': [1.0, 10.0],  # kΩ
            'cov': 0.05,  # 5% tolerance (using CoV since it's standard for resistors)
            'distribution': 'normal',
            'description': 'First resistor (kΩ) with CoV-based tolerance'
        },
        {
            'name': 'R2',
            'vars_type': 'design_vars',
            'range_bounds': [1.0, 10.0],  # kΩ
            'cov': 0.05,  # 5% tolerance
            'distribution': 'normal',
            'description': 'Second resistor (kΩ)'
        },
        {
            'name': 'C',
            'vars_type': 'design_vars',
            'range_bounds': [0.1, 10.0],  # μF
            'cov': 0.10,  # 10% tolerance (typical for capacitors)
            'distribution': 'normal',
            'description': 'Capacitor (μF)'
        },
        {
            'name': 'Vin',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 5.0,   # 5V
            'std': 0.25,   # 0.25V (5% of nominal)
            'description': 'Input voltage (V)'
        },
        {
            'name': 'Frequency',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 0.5,    # 0.5 kHz
            'high': 5.0,   # 5 kHz
            'description': 'Operating frequency (kHz)'
        },
        {
            'name': 'Temperature',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 25,    # 25°C (room temperature)
            'std': 15,     # ±15°C
            'description': 'Operating temperature (°C)'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/circuit.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance and run analysis
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/circuit.json",
    true_func=circuit_model,
    output_dir="RESULT_CIRCUIT"
)

# Run analysis
results = uq.run_analysis(
    num_design_samples=200,
    num_mcs_samples=5000
)

# Find optimal design
best_idx = results['Mean'].idxmax()
best_design = {
    'R1': results.loc[best_idx, 'R1'],
    'R2': results.loc[best_idx, 'R2'],
    'C': results.loc[best_idx, 'C']
}

print(f"Best design found: {best_design}")
print(f"Mean performance: {results.loc[best_idx, 'Mean']:.4f}")
print(f"Standard deviation: {results.loc[best_idx, 'StdDev']:.4f}")

# Find robust design (high mean, low std dev)
robustness_metric = results['Mean'] / results['StdDev']
robust_idx = robustness_metric.idxmax()
robust_design = {
    'R1': results.loc[robust_idx, 'R1'],
    'R2': results.loc[robust_idx, 'R2'],
    'C': results.loc[robust_idx, 'C']
}

print(f"\nMost robust design: {robust_design}")
print(f"Mean performance: {results.loc[robust_idx, 'Mean']:.4f}")
print(f"Standard deviation: {results.loc[robust_idx, 'StdDev']:.4f}")

# Analyze both designs with detailed analysis
result_optimal = uq.analyze_specific_point(
    design_point=best_design,
    num_mcs_samples=10000,
    create_pdf=True,
    create_reliability=True
)

result_robust = uq.analyze_specific_point(
    design_point=robust_design,
    num_mcs_samples=10000,
    create_pdf=True,
    create_reliability=True
)
```


## 5. Advance Usage {#advance-usage}

### Comparison of Different Uncertainty Specifications

This example compares all three methods of specifying uncertainty (CoV, Std, and Delta) on the same problem:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation
import matplotlib.pyplot as plt

# Define a simple exponential function
def exp_function(X):
    return np.exp(X[:, 0])

# Create three different configurations
# 1. Using CoV
data_info_cov = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [0, 3],
            'cov': 0.1,  # 10% coefficient of variation
            'distribution': 'normal',
            'description': 'Using CoV for uncertainty'
        }
    ]
}

# 2. Using Std
data_info_std = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [0, 3],
            'std': 0.2,  # Fixed standard deviation
            'distribution': 'normal',
            'description': 'Using Std for uncertainty'
        }
    ]
}

# 3. Using Delta (uniform)
data_info_delta = {
    'variables': [
        {
            'name': 'x',
            'vars_type': 'design_vars',
            'range_bounds': [0, 3],
            'delta': 0.2,  # Half-width for uniform distribution
            'distribution': 'uniform',
            'description': 'Using Delta for uniform uncertainty'
        }
    ]
}

# Save configurations
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)

with open("DATA_PREPARATION/exp_cov.json", 'w') as f:
    json.dump(data_info_cov, f)
    
with open("DATA_PREPARATION/exp_std.json", 'w') as f:
    json.dump(data_info_std, f)
    
with open("DATA_PREPARATION/exp_delta.json", 'w') as f:
    json.dump(data_info_delta, f)

# Create UQ instances
uq_cov = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/exp_cov.json",
    true_func=exp_function,
    output_dir="RESULT_EXP_COV"
)

uq_std = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/exp_std.json",
    true_func=exp_function,
    output_dir="RESULT_EXP_STD"
)

uq_delta = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/exp_delta.json",
    true_func=exp_function,
    output_dir="RESULT_EXP_DELTA"
)

# Run analyses
results_cov = uq_cov.run_analysis(num_design_samples=200, num_mcs_samples=5000)
results_std = uq_std.run_analysis(num_design_samples=200, num_mcs_samples=5000)
results_delta = uq_delta.run_analysis(num_design_samples=200, num_mcs_samples=5000)

# Create comparison plot
plt.figure(figsize=(15, 10))

# Plot means
plt.subplot(2, 1, 1)
plt.plot(results_cov['x'], results_cov['Mean'], 'r-', label='CoV (Normal)')
plt.plot(results_std['x'], results_std['Mean'], 'g-', label='Std (Normal)')
plt.plot(results_delta['x'], results_delta['Mean'], 'b-', label='Delta (Uniform)')
plt.xlabel('x')
plt.ylabel('Mean')
plt.title('Comparison of Mean Response')
plt.legend()
plt.grid(True)

# Plot standard deviations
plt.subplot(2, 1, 2)
plt.plot(results_cov['x'], results_cov['StdDev'], 'r-', label='CoV (Normal)')
plt.plot(results_std['x'], results_std['StdDev'], 'g-', label='Std (Normal)')
plt.plot(results_delta['x'], results_delta['StdDev'], 'b-', label='Delta (Uniform)')
plt.xlabel('x')
plt.ylabel('Standard Deviation')
plt.title('Comparison of Uncertainty Propagation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('uncertainty_specification_comparison.png')
plt.show()

# Analysis at a specific point
x_value = 2.0
design_point = {'x': x_value}

result_cov = uq_cov.analyze_specific_point(
    design_point=design_point,
    num_mcs_samples=10000
)

result_std = uq_std.analyze_specific_point(
    design_point=design_point,
    num_mcs_samples=10000
)

result_delta = uq_delta.analyze_specific_point(
    design_point=design_point,
    num_mcs_samples=10000
)

print(f"\nComparison at x = {x_value}:")
print(f"1. CoV method:")
print(f"   Mean: {result_cov['statistics']['mean']:.4f}")
print(f"   Std: {result_cov['statistics']['std']:.4f}")
print(f"   95% CI: [{result_cov['statistics']['percentiles']['2.5']:.4f}, {result_cov['statistics']['percentiles']['97.5']:.4f}]")

print(f"\n2. Std method:")
print(f"   Mean: {result_std['statistics']['mean']:.4f}")
print(f"   Std: {result_std['statistics']['std']:.4f}")
print(f"   95% CI: [{result_std['statistics']['percentiles']['2.5']:.4f}, {result_std['statistics']['percentiles']['97.5']:.4f}]")

print(f"\n3. Delta method (uniform):")
print(f"   Mean: {result_delta['statistics']['mean']:.4f}")
print(f"   Std: {result_delta['statistics']['std']:.4f}")
print(f"   95% CI: [{result_delta['statistics']['percentiles']['2.5']:.4f}, {result_delta['statistics']['percentiles']['97.5']:.4f}]")
```

### Using Environmental Variables with Different Specifications

This example demonstrates how to use environmental variables with different uncertainty specifications:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation

# Simple function to demonstrate environmental variable effects
def env_sensitivity_function(X):
    """
    Function showing sensitivity to different environmental variables
    X[:, 0] = Design variable
    X[:, 1] = Normal env variable with std
    X[:, 2] = Normal env variable with cov
    X[:, 3] = Uniform env variable with min/max
    X[:, 4] = Uniform env variable with low/high
    """
    dv = X[:, 0]         # Design variable
    env1 = X[:, 1]        # Normal with std
    env2 = X[:, 2]        # Normal with cov
    env3 = X[:, 3]        # Uniform with min/max
    env4 = X[:, 4]        # Uniform with low/high
    
    # Base response depends on design variable
    base = dv**2
    
    # Environmental effects
    effect1 = 2 * env1   # Linear effect
    effect2 = env2**2    # Quadratic effect
    effect3 = np.sin(3 * env3)  # Oscillatory effect
    effect4 = np.exp(0.1 * env4)  # Exponential effect
    
    return base + effect1 + effect2 + effect3 + effect4

# Setup data configuration with different ways to specify env variables
data_info = {
    'variables': [
        {
            'name': 'design_var',
            'vars_type': 'design_vars',
            'range_bounds': [0, 5],
            'std': 0.2,
            'distribution': 'normal',
            'description': 'Design variable with std'
        },
        {
            'name': 'env_normal_std',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 0,
            'std': 0.5,  # Using std directly
            'description': 'Normal env variable with std'
        },
        {
            'name': 'env_normal_cov',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 2.0,
            'cov': 0.1,  # Using cov instead of std
            'description': 'Normal env variable with cov'
        },
        {
            'name': 'env_uniform_minmax',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'min': -1.0,  # Using min/max notation
            'max': 1.0,
            'description': 'Uniform env variable with min/max'
        },
        {
            'name': 'env_uniform_lowhigh',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 0.0,  # Using low/high notation (for backward compatibility)
            'high': 2.0,
            'description': 'Uniform env variable with low/high'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/env_variables_demo.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/env_variables_demo.json",
    true_func=env_sensitivity_function,
    output_dir="RESULT_ENV_VARS_DEMO"
)

# Run analysis
results = uq.run_analysis(
    num_design_samples=100,
    num_mcs_samples=5000
)

# Analyze a specific point with environmental variable overrides
design_point = {'design_var': 2.5}

# Default environmental conditions
result_default = uq.analyze_specific_point(
    design_point=design_point,
    num_mcs_samples=10000
)

# Override with different specifications
env_override = {
    'env_normal_std': {'mean': 1.0, 'std': 1.0},  # Override using std
    'env_normal_cov': {'mean': 3.0, 'cov': 0.2},  # Override using cov
    'env_uniform_minmax': {'min': -2.0, 'max': 0.0},  # Override using min/max
    'env_uniform_lowhigh': {'low': 1.0, 'high': 3.0}   # Override using low/high
}

result_override = uq.analyze_specific_point(
    design_point=design_point,
    env_vars_override=env_override,
    num_mcs_samples=10000
)

# Print comparison
print(f"Analysis at design_var = {design_point['design_var']}:")
print("\nDefault Environmental Conditions:")
print(f"Mean: {result_default['statistics']['mean']:.4f}")
print(f"Std: {result_default['statistics']['std']:.4f}")
print(f"95% CI: [{result_default['statistics']['percentiles']['2.5']:.4f}, {result_default['statistics']['percentiles']['97.5']:.4f}]")

print("\nModified Environmental Conditions:")
print(f"Mean: {result_override['statistics']['mean']:.4f}")
print(f"Std: {result_override['statistics']['std']:.4f}")
print(f"95% CI: [{result_override['statistics']['percentiles']['2.5']:.4f}, {result_override['statistics']['percentiles']['97.5']:.4f}]")
```

### Environmental Variable Sensitivity Analysis

This example shows how to perform a basic sensitivity analysis with different environmental variable configurations:

```python
import numpy as np
from PyEGRO.uncertainty.UQmcs import UncertaintyPropagation
import matplotlib.pyplot as plt

# Define a function sensitive to environmental variables
def env_sensitivity(X):
    """
    Function with design and environmental variables
    X[:, 0] = Design variable (temperature)
    X[:, 1] = Environmental variable 1 (humidity)
    X[:, 2] = Environmental variable 2 (pressure)
    """
    temp = X[:, 0]
    humidity = X[:, 1]
    pressure = X[:, 2]
    
    # Complex interaction between variables
    return temp**2 + 5*humidity -. + 0.2*pressure + 0.5*temp*humidity - 0.1*temp*pressure

# Create data configuration
data_info = {
    'variables': [
        {
            'name': 'temperature',
            'vars_type': 'design_vars',
            'range_bounds': [20, 30],
            'std': 0.5,  # Standard deviation of 0.5°C
            'distribution': 'normal',
            'description': 'Operating temperature (°C)'
        },
        {
            'name': 'humidity',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 50,
            'std': 10,  # Standard deviation of 10%
            'description': 'Relative humidity (%)'
        },
        {
            'name': 'pressure',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 101.3,
            'std': 2.0,  # Standard deviation of 2 kPa
            'description': 'Atmospheric pressure (kPa)'
        }
    ]
}

# Save configuration
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/sensitivity.json", 'w') as f:
    json.dump(data_info, f)

# Create UncertaintyPropagation instance
uq = UncertaintyPropagation(
    data_info_path="DATA_PREPARATION/sensitivity.json",
    true_func=env_sensitivity,
    output_dir="RESULT_SENSITIVITY"
)

# Run analysis
results = uq.run_analysis(
    num_design_samples=50,
    num_mcs_samples=5000
)

# Define different environmental scenarios
nominal_point = {'temperature': 25.0}
scenarios = [
    {"name": "Nominal", "override": None},
    {"name": "High Humidity", "override": {'humidity': {'mean': 80, 'std': 5}}},
    {"name": "Low Humidity", "override": {'humidity': {'mean': 20, 'std': 5}}},
    {"name": "High Pressure", "override": {'pressure': {'mean': 110, 'std': 1}}},
    {"name": "Low Pressure", "override": {'pressure': {'mean': 90, 'std': 1}}},
    {"name": "Extreme Conditions", "override": {
        'humidity': {'mean': 85, 'std': 8},
        'pressure': {'mean': 88, 'std': 3}
    }}
]

# Analyze each scenario
scenario_results = []
for scenario in scenarios:
    result = uq.analyze_specific_point(
        design_point=nominal_point,
        env_vars_override=scenario["override"],
        num_mcs_samples=10000,
        create_pdf=True,
        create_reliability=True
    )
    
    scenario_results.append({
        "name": scenario["name"],
        "mean": result['statistics']['mean'],
        "std": result['statistics']['std'],
        "cov": result['statistics']['cov'],
        "ci_lower": result['statistics']['percentiles']['2.5'],
        "ci_upper": result['statistics']['percentiles']['97.5']
    })

# Create comparison plot
plt.figure(figsize=(12, 8))

# Plot means with error bars
means = [r["mean"] for r in scenario_results]
stds = [r["std"] for r in scenario_results]
names = [r["name"] for r in scenario_results]
x_pos = np.arange(len(names))

plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, capsize=10)
plt.xticks(x_pos, names, rotation=45)
plt.ylabel('Mean Response')
plt.title('Environmental Sensitivity Analysis')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, (mean, std) in enumerate(zip(means, stds)):
    plt.text(i, mean + std + 0.5, f"{mean:.1f}±{std:.1f}", ha='center')

plt.tight_layout()
plt.savefig('env_sensitivity_analysis.png')
plt.show()

# Print detailed results
print("\nDetailed Scenario Results:")
print("-" * 80)
print(f"{'Scenario':<20} {'Mean':<8} {'StdDev':<8} {'CoV':<8} {'95% CI':<20}")
print("-" * 80)
for r in scenario_results:
    print(f"{r['name']:<20} {r['mean']:<8.2f} {r['std']:<8.2f} {r['cov']:<8.2f} [{r['ci_lower']:<8.2f}, {r['ci_upper']:<8.2f}]")
```

### Remarks

The UQmcs module now supports multiple ways to specify uncertainty in both design and environmental variables, making it more flexible and robust for various engineering scenarios. The key improvements include:

1. **Support for both CoV and Std**: 
   - Use `cov` when uncertainty scales with the variable's value
   - Use `std` for direct control of uncertainty width or when dealing with variables that cross zero

2. **Uniform uncertainty with Delta parameter**:
   - The `delta` parameter provides a simple way to specify uniform uncertainty around design points

3. **Consistent notation for bounds**:
   - Environmental variables now support both `low`/`high` and `min`/`max` notation for better compatibility

These features allow for more accurate uncertainty modeling across a wide range of problems, from mechanical design to chemical processes to electronic circuits.

When choosing between uncertainty specification methods, consider:
- Use `std` for variables whose range crosses zero
- Use `cov` for positive variables where uncertainty scales with magnitude
- Use `delta` with uniform distribution for manufacturing tolerances or when uncertainty has sharp bounds

For more information, refer to the UQmcs API Reference document.

### ** Load .fig.pkl for ploting

```python
import pickle
import matplotlib.pyplot as plt

# Load the pickled figure
with open('uncertainty_analysis.fig.pkl', 'rb') as file:
    fig = pickle.load(file)

# Display the figure
plt.show()
```