# Metamodel-Based Sensitivity Analysis Benchmarking

This document demonstrates the implementation of benchmark functions for sensitivity analysis using metamodels with the PyEGRO library, comparing with the analytical benchmarks presented in the previous document.

## 1. Introduction

For computationally expensive models, using a metamodel (surrogate model) can significantly reduce the computational cost of sensitivity analysis. In this benchmarking, we'll implement the same three well-known test functions using Gaussian Process Regression (GPR) metamodels:

1. Ishigami function
2. Hartmann 6-D function
3. A1 function (k-function)

Instead of directly computing sensitivity indices from the original functions, we'll first train a metamodel on a limited number of samples, then use the trained metamodel for sensitivity analysis.

## 2. Metamodel with PyEGRO

### 2.1 Ishigami Function Implementation

#### 2.1.1 Generate Training and Testing Data

```python
from PyEGRO.doe.initial_design import InitialDesign
import numpy as np

def ishigami_function(X):
    """Ishigami function implementation"""
    a = 7
    b = 0.1
    
    # Extract single sample or first sample from batch
    if X.ndim > 1:
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
    else:
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
    
    return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

# Create training data set
sampling_training = InitialDesign(
    sampling_method='lhs',  # Latin Hypercube Sampling
    results_filename='training_data_ishigami',
    show_progress=True
)
   
# Add design variables
sampling_training.add_design_variable(
    name='x1',
    range_bounds=[-np.pi, np.pi],
    distribution='uniform',
    description='First input variable'
)

sampling_training.add_design_variable(
    name='x2',
    range_bounds=[-np.pi, np.pi],
    distribution='uniform',
    description='Second input variable'
)

sampling_training.add_design_variable(
    name='x3',
    range_bounds=[-np.pi, np.pi],
    distribution='uniform',
    description='Third input variable'
)

# Save the design configuration
sampling_training.save('data_info_ishigami')

# Generate samples and evaluate the function
training_results = sampling_training.run(
    objective_function=ishigami_function,
    num_samples=200  # Number of training samples
)

# Create testing data set for model validation
sampling_testing = InitialDesign(
    sampling_method='lhs',
    results_filename='testing_data_ishigami',
    show_progress=True
)

# Copy the same variable definitions
for var in sampling_training.variables:
    sampling_testing.add_design_variable(**var)
   
# Generate testing samples
testing_results = sampling_testing.run(
    objective_function=ishigami_function,
    num_samples=50  # Number of testing samples
)
```

#### 2.1.2 Train Gaussian Process Regression (GPR) Metamodel

```python
import pandas as pd
import json
import numpy as np
from PyEGRO.meta.gpr import MetaTraining
from PyEGRO.meta.gpr.visualization import visualize_gpr

# Load initial data and problem configuration
with open('DATA_PREPARATION/data_info_ishigami.json', 'r') as f:
    data_info = json.load(f)

# Load training and testing data
training_data = pd.read_csv('DATA_PREPARATION/training_data_ishigami.csv')
test_data = pd.read_csv('DATA_PREPARATION/testing_data_ishigami.csv')

# Get variable information
variable_names = [var['name'] for var in data_info['variables']]
target_column = 'y'  # Default target column in PyEGRO

# Extract features and targets
X_train = training_data[variable_names].values
y_train = training_data[target_column].values.reshape(-1, 1)
X_test = test_data[variable_names].values
y_test = test_data[target_column].values.reshape(-1, 1)

# Get bounds for visualization
bounds = np.array([
    [var['range_bounds'][0], var['range_bounds'][1]] 
    for var in data_info['variables']
])

# Initialize and train GPR model
print("Training GPR model for Ishigami function...")
meta = MetaTraining(
    num_iterations=1000,
    prefer_gpu=True,           # Use GPU if available
    show_progress=True,
    output_dir='RESULT_MODEL_GPR_ISHIGAMI',
    kernel='matern25',         # Matérn kernel with ν=2.5
    learning_rate=0.01,
    patience=50                # Early stopping patience
)

# Train the model
model, scaler_X, scaler_y = meta.train(
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=variable_names
)

# Generate visualization of the surrogate model
figures = visualize_gpr(
    meta=meta,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    variable_names=variable_names,
    bounds=bounds,
    savefig=True
)
```

#### 2.1.3 Sensitivity Analysis Using the Metamodel

```python
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis
from PyEGRO.meta.gpr.gpr_utils import DeviceAgnosticGPR

# Load the pre-trained Metamodel
model_handler = DeviceAgnosticGPR(prefer_gpu=True)
model_handler.load_model('RESULT_MODEL_GPR_ISHIGAMI')

# Run sensitivity analysis with the Metamodel
results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/data_info_ishigami.json",
    model_handler=model_handler,
    num_samples=8192,  # Can use more samples since evaluations are cheap with metamodel
    output_dir="RESULT_SA_METAMODEL_ISHIGAMI",
    random_seed=42,
    show_progress=True
)
```

![Sensitivity Indices for Ishigami Function using Metamodel](sensitivity_indices_metamodel_Ishigami.png){ width="800" }


### 2.2 Hartmann 6-D Function Implementation

#### 2.2.1 Generate Training and Testing Data

```python
from PyEGRO.doe.initial_design import InitialDesign
import numpy as np

def hartmann_6d_function(X):
    """
    Hartmann 6-D function - benchmark for sensitivity analysis.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    
    # Handle single sample or batch of samples
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    n_samples = X.shape[0]
    result = np.zeros(n_samples)
    
    for i in range(n_samples):
        outer_sum = 0
        for j in range(4):
            inner_sum = 0
            for k in range(6):
                inner_sum += A[j, k] * (X[i, k] - P[j, k])**2
            outer_sum += alpha[j] * np.exp(-inner_sum)
        result[i] = -outer_sum
    
    return result

# Create training data set
sampling_training = InitialDesign(
    sampling_method='lhs',
    results_filename='training_data_hartmann',
    show_progress=True
)
   
# Add design variables
for i in range(6):
    sampling_training.add_design_variable(
        name=f'x{i+1}',
        range_bounds=[0, 1],
        distribution='uniform',
        description=f'Input variable {i+1}'
    )

# Save the design configuration
sampling_training.save('data_info_hartmann')

# Generate samples and evaluate the function
training_results = sampling_training.run(
    objective_function=hartmann_6d_function,
    num_samples=500  # Increased samples for higher dimensional function
)

# Create testing data set for model validation
sampling_testing = InitialDesign(
    sampling_method='lhs',
    results_filename='testing_data_hartmann',
    show_progress=True
)

# Copy the same variable definitions
for var in sampling_training.variables:
    sampling_testing.add_design_variable(**var)
   
# Generate testing samples
testing_results = sampling_testing.run(
    objective_function=hartmann_6d_function,
    num_samples=100
)
```

#### 2.2.2 Train and Apply Metamodel for Sensitivity Analysis

```python
import pandas as pd
import json
import numpy as np
from PyEGRO.meta.gpr import MetaTraining
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis
from PyEGRO.meta.gpr.gpr_utils import DeviceAgnosticGPR

# Load and prepare data
with open('DATA_PREPARATION/data_info_hartmann.json', 'r') as f:
    data_info = json.load(f)

training_data = pd.read_csv('DATA_PREPARATION/training_data_hartmann.csv')
test_data = pd.read_csv('DATA_PREPARATION/testing_data_hartmann.csv')

variable_names = [var['name'] for var in data_info['variables']]
target_column = 'y'

X_train = training_data[variable_names].values
y_train = training_data[target_column].values.reshape(-1, 1)
X_test = test_data[variable_names].values
y_test = test_data[target_column].values.reshape(-1, 1)

# Train GPR model
meta = MetaTraining(
    num_iterations=1500,
    prefer_gpu=True,
    show_progress=True,
    output_dir='RESULT_MODEL_GPR_HARTMANN',
    kernel='matern25',
    learning_rate=0.01,
    patience=50
)

model, scaler_X, scaler_y = meta.train(
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=variable_names
)

# Load model and run sensitivity analysis
model_handler = DeviceAgnosticGPR(prefer_gpu=True)
model_handler.load_model('RESULT_MODEL_GPR_HARTMANN')

results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/data_info_hartmann.json",
    model_handler=model_handler,
    num_samples=16384,
    output_dir="RESULT_SA_METAMODEL_HARTMANN",
    random_seed=42,
    show_progress=True
)
```

![Sensitivity Indices for Hartmann Function using Metamodel](sensitivity_indices_metamodel_Hartmann.png){ width="800" }


### 2.3 A1 Function (k-function) Implementation

#### 2.3.1 Generate Training and Testing Data

```python
from PyEGRO.doe.initial_design import InitialDesign
import numpy as np

def a1_function(X):
    """
    A1 function (k-function) - benchmark for sensitivity analysis.
    Formula: sum_{j=1}^{k} (-1)^j * prod_{i=1}^j x_i
    """
    # Handle single sample or batch of samples
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    n_samples = X.shape[0]
    n_vars = X.shape[1]
    result = np.zeros(n_samples)
    
    for i in range(n_samples):
        sample_result = 0
        for j in range(1, n_vars + 1):
            term = (-1)**j
            for k in range(j):
                term *= X[i, k]
            sample_result += term
        result[i] = sample_result
    
    return result

# Create training data set
sampling_training = InitialDesign(
    sampling_method='lhs',
    results_filename='training_data_a1',
    show_progress=True
)
   
# Add design variables (n = 10 as specified in the paper)
for i in range(10):
    sampling_training.add_design_variable(
        name=f'x{i+1}',
        range_bounds=[0, 1],
        distribution='uniform',
        description=f'Input variable {i+1}'
    )

# Save the design configuration
sampling_training.save('data_info_a1')

# Generate samples and evaluate the function
training_results = sampling_training.run(
    objective_function=a1_function,
    num_samples=1000  # More samples for higher dimensional function
)

# Create testing data set for model validation
sampling_testing = InitialDesign(
    sampling_method='lhs',
    results_filename='testing_data_a1',
    show_progress=True
)

# Copy the same variable definitions
for var in sampling_training.variables:
    sampling_testing.add_design_variable(**var)
   
# Generate testing samples
testing_results = sampling_testing.run(
    objective_function=a1_function,
    num_samples=200
)
```

#### 2.3.2 Train and Apply Metamodel for Sensitivity Analysis

```python
import pandas as pd
import json
import numpy as np
from PyEGRO.meta.gpr import MetaTraining
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis
from PyEGRO.meta.gpr.gpr_utils import DeviceAgnosticGPR

# Load and prepare data
with open('DATA_PREPARATION/data_info_a1.json', 'r') as f:
    data_info = json.load(f)

training_data = pd.read_csv('DATA_PREPARATION/training_data_a1.csv')
test_data = pd.read_csv('DATA_PREPARATION/testing_data_a1.csv')

variable_names = [var['name'] for var in data_info['variables']]
target_column = 'y'

X_train = training_data[variable_names].values
y_train = training_data[target_column].values.reshape(-1, 1)
X_test = test_data[variable_names].values
y_test = test_data[target_column].values.reshape(-1, 1)

# Train GPR model
meta = MetaTraining(
    num_iterations=2000,
    prefer_gpu=True,
    show_progress=True,
    output_dir='RESULT_MODEL_GPR_A1',
    kernel='matern25',
    learning_rate=0.01,
    patience=100
)

model, scaler_X, scaler_y = meta.train(
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=variable_names
)

# Load model and run sensitivity analysis
model_handler = DeviceAgnosticGPR(prefer_gpu=True)
model_handler.load_model('RESULT_MODEL_GPR_A1')

results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/data_info_a1.json",
    model_handler=model_handler,
    num_samples=32768,
    output_dir="RESULT_SA_METAMODEL_A1",
    random_seed=42,
    show_progress=True
)
```

![Sensitivity Indices for A1 k-function using Metamodel](sensitivity_indices_metamodel_k_function.png){ width="800" }

## 3. Conclusion

Metamodel-based sensitivity analysis provides a powerful approach for analyzing computationally expensive models. The key advantages include:

1. **Computational Efficiency**: Only a limited number of original function evaluations are needed
2. **Flexibility**: Works with various types of models, including black-box functions
3. **Scalability**: Makes sensitivity analysis feasible for complex simulation models

When comparing with direct sensitivity analysis methods, metamodel-based approaches trade some accuracy for significant computational savings. The accuracy of the results depends on how well the metamodel captures the behavior of the original function, which can be assessed using validation metrics.

For the benchmark functions tested here, Gaussian Process Regression metamodels provide a good balance between accuracy and computational efficiency.

## References

1. Azzini, I., & Rosati, R. (2022). A function dataset for benchmarking in sensitivity analysis. *Data in Brief*, *42*, 108071.
2. Rasmussen, C.E., & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
3. Gratiet, L.L., Marelli, S., & Sudret, B. (2017). Metamodel-based sensitivity analysis: Polynomial chaos expansions and Gaussian processes. *Handbook of Uncertainty Quantification*, 1289-1325.