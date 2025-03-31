# Efficient Global Optimization (EGO) / Usage


### **<span style='color: teal;'>Basic Usage Examples</span>**

To access usage information

```python

import PyEGRO.ego
PyEGRO.ego.print_usage()

```

---



**Example 1: Optimizing a Quadratic Function**

```python
import numpy as np
import pandas as pd
from PyEGRO.ego import EfficientGlobalOptimization, TrainingConfig

# Define the objective function
def objective_function(x):
    x1, x2 = x[:,0], x[:,1]
    y = x1**2 + x2**2
    return y

# Set bounds and variable names
bounds = np.array([[-5, 5], [-5, 5]])
variable_names = ['x1', 'x2']

# Generate initial data
n_initial = 5
initial_x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_initial, len(variable_names)))
initial_y = objective_function(initial_x)
initial_data = pd.DataFrame(initial_x, columns=variable_names)
initial_data['y'] = initial_y

# Configure EGO
config = TrainingConfig(
    max_iterations=20,
    rmse_threshold=0.001,
    acquisition_name="eigf"
)

# Run EGO
ego = EfficientGlobalOptimization(
    objective_func=objective_function,
    bounds=bounds,
    variable_names=variable_names,
    config=config,
    initial_data=initial_data
)
result = ego.run()

# Output the optimization history
print(result.history)
```

---

### **<span style='color: teal;'>Advanced Usage</span>**


**Example 2: Optimizing with Preloaded Data**

```python
import json
import pandas as pd
from PyEGRO.ego import EfficientGlobalOptimization, TrainingConfig

# Load pre-existing data
with open('data_info.json', 'r') as f:
    data_info = json.load(f)
initial_data = pd.read_csv('training_data.csv')

# Extract bounds and variable names from data info
bounds = np.array(data_info['input_bound'])
variable_names = [var['name'] for var in data_info['variables']]

# Define the objective function
# Define the objective function
def objective_function(x):
    x1, x2 = x[:,0], x[:,1]
    y = x1**2 + x2**2
    return y

# Configure EGO
config = TrainingConfig(
    max_iterations=50,
    rmse_threshold=0.0001,
    acquisition_name="eigf"
)

# Run EGO
ego = EfficientGlobalOptimization(
    objective_func=objective_function,
    bounds=bounds,
    variable_names=variable_names,
    config=config,
    initial_data=initial_data
)
result = ego.run()

# Save the results
result.save("output_directory")
```


**Configuration Options**

**TrainingConfig() Parameters**:

- **max\_iterations**: Maximum number of optimization iterations (default: 100).

- **rmse\_threshold**: Stopping criteria based on RMSE improvement (default: 0.001).

- **acquisition\_name**: Choice of acquisition function (e.g., "ei", "e3i", "eigf").

- **device**: Specifies whether to use "cpu" or "cuda" (default: "cpu").

- **save\_dir**: Directory to save the results and trained model.

- **verbose**: Displays progress during the optimization (default: True).

---

This document provides examples for using the EGO module in PyEGRO to solve complex optimization problems, preloading data, and loading trained models for predictions. For further assistance, refer to the PyEGRO documentation or contact support.

