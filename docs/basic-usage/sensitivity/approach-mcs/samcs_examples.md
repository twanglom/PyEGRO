# PyEGRO.sensitivity.SAmcs Usage Examples

This document provides examples of how to use the PyEGRO.sensitivity.SAmcs module for sensitivity analysis using Monte Carlo Simulation with the Sobol method.

## Table of Contents

1. [Quick Start](#quick-start)
   - [Basic Setup](#basic-setup)
   - [Running the Analysis](#running-the-analysis)
2. [Working with Different Models](#working-with-different-models)
   - [Using a Direct Function](#using-a-direct-function)
   - [Using a Surrogate Model](#using-a-surrogate-model)
3. [Example Applications](#example-applications)
   - [Beam Design Problem](#beam-design-problem)
   - [Borehole Function](#borehole-function)
   - [Chemical Reaction Model](#chemical-reaction-model)
   - [Circuit Design Problem](#circuit-design-problem)

---

## Quick Start

### Basic Setup

The simplest way to use the PyEGRO.sensitivity.SAmcs module is with the convenience function `run_sensitivity_analysis`.

```python
import numpy as np
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis

# Define a simple test function
def test_function(X):
    """
    Ishigami function - common benchmark for sensitivity analysis.
    """
    a = 7
    b = 0.1
    
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    
    return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

# Define data_info with variable definitions
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [-np.pi, np.pi],
            'description': 'First input variable'
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [-np.pi, np.pi],
            'description': 'Second input variable'
        },
        {
            'name': 'x3',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [-np.pi, np.pi],
            'description': 'Third input variable'
        }
    ]
}

# Save data_info to a JSON file
import json
import os

os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/data_info.json", 'w') as f:
    json.dump(data_info, f)

# Run the analysis
results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    true_func=test_function,
    num_samples=1024,  # Base sample size
    output_dir="RESULT_SA_ISHIGAMI",
    random_seed=42,
    show_progress=True
)

# Print the results
print("\nSensitivity Indices:")
print(results_df)
```

### Running the Analysis

For more control over the analysis process, you can use the `MCSSensitivityAnalysis` class directly:

```python
from PyEGRO.sensitivity.SAmcs import MCSSensitivityAnalysis

# Initialize the analysis
analyzer = MCSSensitivityAnalysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    true_func=test_function,
    output_dir="RESULT_SA_DETAILED",
    show_variables_info=True,
    random_seed=42
)

# Run the analysis
results_df = analyzer.run_analysis(
    num_samples=2048,  # Increase sample size for better accuracy
    show_progress=True
)

# Examine the summary statistics
import json
with open("RESULT_SA_DETAILED/analysis_summary.json", 'r') as f:
    summary = json.load(f)
    
print("\nAnalysis Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
```

## Working with Different Models

### Using a Direct Function

For analytical functions or inexpensive simulations, you can directly use the function for evaluations:

```python
import numpy as np
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis

# Define a complex function with interactions
def complex_function(X):
    """
    Function with variable interactions and non-linear effects.
    """
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # Base effects
    base = 2*x1**2 + x2 + 0.5*x3 + 0.1*x4
    
    # Interaction effects
    interactions = 5*x1*x2 + 2*x2*x3 + x3*x4 + 0.5*x1*x4
    
    # Non-linear effects
    non_linear = np.sin(x1) * np.cos(x2) + np.exp(0.1*x3) + np.log(abs(x4) + 1)
    
    return base + interactions + non_linear

# Define data_info
data_info = {
    'variables': [
        {
            'name': 'x1',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [-2, 2],
            'description': 'Primary variable'
        },
        {
            'name': 'x2',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [-2, 2],
            'cov': 0.1,
            'description': 'Secondary variable'
        },
        {
            'name': 'x3',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 0,
            'std': 1,
            'description': 'Environmental variable 1'
        },
        {
            'name': 'x4',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': -3,
            'high': 3,
            'description': 'Environmental variable 2'
        }
    ]
}

# Save data_info
import json
with open("DATA_PREPARATION/complex_data_info.json", 'w') as f:
    json.dump(data_info, f)

# Run analysis
results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/complex_data_info.json",
    true_func=complex_function,
    num_samples=2048,
    output_dir="RESULT_SA_COMPLEX",
    show_progress=True
)

# Print top influential variables
sorted_results = results_df.sort_values('Total-order', ascending=False)
print("\nVariables Ranked by Total-order Influence:")
print(sorted_results[['Parameter', 'Total-order', 'First-order']])
```

### Using a Metamodel

For computationally expensive models, you can use a Metamodel:

```python
import numpy as np
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis
from PyEGRO.meta.gpr.gpr_utils import DeviceAgnosticGPR

# Load a pre-trained Metamodel
model_handler = DeviceAgnosticGPR(prefer_gpu=True)
model_handler.load_model('RESULT_MODEL_GPR')

# Run analysis with the Metamodel
results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/data_info.json",
    model_handler=model_handler,
    num_samples=4096,  # Can use more samples since evaluations are cheap
    output_dir="RESULT_SA_SURROGATE",
    show_progress=True
)

# Examine first-order vs total-order differences to detect interactions
interaction_strength = results_df['Total-order'] - results_df['First-order']
results_df['Interaction_Strength'] = interaction_strength

print("\nVariable Interactions:")
for i, row in results_df.iterrows():
    print(f"{row['Parameter']}: Interaction Strength = {row['Interaction_Strength']:.4f}")
```

## Example Applications

### Beam Design Problem

This example demonstrates sensitivity analysis for a beam design problem:

```python
import numpy as np
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis

# Define a beam design problem
def beam_performance(X):
    """
    Calculate the maximum deflection of a cantilever beam.
    
    Variables:
    - length: Beam length (m)
    - width: Beam width (m)
    - height: Beam height (m)
    - load: Applied load (N)
    - modulus: Young's modulus (Pa)
    """
    length = X[:, 0]
    width = X[:, 1]
    height = X[:, 2]
    load = X[:, 3]
    modulus = X[:, 4]
    
    # Calculate moment of inertia
    inertia = (width * height**3) / 12
    
    # Calculate maximum deflection
    deflection = (load * length**3) / (3 * modulus * inertia)
    
    return deflection

# Define data_info
beam_data = {
    'variables': [
        {
            'name': 'length',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [1.0, 3.0],
            'cov': 0.01,
            'description': 'Beam length (m)'
        },
        {
            'name': 'width',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [0.05, 0.15],
            'cov': 0.02,
            'description': 'Beam width (m)'
        },
        {
            'name': 'height',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [0.1, 0.3],
            'cov': 0.02,
            'description': 'Beam height (m)'
        },
        {
            'name': 'load',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 1000,
            'std': 200,
            'description': 'Applied load (N)'
        },
        {
            'name': 'modulus',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 2.1e11,
            'std': 1e10,
            'description': 'Young\'s modulus (Pa)'
        }
    ]
}

# Save data_info
import json
import os
os.makedirs("DATA_PREPARATION", exist_ok=True)
with open("DATA_PREPARATION/beam_data.json", 'w') as f:
    json.dump(beam_data, f)

# Run analysis
results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/beam_data.json",
    true_func=beam_performance,
    num_samples=2048,
    output_dir="RESULT_SA_BEAM",
    show_progress=True
)

# Analyze and interpret results
print("\nBeam Design Sensitivity Analysis Results:")
print("-" * 50)
print(results_df)

# Load summary stats
with open("RESULT_SA_BEAM/analysis_summary.json", 'r') as f:
    summary = json.load(f)

print("\nDesign Insights:")
print(f"Most influential parameter: {summary['most_influential_parameter']}")
print(f"Interaction strength: {summary['interaction_strength']:.4f}")

print("\nEngineering Recommendations:")
# Sort by total-order
sorted_params = results_df.sort_values('Total-order', ascending=False)['Parameter'].tolist()

print(f"1. Focus on controlling {sorted_params[0]} for maximum effect on performance")
print(f"2. Parameters to optimize first: {', '.join(sorted_params[:2])}")
print(f"3. Parameters with minimal impact: {', '.join(sorted_params[-2:])}")
```

### Borehole Function

The borehole function is a common benchmark in engineering sensitivity analysis:

```python
import numpy as np
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis

# Define the borehole function
def borehole_function(X):
    """
    Borehole function - models water flow through a borehole.
    """
    rw = X[:, 0]  # radius of borehole (m)
    r = X[:, 1]   # radius of influence (m)
    Tu = X[:, 2]  # transmissivity of upper aquifer (m²/year)
    Hu = X[:, 3]  # potentiometric head of upper aquifer (m)
    Tl = X[:, 4]  # transmissivity of lower aquifer (m²/year)
    Hl = X[:, 5]  # potentiometric head of lower aquifer (m)
    L = X[:, 6]   # length of borehole (m)
    Kw = X[:, 7]  # hydraulic conductivity of borehole (m/year)
    
    numerator = 2 * np.pi * Tu * (Hu - Hl)
    
    denominator = np.log(r / rw) * (1 + (2 * L * Tu) / (np.log(r / rw) * rw**2 * Kw) + Tu/Tl)
    
    return numerator / denominator

# Define data_info
borehole_data = {
    'variables': [
        {
            'name': 'rw',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [0.05, 0.15],
            'description': 'Radius of borehole (m)'
        },
        {
            'name': 'r',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [100, 50000],
            'description': 'Radius of influence (m)'
        },
        {
            'name': 'Tu',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 63070, 
            'high': 115600,
            'description': 'Transmissivity of upper aquifer (m²/year)'
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
            'description': 'Transmissivity of lower aquifer (m²/year)'
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
            'distribution': 'uniform',
            'range_bounds': [1120, 1680],
            'description': 'Length of borehole (m)'
        },
        {
            'name': 'Kw',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 9855, 
            'high': 12045,
            'description': 'Hydraulic conductivity of borehole (m/year)'
        }
    ]
}

# Save data_info
import json
with open("DATA_PREPARATION/borehole_data.json", 'w') as f:
    json.dump(borehole_data, f)

# Run analysis
results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/borehole_data.json",
    true_func=borehole_function,
    num_samples=2048,
    output_dir="RESULT_SA_BOREHOLE",
    show_progress=True
)

# Print results
print("\nBorehole Function Sensitivity Analysis Results:")
print("-" * 50)
print(results_df.sort_values('Total-order', ascending=False))

# Calculate interaction effects
interaction_effect = results_df['Total-order'] - results_df['First-order']
results_df['Interaction_Effect'] = interaction_effect

# Identify key parameters and interactions
top_params = results_df.sort_values('Total-order', ascending=False).head(3)['Parameter'].tolist()
top_interactions = results_df.sort_values('Interaction_Effect', ascending=False).head(3)['Parameter'].tolist()

print("\nBorehole Analysis Insights:")
print(f"Most influential parameters: {', '.join(top_params)}")
print(f"Parameters with strongest interactions: {', '.join(top_interactions)}")
```

### Chemical Reaction Model

This example illustrates sensitivity analysis for a chemical reaction kinetics model:

```python
import numpy as np
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis

# Define a chemical reaction model
def chemical_reaction_model(X):
    """
    Model of a chemical reaction with Arrhenius kinetics.
    
    Variables:
    - A: Pre-exponential factor (1/s)
    - Ea: Activation energy (J/mol)
    - T: Temperature (K)
    - Ca0: Initial concentration of reactant A (mol/L)
    - Cb0: Initial concentration of reactant B (mol/L)
    - t: Reaction time (s)
    - n: Reaction order for A
    - m: Reaction order for B
    """
    A = X[:, 0]
    Ea = X[:, 1]
    T = X[:, 2]
    Ca0 = X[:, 3]
    Cb0 = X[:, 4]
    t = X[:, 5]
    n = X[:, 6]
    m = X[:, 7]
    
    # Gas constant
    R = 8.314  # J/(mol·K)
    
    # Arrhenius equation for rate constant
    k = A * np.exp(-Ea / (R * T))
    
    # Simplified batch reactor model (analytical solution for equal orders)
    # For demonstration purposes, using a simplified conversion calculation
    if np.all(np.abs(n - m) < 1e-6):  # Check if n and m are approximately equal
        # For n=m=1 (second-order reaction)
        rate = k * t
        conversion = (Ca0 * Cb0 * rate) / (1 + Ca0 * rate)
    else:
        # For different orders, use a simple approximation
        conversion = 1 - np.exp(-k * t * (Ca0**(n-1)) * (Cb0**m))
    
    return conversion

# Define data_info
reaction_data = {
    'variables': [
        {
            'name': 'A',
            'vars_type': 'design_vars',
            'distribution': 'lognormal',
            'range_bounds': [1e7, 1e10],
            'cov': 0.1,
            'description': 'Pre-exponential factor (1/s)'
        },
        {
            'name': 'Ea',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [50000, 80000],
            'cov': 0.05,
            'description': 'Activation energy (J/mol)'
        },
        {
            'name': 'T',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [300, 500],
            'cov': 0.02,
            'description': 'Temperature (K)'
        },
        {
            'name': 'Ca0',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 2.0,
            'std': 0.1,
            'description': 'Initial concentration of A (mol/L)'
        },
        {
            'name': 'Cb0',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 1.5,
            'std': 0.1,
            'description': 'Initial concentration of B (mol/L)'
        },
        {
            'name': 't',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [100, 1000],
            'description': 'Reaction time (s)'
        },
        {
            'name': 'n',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [0.8, 1.2],
            'description': 'Reaction order for A'
        },
        {
            'name': 'm',
            'vars_type': 'design_vars',
            'distribution': 'uniform',
            'range_bounds': [0.8, 1.2],
            'description': 'Reaction order for B'
        }
    ]
}

# Save data_info
import json
with open("DATA_PREPARATION/reaction_data.json", 'w') as f:
    json.dump(reaction_data, f)

# Run analysis
results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/reaction_data.json",
    true_func=chemical_reaction_model,
    num_samples=2048,
    output_dir="RESULT_SA_REACTION",
    show_progress=True
)

# Print results
print("\nChemical Reaction Sensitivity Analysis Results:")
print("-" * 50)
print(results_df.sort_values('Total-order', ascending=False))

# Chemical engineering insights
top_params = results_df.sort_values('Total-order', ascending=False).head(3)['Parameter'].tolist()

print("\nChemical Process Insights:")
print(f"Most influential parameters: {', '.join(top_params)}")
print("\nRecommendations for Process Optimization:")
if 'T' in top_params:
    print("- Temperature control is critical for reaction performance")
if 'A' in top_params or 'Ea' in top_params:
    print("- Catalyst selection (affecting kinetic parameters) is important")
if 't' in top_params:
    print("- Reaction time should be carefully optimized")
if 'n' in top_params or 'm' in top_params:
    print("- The reaction mechanism should be thoroughly investigated")
```

### Circuit Design Problem

This example demonstrates sensitivity analysis for an electronic circuit design:

```python
import numpy as np
from PyEGRO.sensitivity.SAmcs import run_sensitivity_analysis

# Define a circuit model (RC low-pass filter)
def circuit_performance(X):
    """
    Calculate the cutoff frequency and performance metrics of an RC filter.
    
    Variables:
    - R1: Resistor 1 value (ohms)
    - R2: Resistor 2 value (ohms)
    - C1: Capacitor 1 value (farads)
    - C2: Capacitor 2 value (farads)
    - Vin: Input voltage (volts)
    - T: Temperature (°C)
    - f: Frequency of interest (Hz)
    """
    R1 = X[:, 0]
    R2 = X[:, 1]
    C1 = X[:, 2]
    C2 = X[:, 3]
    Vin = X[:, 4]
    T = X[:, 5]
    f = X[:, 6]
    
    # Temperature effect on resistors (simplified)
    alpha_R = 0.0004  # Temperature coefficient for resistors
    R1_actual = R1 * (1 + alpha_R * (T - 25))
    R2_actual = R2 * (1 + alpha_R * (T - 25))
    
    # Temperature effect on capacitors (simplified)
    alpha_C = -0.0003  # Temperature coefficient for capacitors
    C1_actual = C1 * (1 + alpha_C * (T - 25))
    C2_actual = C2 * (1 + alpha_C * (T - 25))
    
    # Parallel combination of R2 and C2
    Z_R2 = R2_actual
    Z_C2 = 1 / (2j * np.pi * f * C2_actual)
    Z_parallel = (Z_R2 * Z_C2) / (Z_R2 + Z_C2)
    
    # Series with R1 and C1
    Z_R1 = R1_actual
    Z_C1 = 1 / (2j * np.pi * f * C1_actual)
    Z_total = Z_R1 + Z_C1 + Z_parallel
    
    # Calculate transfer function magnitude
    H = Z_parallel / Z_total
    gain_dB = 20 * np.log10(np.abs(H))
    
    # For sensitivity analysis, use the absolute gain loss
    # (Negative because we want to minimize loss)
    return -gain_dB

# Define data_info
circuit_data = {
    'variables': [
        {
            'name': 'R1',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [1000, 10000],  # 1k to 10k ohms
            'cov': 0.05,  # 5% tolerance
            'description': 'Resistor 1 (ohms)'
        },
        {
            'name': 'R2',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [10000, 100000],  # 10k to 100k ohms
            'cov': 0.05,  # 5% tolerance
            'description': 'Resistor 2 (ohms)'
        },
        {
            'name': 'C1',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [1e-9, 1e-6],  # 1nF to 1uF
            'cov': 0.1,  # 10% tolerance
            'description': 'Capacitor 1 (farads)'
        },
        {
            'name': 'C2',
            'vars_type': 'design_vars',
            'distribution': 'normal',
            'range_bounds': [1e-9, 1e-6],  # 1nF to 1uF
            'cov': 0.1,  # 10% tolerance
            'description': 'Capacitor 2 (farads)'
        },
        {
            'name': 'Vin',
            'vars_type': 'env_vars',
            'distribution': 'normal',
            'mean': 5.0,
            'std': 0.25,  # 5% variation
            'description': 'Input voltage (volts)'
        },
        {
            'name': 'T',
            'vars_type': 'env_vars',
            'distribution': 'uniform',
            'low': 0,
            'high': 70,
            'description': 'Temperature (°C)'
        },
        {
            'name': 'f',
            'vars_type': 'env_vars',
            'distribution': 'lognormal',
            'mean': 1000,  # 1kHz
            'cov': 0.5,
            'description': 'Frequency (Hz)'
        }
    ]
}

# Save data_info
import json
with open("DATA_PREPARATION/circuit_data.json", 'w') as f:
    json.dump(circuit_data, f)

# Run analysis
results_df = run_sensitivity_analysis(
    data_info_path="DATA_PREPARATION/circuit_data.json",
    true_func=circuit_performance,
    num_samples=2048,
    output_dir="RESULT_SA_CIRCUIT",
    show_progress=True
)

# Print results
print("\nCircuit Design Sensitivity Analysis Results:")
print("-" * 50)
print(results_df.sort_values('Total-order', ascending=False))

# Electronics design insights
top_params = results_df.sort_values('Total-order', ascending=False).head(3)['Parameter'].tolist()

print("\nCircuit Design Insights:")
print(f"Most influential parameters: {', '.join(top_params)}")
print("\nDesign Recommendations:")
for param in top_params:
    if param in ['R1', 'R2']:
        print(f"- Use precision resistors for {param} with tighter tolerance")
    elif param in ['C1', 'C2']:
        print(f"- Use high-stability capacitors for {param} (e.g., film instead of ceramic)")
    elif param == 'T':
        print("- Implement temperature compensation or temperature control")
    elif param == 'f':
        print("- Design the circuit for a wider frequency bandwidth")
    elif param == 'Vin':
        print("- Add voltage regulation to stabilize input voltage")
```