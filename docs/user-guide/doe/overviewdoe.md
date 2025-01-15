# Mathematical Formulation

**Sampling Methods**

1. **Latin Hypercube Sampling (LHS):** Divides each variable's range into equal intervals and ensures one sample is drawn from each interval. The process ensures balanced coverage by sampling from:

    $$ x_i \sim U\left(\frac{k-1}{N}, \frac{k}{N}\right), \quad k = 1, \ldots, N $$

    where $N$ is the total number of samples.

2. **Sobol Sequence:** Generates low-discrepancy sequences with uniformity, minimizing discrepancy $D$:

    $$ D \leq \frac{\log(N)^d}{N} $$

    where $d$ is the dimensionality of the problem.

3. **Random Sampling:** Random sampling draws samples $x$ uniformly from the range $[a_i, b_i]$:

    $$ x_i \sim U(a_i, b_i) $$

**Objective Function Evaluation**

Samples $x$ are evaluated using an objective function $f(x)$:

$$ \text{Result} = f(x) $$

This provides flexibility for custom functions during design optimization.