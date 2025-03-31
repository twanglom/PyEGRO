
# Design of Experiment / Overview


### **<span style=color:teal;>Sampling Methods</span>**

1. **Latin Hypercube Sampling (LHS):**
   Latin Hypercube Sampling divides each variable's range into $N$ equal intervals and ensures one sample is drawn from each interval without repetition. It maximizes uniformity in the sampling process:

    $$ x_i \sim U\left(\frac{k-1}{N}, \frac{k}{N}\right), \quad k = 1, \ldots, N $$

    where:
    - \( N \): Total number of samples
    - \( U(a, b) \): Uniform distribution between \( a \) and \( b \)

2. **Sobol Sequence:**
   Sobol sequences are quasi-random low-discrepancy sequences designed to achieve uniformity in the sampling space. The discrepancy \( D \) measures the uniformity, which is minimized in Sobol sampling:

    $$ D \leq \frac{\log(N)^d}{N} $$

    where:
    - \( N \): Total number of samples
    - \( d \): Dimensionality of the problem

3. **Random Sampling:**
   Random sampling draws samples \( x \) uniformly from the range \( [a_i, b_i] \):

    $$ x_i \sim U(a_i, b_i) $$

**Sampling Criteria**

Sampling criteria are additional constraints or goals applied during sampling, such as:
- **Maximin Criterion:** Maximizes the minimum distance between samples.
- **Centering Criterion:** Ensures that samples are centered within each interval.


### **<span style=color:teal;>Objective Function Evaluation</span>**


Once the samples \( x \) are generated, they are passed to an **objective function** \( f(x) \) to evaluate the desired output:

$$ \text{Result} = f(x) $$

Where:
- \( x \): Sampled input vector
- \( f(x) \): User-defined objective function

**Performance Metrics for Sampling**

To evaluate the quality of the sampling, metrics such as **space-filling**, **discrepancy**, and **variance** are often used.

