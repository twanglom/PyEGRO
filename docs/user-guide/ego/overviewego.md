# Efficient Global Optimization (EGO) / Overview

**Applications for enhancing Global Model Accuracy**

Efficient Global Optimization (EGO) is widely applied to improve the accuracy of global models for predictive tasks across the entire input domain. By leveraging surrogate models, such as Gaussian Processes (GP), EGO provides high-fidelity approximations of complex systems, ensuring accurate predictions for areas that are underexplored or uncertain.


### **<span style='color: teal;'>Key Features</span>**

**Surrogate Modeling**:

   - Uses Gaussian Processes Regression (GPR) to approximate expensive-to-evaluate objective functions.


**Acquisition Functions**

   - Expected Improvement (EI)
   - \(\zeta\)-Expected Improvement (\(\zeta\)-EI)
   - Exploration Enhanced EI (E3I)
   - Expected Improvement for Global Fit (EIGF)
   - Distance-Enhanced Gradient (CRI3)

**Optimization Configuration**

   - Maximum iterations and stopping criteria
   - RMSE thresholds and patience settings
   - Relative improvement thresholds
   - Flexible device selection (CPU/GPU)

**Visualization Tools**

   - 1D and 2D optimization problem visualizations
   - Convergence plots
   - Uncertainty analysis and parameter tracking

**Progress Tracking**

   - Detailed console outputs
   - Metrics tracking for each iteration
   - Integration with rich progress bars for an interactive experience



### **<span style='color: teal;'>Acquisition Functions</span>**

Acquisition functions guide the optimization process. Below are the formulations:

**<span style='color: Coral;'>Expected Improvement (EI)</span>**

$$
EI(x) =
\begin{cases} 
\sigma(x)\phi(Z) + (y_{\text{min}} - \mu(x))\Phi(Z) & \text{if } \sigma(x) > 0 \\
0 & \text{if } \sigma(x) = 0 
\end{cases}
$$

$$
Z = \frac{y_{\text{min}} - \mu(x)}{\sigma(x)}
$$

where \(\phi(Z)\) is the Probability Density Function (PDF) and \(\Phi(Z)\) is the Cumulative Density Function (CDF).



**<span style='color: Coral;'>\(\zeta\)-Expected Improvement (\(\zeta\)-EI)</span>**

$$
\zeta-EI(x) =
\begin{cases} 
\sigma(x)\phi(Z) + (y_{\text{min}} - \mu(x) - \zeta)\Phi(Z) & \text{if } \sigma(x) > 0 \\
0 & \text{if } \sigma(x) = 0 
\end{cases}
$$

$$
Z = \frac{y_{\text{min}} - \mu(x) - \zeta}{\sigma(x)}
$$


**<span style='color: Coral;'>Exploration Enhanced EI (E3I)</span>** [ <a href="#reference-1" style="color:blue; text-decoration:underline;">1</a> ]


$$
\alpha^{E3I}(x) = \frac{1}{M}\mathbb{E}_x\left[I(x, g_m)\right] =
\begin{cases} 
\frac{\sigma(x)}{M}\sum_{m=1}^{M}\tau(Z) & \text{if } \sigma(x) > 0 \\
0 & \text{if } \sigma(x) = 0
\end{cases}
$$

$$
\tau(Z) = Z\Phi(Z) + \phi(Z), \quad Z = \frac{y_{\text{min}} - g_m(x)}{\sigma(x)}
$$

where \(g_m\) is the maximum value from \(M\) Thompson samples.


**<span style='color: Coral;'>Expected Improvement for Global Fit (EIGF)</span>** [ <a href="#reference-2" style="color:blue; text-decoration:underline;">2</a> ]


$$
EIGF(x) = (\hat{y}(x) - y(x_{\text{nearest}}))^2 + \sigma^2(x)
$$

where:
- \(\hat{y}(x)\): Predicted mean at point \(x\).
- \(y(x_{\text{nearest}})\): Observed output at the nearest sampled point to \(x\).
- \(\sigma^2(x)\): Variance of the prediction at \(x\).



**<span style='color: Coral;'>Distance-Enhanced Gradient (CRI3)</span>** [ <a href="#reference-3" style="color:blue; text-decoration:underline;">3</a> ]


$$
\text{Crit}(\xi) = (\left|\frac{\partial \hat{f}(\xi)}{\partial \xi}\right| \Delta(\xi) + D_f(\xi)) \hat{\sigma}(\xi) PDF(\xi).
$$

where:

- \(\Delta(\xi) = \min_{i=1,2,...,N}|\xi - \xi^{(i)}|\): Distance to the nearest sample.

- \(D_f(\xi) = |\hat{f}(\xi|\theta) - \hat{f}(\xi|2\theta)|\): Polynomial error estimate.

- \(\hat{\sigma}(\xi)PDF(\xi)\): Predictive variance multiplied by the PDF of the input space.



### **<span style='color: teal;'>Output and Files</span>**


Each EGO run produces:

1. **Optimization History:** Tracks iteration details, objective values, and model parameters.

2. **Trained Models:** Stores the final GP model and associated hyperparameters.

3. **Visualization Plots:** For tracking optimization progress and uncertainty.

---

**References**

<a id="reference-1"></a>1. Berk, J., Nguyen, V., Gupta, S., Rana, S., & Venkatesh, S. (2019). Exploration enhanced expected improvement for Bayesian optimization. In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2018, Dublin, Ireland, September 10–14, 2018, Proceedings, Part II 18* (pp. 621-637). Springer International Publishing.

<a id="reference-2"></a>2. Lam, C. Q. (2008). Sequential adaptive designs in computer experiments for response surface model fit (Doctoral dissertation, The Ohio State University).

<a id="reference-3"></a>3. Shimoyama, K., & Kawai, S. (2019). A kriging-based dynamic adaptive sampling method for uncertainty quantification. *Transactions of the Japan Society for Aeronautical and Space Sciences, 62*(3), 137-150.

---





### **<span style='color: teal;'>Algorithm Comparison</span>**
























