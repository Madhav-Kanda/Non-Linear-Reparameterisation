# Windowed Non Linear Reparameterization
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)


This repository contains the code for Windowed Non Linear Reparameterization built on top of Blacjax. This work was done as a part of summer internship at Aalto University under the guidance of Dr. Nikolas Siccha and Prof. Aki Vehtari.

### Why:

At times MCMC algorithms have trouble sampling from distributions. One such example is Neal’s funnel in which due to strong non-linear dependence between latent variables. Non-centering the model removes this dependence, converting the funnel into a spherical Gaussian distribution.

### Centeredness vs Non-centeredness:
- The best parameterization for a given model may lie somewhere between centered and non centered representation.
- Existing solutions:
  
  - Variationally Inferred Parameterization[[1]](https://arxiv.org/pdf/1906.03028.pdf)
  
  - NeuTra-lizing Bad Geometry in HMC[[2]](https://arxiv.org/pdf/1903.03704.pdf)
  
### Problems with existing solutions
- Requires separate pre-processing steps apart from the regular warmup and sampling steps which increases the computation cost.
- Need to tune the hyperparameters for the existing solutions to get good results

### Proposed Solution
- Finding the optimal centeredness during the succesive windows of warmup.
- Loss function for finding centeredness should be such that it takes the parameterized distribution as close as possible to Normal distribution.

### Warmup Phase
- Used for adaptation of inverse mass matrix ($M^{-1}$) and time step size ($\Delta t$).
- Consist of three stages:

  <img width="536" alt="image" src="https://github.com/Madhav-Kanda/Non-Linear-Reparameterisation/assets/76394914/a18877c5-c1fe-45f7-b4da-283c9550594b">

 - Initial buffer (I): Time step adaptation ($\Delta t$)
 - Window buffer (II): Both Time step ($\Delta t$) & Inverse mass matrix adaptation ($M^{-1}$)
 - Term buffer (III): Final Time step adaptation ($\Delta t$)

### Modified Warmup Phase
- Used for adaptation of inverse mass matrix ($M^{-1}$), time step size ($\Delta t$) and centeredness ($c$).
- Initial buffer and Term buffer remains the same.
- Using the samples obtained after each window buffer, optimize the centeredness ($c$) so as to reduce the distance between the present reparameterized distribution and an independent normal distribution.
- For each succesive window, reparameterize the model based on the optimal centeredness obtained and repeat the step for finding optimal centeredness.

### Implementation

- Modified BlackJax sampler code to incorporate the proposed algorithm.
- Inference time up from 2 seconds to 20 seconds.
- To evaluate the results used a model whose true centeredness already known:
    - $\mu$ ~ $N(0,1)$
    - $\log \sigma$ ~ $N(0,1)$
    - $x$ ~ $N((1-c_{i}) \mu, \sigma^{(1-c_{i})})$
 
### Results
- The Image shows the comparison of the centeredness achieved by our method v/s Variationally Inferred Parameterisation (VIP). It is evident that our method converged to the true centeredness of the model.
  <img width="541" alt="image" src="https://github.com/Madhav-Kanda/Non-Linear-Reparameterisation/assets/76394914/04b12af6-4255-47ae-84e9-d30c589ea19e">
- The Effective Sample Size per gradient evaluation for our method turns out to be better than VIP.
  | Parameter  | ESS/∇ (vip)  | ESS/∇ (our)  |
  |------------|--------------|--------------|
  | μ          | 9.27e-4      | 5.54e-4      |
  | τ          | 4.22e-5      | **5.07e-4**  |
  | θ          | 9.73e-4      | **1.66e-3**  |


