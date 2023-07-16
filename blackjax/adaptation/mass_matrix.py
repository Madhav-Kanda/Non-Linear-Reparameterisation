# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Algorithms to adapt the mass matrix used by algorithms in the Hamiltonian
Monte Carlo family to the current geometry.

The Stan Manual :cite:p:`stan_hmc_param` is a very good reference on automatic tuning of
parameters used in Hamiltonian Monte Carlo.

"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jax.nn import sigmoid
import numpyro.distributions as dist

from blackjax.types import Array, ArrayLike

__all__ = [
    "WelfordAlgorithmState",
    "MassMatrixAdaptationState",
    "mass_matrix_adaptation",
    "welford_algorithm",
]


class WelfordAlgorithmState(NamedTuple):
    """State carried through the Welford algorithm.

    mean
        The running sample mean.
    m2
        The running value of the sum of difference of squares. See documentation
        of the `welford_algorithm` function for an explanation.
    sample_size
        The number of successive states the previous values have been computed on;
        also the current number of iterations of the algorithm.

    """

    mean: Array
    m2: Array
    sample_size: int


class MassMatrixAdaptationState(NamedTuple):
    """State carried through the mass matrix adaptation.

    inverse_mass_matrix
        The curent value of the inverse mass matrix.
    wc_state
        The current state of the Welford Algorithm.

    """

    inverse_mass_matrix: Array
    wc_state: WelfordAlgorithmState


def mass_matrix_adaptation(
    is_diagonal_matrix: bool = True,
) -> tuple[Callable, Callable, Callable]:
    """Adapts the values in the mass matrix by computing the covariance
    between parameters.

    Parameters
    ----------
    is_diagonal_matrix
        When True the algorithm adapts and returns a diagonal mass matrix
        (default), otherwise adaps and returns a dense mass matrix.

    Returns
    -------
    init
        A function that initializes the step of the mass matrix adaptation.
    update
        A function that updates the state of the mass matrix.
    final
        A function that computes the inverse mass matrix based on the current
        state.

    """
    wc_init, wc_update, wc_final = welford_algorithm(is_diagonal_matrix)

    def init(n_dims: int) -> MassMatrixAdaptationState:
        """Initialize the matrix adaptation.

        Parameters
        ----------
        ndims
            The number of dimensions of the mass matrix, which corresponds to
            the number of dimensions of the chain position.

        """
        if is_diagonal_matrix:
            inverse_mass_matrix = jnp.ones(n_dims)
        else:
            inverse_mass_matrix = jnp.identity(n_dims)

        wc_state = wc_init(n_dims)

        return MassMatrixAdaptationState(inverse_mass_matrix, wc_state)

    def update(
        mm_state: MassMatrixAdaptationState, position: ArrayLike
    ) -> MassMatrixAdaptationState:
        """Update the algorithm's state.

        Parameters
        ----------
        state:
            The current state of the mass matrix adapation.
        position:
            The current position of the chain.

        """
        inverse_mass_matrix, wc_state = mm_state
        position, _ = jax.flatten_util.ravel_pytree(position)
        wc_state = wc_update(wc_state, position)
        return MassMatrixAdaptationState(inverse_mass_matrix, wc_state)

    def log_jacobian(samples, c, samples_keys):
        log_sigma = jnp.array(samples[samples_keys[1]])
        logJ = jnp.sum(log_sigma * (1 - c)[:, jnp.newaxis])
        logJ /= len(log_sigma)
        return logJ


    def kl_value_constrained(centeredness,samples, samples_keys,prev_c):
        reparam_sample, mvn, mu_theta, std_theta = reparameterize_samples_dist(samples, sigmoid(centeredness), samples_keys, prev_c)
        jacobian_log = log_jacobian(samples, sigmoid(centeredness),samples_keys)
        kl = -mvn.log_prob(reparam_sample.T).mean() + jacobian_log    
        return kl  

    def best_centered_cov(samples,c,samples_keys,prev_c):
        param_samples = jnp.array(samples[samples_keys[2]]).T
        param_mean = jnp.array(samples[samples_keys[0]])
        param_std = jnp.exp(jnp.array(samples[samples_keys[1]])) # Adding this extra transformation to get it back to constrained space!!
        # param_std = jnp.ones(param_mean.shape)*2

        new_param_samples = jnp.expand_dims(c, axis=1) * jnp.expand_dims(param_mean, axis = 0)  + (param_samples - (jnp.expand_dims(prev_c,axis=1)*jnp.expand_dims(param_mean, axis = 0))) * jnp.power(param_std, (c - prev_c)[:, jnp.newaxis])
        std_samples = jnp.std(new_param_samples, axis=1, ddof = 1)
        
        std_mean = jnp.std(param_mean, ddof = 1)
        std_logsd = jnp.std(jnp.log(param_std),ddof=1)
        
        std = jnp.zeros(2+ std_samples.size)
        for i in range(2+ std_samples.size):
            if i < std_mean.size:
                std = std.at[i].set(std_mean)
            elif i < std_mean.size+ std_logsd.size:
                std = std.at[i].set(std_logsd)
            else:
                std = std.at[i].set(std_samples[i-2])
        return (std**2)

    def reparameterize_samples_dist(samples, c, samples_keys,prev_c):
        # print(samples_keys[0],samples_keys[1],samples_keys[2])
        param_samples = jnp.array(samples[samples_keys[2]]).T
        param_mean = jnp.array(samples[samples_keys[0]])
        param_std = jnp.exp(jnp.array(samples[samples_keys[1]]))

        new_param_samples = jnp.expand_dims(c, axis=1) * jnp.expand_dims(param_mean, axis = 0) + (param_samples - (jnp.expand_dims(prev_c,axis=1)*jnp.expand_dims(param_mean, axis = 0))) * jnp.power(param_std, (c - prev_c)[:, jnp.newaxis])

        theta_mu = jnp.mean(new_param_samples, axis=1)
        theta_std = jnp.std(new_param_samples, axis=1, ddof = 1)

        covariance_matrix = jnp.diag(theta_std ** 2)
        mvn = dist.MultivariateNormal(loc=jnp.array(theta_mu), covariance_matrix=covariance_matrix)

        return new_param_samples, mvn, theta_mu, theta_std

    def final(mm_state: MassMatrixAdaptationState, samples, centeredness, prev_c) -> MassMatrixAdaptationState:
        """Final iteration of the mass matrix adaptation.

        In this step we compute the mass matrix from the covariance matrix computed
        by the Welford algorithm, and re-initialize the later.

        """
        _, wc_state = mm_state
        covariance, count, mean = wc_final(wc_state)

        samples_keys = list(samples.keys())
        if centeredness is None:
            centeredness = jnp.ones(samples[samples_keys[2]].shape[1])*0.5
            # centeredness = jnp.ones(1)*0.5

        if prev_c is None:
            prev_c = jnp.ones(centeredness.shape)

        print(centeredness.shape)
        kl_value_ = lambda x: kl_value_constrained(x, samples,samples_keys,prev_c)
        res = minimize(kl_value_, centeredness, method='BFGS')
        centeredness = sigmoid(res.x)
        # centeredness = res.x
        covariance = best_centered_cov(samples,centeredness,samples_keys,prev_c)
        # print("new covariance: ", covariance)

        # Regularize the covariance matrix, see Stan
        scaled_covariance = (count / (count + 5)) * covariance
        shrinkage = 1e-3 * (5 / (count + 5))
        if is_diagonal_matrix:
            inverse_mass_matrix = scaled_covariance + shrinkage
        else:
            inverse_mass_matrix = scaled_covariance + shrinkage * jnp.identity(
                mean.shape[0]
            )

        ndims = jnp.shape(inverse_mass_matrix)[-1]
        new_mm_state = MassMatrixAdaptationState(inverse_mass_matrix, wc_init(ndims))

        return new_mm_state, centeredness, prev_c

    return init, update, final


def welford_algorithm(is_diagonal_matrix: bool) -> tuple[Callable, Callable, Callable]:
    r"""Welford's online estimator of covariance.

    It is possible to compute the variance of a population of values in an
    on-line fashion to avoid storing intermediate results. The naive recurrence
    relations between the sample mean and variance at a step and the next are
    however not numerically stable.

    Welford's algorithm uses the sum of square of differences
    :math:`M_{2,n} = \sum_{i=1}^n \left(x_i-\overline{x_n}\right)^2`
    for updating where :math:`x_n` is the current mean and the following
    recurrence relationships

    Parameters
    ----------
    is_diagonal_matrix
        When True the algorithm adapts and returns a diagonal mass matrix
        (default), otherwise adaps and returns a dense mass matrix.

    Note
    ----
    It might seem pedantic to separate the Welford algorithm from mass adaptation,
    but this covariance estimator is used in other parts of the library.

    """

    def init(n_dims: int) -> WelfordAlgorithmState:
        """Initialize the covariance estimation.

        When the matrix is diagonal it is sufficient to work with an array that contains
        the diagonal value. Otherwise we need to work with the matrix in full.

        Parameters
        ----------
        n_dims: int
            The number of dimensions of the problem, which corresponds to the size
            of the corresponding square mass matrix.

        """
        sample_size = 0
        mean = jnp.zeros((n_dims,))
        if is_diagonal_matrix:
            m2 = jnp.zeros((n_dims,))
        else:
            m2 = jnp.zeros((n_dims, n_dims))
        return WelfordAlgorithmState(mean, m2, sample_size)

    def update(
        wa_state: WelfordAlgorithmState, value: ArrayLike
    ) -> WelfordAlgorithmState:
        """Update the M2 matrix using the new value.

        Parameters
        ----------
        wa_state:
            The current state of the Welford Algorithm
        value: Array, shape (1,)
            The new sample (typically position of the chain) used to update m2

        """
        mean, m2, sample_size = wa_state
        sample_size = sample_size + 1

        delta = value - mean
        mean = mean + delta / sample_size
        updated_delta = value - mean
        if is_diagonal_matrix:
            new_m2 = m2 + delta * updated_delta
        else:
            new_m2 = m2 + jnp.outer(updated_delta, delta)

        return WelfordAlgorithmState(mean, new_m2, sample_size)

    def final(
        wa_state: WelfordAlgorithmState,
    ) -> tuple[Array, int, Array]:
        mean, m2, sample_size = wa_state
        covariance = m2 / (sample_size - 1)
        return covariance, sample_size, mean

    return init, update, final