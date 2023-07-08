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
"""Implementation of the Stan warmup for the HMC family of sampling algorithms."""
from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter


import blackjax.mcmc as mcmc
from numpyro.infer.util import initialize_model
from numpyro.infer.reparam import LocScaleReparam
from numpyro.handlers import reparam
from blackjax.adaptation.base import AdaptationInfo, AdaptationResults
from blackjax.adaptation.mass_matrix import (
    MassMatrixAdaptationState,
    mass_matrix_adaptation,
)
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.base import AdaptationAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size

__all__ = ["WindowAdaptationState", "base", "build_schedule", "window_adaptation"]


class WindowAdaptationState(NamedTuple):
    ss_state: DualAveragingAdaptationState  # step size
    imm_state: MassMatrixAdaptationState  # inverse mass matrix
    step_size: float
    inverse_mass_matrix: Array


def base(
    is_mass_matrix_diagonal: bool,
    target_acceptance_rate: float = 0.80,
) -> tuple[Callable, Callable, Callable]:
    """Warmup scheme for sampling procedures based on euclidean manifold HMC.
    The schedule and algorithms used match Stan's :cite:p:`stan_hmc_param` as closely as possible.

    Unlike several other libraries, we separate the warmup and sampling phases
    explicitly. This ensure a better modularity; a change in the warmup does
    not affect the sampling. It also allows users to run their own warmup
    should they want to.
    We also decouple generating a new sample with the mcmc algorithm and
    updating the values of the parameters.

    Stan's warmup consists in the three following phases:

    1. A fast adaptation window where only the step size is adapted using
    Nesterov's dual averaging scheme to match a target acceptance rate.
    2. A succession of slow adapation windows (where the size of a window is
    double that of the previous window) where both the mass matrix and the step
    size are adapted. The mass matrix is recomputed at the end of each window;
    the step size is re-initialized to a "reasonable" value.
    3. A last fast adaptation window where only the step size is adapted.

    Schematically:

    +---------+---+------+------------+------------------------+------+
    |  fast   | s | slow |   slow     |        slow            | fast |
    +---------+---+------+------------+------------------------+------+
    |1        |2  |3     |3           |3                       |3     |
    +---------+---+------+------------+------------------------+------+

    Step (1) consists in find a "reasonable" first step size that is used to
    initialize the dual averaging scheme. In (2) we initialize the mass matrix
    to the matrix. In (3) we compute the mass matrix to use in the kernel and
    re-initialize the mass matrix adaptation. The step size is still adapated
    in slow adaptation windows, and is not re-initialized between windows.

    Parameters
    ----------
    is_mass_matrix_diagonal
        Create and adapt a diagonal mass matrix if True, a dense matrix
        otherwise.
    target_acceptance_rate:
        The target acceptance rate for the step size adaptation.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns the step size and mass matrix given a warmup
        state.

    """
    mm_init, mm_update, mm_final = mass_matrix_adaptation(is_mass_matrix_diagonal)
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)


    def init(
        position: ArrayLikeTree, initial_step_size: float
    ) -> WindowAdaptationState:
        """Initialze the adaptation state and parameter values.

        Unlike the original Stan window adaptation we do not use the
        `find_reasonable_step_size` algorithm which we found to be unnecessary.
        We may reconsider this choice in the future.

        """
        num_dimensions = pytree_size(position)
        imm_state = mm_init(num_dimensions)

        ss_state = da_init(initial_step_size)

        return WindowAdaptationState(
            ss_state,
            imm_state,
            initial_step_size,
            imm_state.inverse_mass_matrix
        )

    def fast_update(
        position: ArrayLikeTree,
        acceptance_rate: float,
        warmup_state: WindowAdaptationState,
        running_samples
    ) -> WindowAdaptationState:
        """Update the adaptation state when in a "fast" window.

        Only the step size is adapted in fast windows. "Fast" refers to the fact
        that the optimization algorithms are relatively fast to converge
        compared to the covariance estimation with Welford's algorithm

        """
        del position

        new_ss_state = da_update(warmup_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return WindowAdaptationState(
            new_ss_state,
            warmup_state.imm_state,
            new_step_size,
            warmup_state.inverse_mass_matrix
        ),running_samples


    def append_element(running_samples,position):
        if bool(running_samples) == False:
            for key in position.keys():
                running_samples[key] = []
                
        for key in position.keys():
            value = position[key]
            running_samples[key].append(value.tolist())
        return running_samples

    def slow_update(
        position: ArrayLikeTree,
        acceptance_rate: float,
        warmup_state: WindowAdaptationState,
        running_samples
    ) -> WindowAdaptationState:
        """Update the adaptation state when in a "slow" window.
    

        Both the mass matrix adaptation *state* and the step size state are
        adapted in slow windows. The value of the step size is updated as well,
        but the new value of the inverse mass matrix is only computed at the end
        of the slow window. "Slow" refers to the fact that we need many samples
        to get a reliable estimation of the covariance matrix used to update the
        value of the mass matrix.

        """

        new_imm_state = mm_update(warmup_state.imm_state, position)
        new_ss_state = da_update(warmup_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        running_samples = append_element(running_samples,position)
        return WindowAdaptationState(
            new_ss_state, new_imm_state, new_step_size, warmup_state.inverse_mass_matrix
        ), running_samples

    def slow_final(warmup_state: WindowAdaptationState, running_samples,centeredness, varname, prev_c):
        """Update the parameters at the end of a slow adaptation window.

        We compute the value of the mass matrix and reset the mass matrix
        adapation's internal state since middle windows are "memoryless".

        """
        new_imm_state,centeredness,varname,prev_c = mm_final(warmup_state.imm_state, running_samples,centeredness, varname, prev_c)
        new_ss_state = da_init(da_final(warmup_state.ss_state))
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        running_samples = {}

        return WindowAdaptationState(
            new_ss_state,
            new_imm_state,
            new_step_size,
            new_imm_state.inverse_mass_matrix,
        ), running_samples, centeredness, varname, prev_c

    def update(
        adaptation_state: WindowAdaptationState,
        adaptation_stage: tuple,
        position: ArrayLikeTree,
        acceptance_rate: float,
        running_samples, centeredness, varname, prev_c
    ) -> WindowAdaptationState:
        """Update the adaptation state and parameter values.

        Parameters
        ----------
        adaptation_state
            Current adptation state.
        adaptation_stage
            The current stage of the warmup: whether this is a slow window,
            a fast window and if we are at the last step of a slow window.
        position
            Current value of the model parameters.
        acceptance_rate
            Value of the acceptance rate for the last mcmc step.

        Returns
        -------
        The updated adaptation state.

        """
        stage, is_middle_window_end = adaptation_stage
        # warmup_state, running_samples,i = jax.lax.switch(
        #     stage,
        #     (fast_update, slow_update),
        #     position,
        #     acceptance_rate,
        #     adaptation_state,
        #     running_samples,i
        # )

        if stage == 0:
            warmup_state, running_samples = fast_update(
                position,
                acceptance_rate,
                adaptation_state,
                running_samples
            )
        else:
            warmup_state, running_samples = slow_update(
                position,
                acceptance_rate,
                adaptation_state,
                running_samples
            )

        ## Need to update the is_middle_window_end condition !!
        # warmup_state, running_samples,i = jax.lax.cond(
        #     is_middle_window_end,
        #     slow_final,
        #     lambda x, y,i: (x, y,i),  # Updated lambda function to accept two arguments
        #     warmup_state, running_samples,i
        # )
        if is_middle_window_end:
            warmup_state, running_samples,centeredness,varname, prev_c = slow_final(warmup_state, running_samples,centeredness, varname, prev_c)

        return warmup_state, running_samples, centeredness, varname, prev_c

    def final(warmup_state: WindowAdaptationState) -> tuple[float, Array]:
        """Return the final values for the step size and mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.imm_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final


def window_adaptation(
    algorithm: Union[mcmc.hmc.hmc, mcmc.nuts.nuts],
    model,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    progress_bar: bool = False,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the value of the inverse mass matrix and step size parameters of
    algorithms in the HMC fmaily.

    Algorithms in the HMC family on a euclidean manifold depend on the value of
    at least two parameters: the step size, related to the trajectory
    integrator, and the mass matrix, linked to the euclidean metric.

    Good tuning is very important, especially for algorithms like NUTS which can
    be extremely inefficient with the wrong parameter values. This function
    provides a general-purpose algorithm to tune the values of these parameters.
    Originally based on Stan's window adaptation, the algorithm has evolved to
    improve performance and quality.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to
        sample.
    is_mass_matrix_diagonal
        Whether we should adapt a diagonal mass matrix.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    progress_bar
        Whether we should display a progress bar.
    **extra_parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that runs the adaptation and returns an `AdaptationResult` object.

    """
    def logdensity_create(model, centeredness = None, varname = None):
        if centeredness is not None:
            model = reparam(model, config={varname: LocScaleReparam(centered= centeredness)})
        init_params, potential_fn_gen, *_ = initialize_model(jax.random.PRNGKey(0),model,dynamic_args=True)
        logdensity_fn = lambda position: -potential_fn_gen()(position)
        initial_position = init_params.z
        return jax.jit(logdensity_fn), initial_position

    mcmc_kernel = algorithm.build_kernel()
    adapt_init, adapt_step, adapt_final = base(
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )

    def one_step(carry, xs, centeredness,logdensity_f, prev_c,varname):
        _, rng_key, adaptation_stage = xs
        (state, adaptation_state),running_samples = carry

        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_f,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
      
        new_adaptation_state,running_samples, centeredness,varname, prev_c = adapt_step(
            adaptation_state,
            adaptation_stage,
            new_state.position,
            info.acceptance_rate,
            running_samples, centeredness, varname, prev_c
        )
        
        # Unconmment the following:
        if adaptation_stage.tolist() == [1,1]:
            logdensity_f,position = logdensity_create(model,centeredness,varname)
            # print("New: ",logdensity_f)
            new_state = algorithm.init(position, logdensity_f)
            new_adaptation_state = adapt_init(position, initial_step_size)
            
        return (
            ((new_state, new_adaptation_state), running_samples),
            AdaptationInfo(new_state, info, new_adaptation_state),centeredness, logdensity_f, prev_c, varname
        )

    def run(rng_key: PRNGKey, num_steps: int = 1000, **kwargs):
        logdensity_fn, initial_position = logdensity_create(model)
        position = initial_position
        init_state = algorithm.init(position, logdensity_fn)
        if(kwargs.get("initial_step_size") is None):
            init_adaptation_state = adapt_init(position, initial_step_size)
        else:
            init_adaptation_state = adapt_init(position, kwargs.get("initial_step_size"))


        # if progress_bar:
        #     one_step_ = jax.jit(progress_bar_scan(num_steps)(one_step))
        # else:
        #     one_step_ = jax.jit(one_step)

        keys = jax.random.split(rng_key, num_steps)
        info = None
        if len(kwargs) > 1:
            schedule = build_schedule(num_steps, kwargs.get("initial_buffer_size"), kwargs.get("final_buffer_size"), kwargs.get("first_window_size"))
        else:
            schedule = build_schedule(num_steps)

        running_samples = {}

        ### Original ###
        # (last_state,running_samples,i), info = jax.lax.scan(
        #     one_step_,
        #     ((init_state, init_adaptation_state),running_samples,0),
        #     (jnp.arange(num_steps), keys, schedule)
        # )
        centeredness = None
        prev_c = None
        varname = 'theta'

        for i in range(num_steps):
            (last_state,running_samples), info, centeredness,logdensity_, prev_c,varname = one_step(((init_state, init_adaptation_state),running_samples),(i, keys[i], schedule[i]), centeredness, logdensity_fn, prev_c,varname)
            logdensity_fn = logdensity_
            init_state,init_adaptation_state = last_state

        last_chain_state, last_warmup_state, *_ = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        # kernel = mcmc.nuts(logdensity_fn, parameters).step

        return (
            AdaptationResults(
                last_chain_state,
                parameters,
            ),
            info, logdensity_fn
        )

    return AdaptationAlgorithm(run)


def build_schedule(
    num_steps: int,
    initial_buffer_size: int = 75,
    final_buffer_size: int = 50,
    first_window_size: int = 25,
) -> list[tuple[int, bool]]:
    """Return the schedule for Stan's warmup.

    The schedule below is intended to be as close as possible to Stan's :cite:p:`stan_hmc_param`.
    The warmup period is split into three stages:

    1. An initial fast interval to reach the typical set. Only the step size is
    adapted in this window.
    2. "Slow" parameters that require global information (typically covariance)
    are estimated in a series of expanding intervals with no memory; the step
    size is re-initialized at the end of each window. Each window is twice the
    size of the preceding window.
    3. A final fast interval during which the step size is adapted using the
    computed mass matrix.

    Schematically:

    ```
    +---------+---+------+------------+------------------------+------+
    |  fast   | s | slow |   slow     |        slow            | fast |
    +---------+---+------+------------+------------------------+------+
    ```

    The distinction slow/fast comes from the speed at which the algorithms
    converge to a stable value; in the common case, estimation of covariance
    requires more steps than dual averaging to give an accurate value. See :cite:p:`stan_hmc_param`
    for a more detailed explanation.

    Fast intervals are given the label 0 and slow intervals the label 1.

    Parameters
    ----------
    num_steps: int
        The number of warmup steps to perform.
    initial_buffer: int
        The width of the initial fast adaptation interval.
    first_window_size: int
        The width of the first slow adaptation interval.
    final_buffer_size: int
        The width of the final fast adaptation interval.

    Returns
    -------
    A list of tuples (window_label, is_middle_window_end).

    """
    schedule = []

    # Give up on mass matrix adaptation when the number of warmup steps is too small.
    if num_steps < 20:
        schedule += [(0, False)] * num_steps
    else:
        # When the number of warmup steps is smaller that the sum of the provided (or default)
        # window sizes we need to resize the different windows.
        if initial_buffer_size + first_window_size + final_buffer_size > num_steps:
            initial_buffer_size = int(0.15 * num_steps)
            final_buffer_size = int(0.1 * num_steps)
            first_window_size = num_steps - initial_buffer_size - final_buffer_size

        # First stage: adaptation of fast parameters
        schedule += [(0, False)] * (initial_buffer_size)
        # schedule.append((0, False))

        # Second stage: adaptation of slow parameters in successive windows
        # doubling in size.
        final_buffer_start = num_steps - final_buffer_size

        next_window_size = first_window_size
        next_window_start = initial_buffer_size
        while next_window_start < final_buffer_start:
            current_start, current_size = next_window_start, next_window_size
            if 3 * current_size <= final_buffer_start - current_start:
                next_window_size = 2 * current_size
            else:
                current_size = final_buffer_start - current_start
            next_window_start = current_start + current_size
            schedule += [(1, False)] * (next_window_start - 1 - current_start)
            schedule.append((1, True))

        # Last stage: adaptation of fast parameters
        schedule += [(0, False)] * (num_steps - final_buffer_start)
        # schedule.append((0, False))

    schedule = jnp.array(schedule)

    return schedule