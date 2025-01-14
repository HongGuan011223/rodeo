r"""
Stochastic block solver for ODE initial value problems.

The ODE-IVP to be solved is defined as

.. math:: W X_t = f(X_t, t, \theta)

on the time interval :math:`t \in [a, b]` with initial condition :math:`X_a = x_0`.  

The stochastic solver proceeds via Kalman filtering and smoothing of "interrogations" of the ODE model as described in Chkrebtii et al 2016, Schober et al 2019.  In the context of the underlying Kalman filterer/smoother, the Gaussian state-space model is

.. math::

    X_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

    z_n = W_n X_n - f(X_n, t, \theta) + V_n^{1/2} \eta_n.

We assume that :math:`c_n = c, Q_n = Q, R_n = R`, and :math:`W_n = W` for all :math:`n`.

This module optimizes the calculations when :math:`Q`, :math:`R`, and :math:`W`, are block diagonal matrices of conformable and "stackable" sizes.  That is, recall that the dimension of these matrices are `n_state x n_state`, `n_state x n_state`, and `n_meas x n_state`, respectively.  Then suppose that :math:`Q` and :math:`R` consist of `n_block` blocks of size `n_bstate x n_bstate`, where `n_bstate = n_state/n_block`, and :math:`W` consists of `n_block` blocks of size `n_bmeas x n_bstate`, where `n_bmeas = n_meas/n_block`.  Then :math:`Q`, :math:`R`, :math:`W` can be stored as 3D arrays of size `n_block x n_bstate x n_bstate` and `n_block x n_bmeas x n_bstate`.  It is under this paradigm that the `ode` module operates.

"""

import jax
import jax.numpy as jnp
from rodeo.kalmantv import *


def interrogate_rodeo(key, fun, W, t, theta,
                      mean_state_pred, var_state_pred):
    r"""
    Rodeo interrogation method.

    Args:
        key (PRNGKey): Jax PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = f(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior.
        t (float): Time point.
        theta (ndarray(n_theta)): ODE parameter.
        mean_state_pred (ndarray(n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1]; denoted by :math:`\mu_{t|t-1}`.
        var_state_pred (ndarray(n_block, n_bstate, n_bstate)): Covariance of estimate for state at time t given observations from times [a...t-1]; denoted by :math:`\Sigma_{t|t-1}`.

    Returns:
        (tuple):
        - **trans_meas** (ndarray(n_block, n_bmeas, n_bstate)): Interrogation transition matrix.
        - **mean_meas** (ndarray(n_block, n_bmeas)): Interrogation offset.
        - **var_meas** (ndarray(n_block, n_bmeas, n_bmeas)): Interrogation variance.

    """
    n_block = mean_state_pred.shape[0]
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        W, var_state_pred
    )
    mean_meas = -fun(mean_state_pred, t, theta)
    return W, mean_meas, var_meas


def interrogate_chkrebtii(key, fun, W, t, theta,
                          mean_state_pred, var_state_pred):
    r"""
    Interrogate method of Chkrebtii et al (2016); DOI: 10.1214/16-BA1017.

    Same arguments and returns as :func:`~ode.interrogate_rodeo`.

    """
    n_block, n_bstate = mean_state_pred.shape
    key, *subkeys = jax.random.split(key, num=n_block+1)
    subkeys = jnp.array(subkeys)
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        W, var_state_pred
    )
    x_state = jax.vmap(lambda b:
                       jax.random.multivariate_normal(
                           subkeys[b],
                           mean_state_pred[b],
                           var_state_pred[b]
                       ))(jnp.arange(n_block))
    mean_meas = -fun(x_state, t, theta)
    return W, mean_meas, var_meas

def interrogate_schober(key, fun, W, t, theta,
                        mean_state_pred, var_state_pred):
    r"""
    Interrogate method of Schober et al (2019); DOI: https://doi.org/10.1007/s11222-017-9798-7.

    Same arguments and returns as :func:`~ode.interrogate_rodeo`.

    """
    n_block, n_bmeas, _ = W.shape
    var_meas = jnp.zeros((n_block, n_bmeas, n_bmeas))
    mean_meas = -fun(mean_state_pred, t, theta)
    return W, mean_meas, var_meas

def interrogate_tronarp(key, fun, W, t, theta,
                        mean_state_pred, var_state_pred):
    r"""
    First order interrogate method of Tronarp et al (2019); DOI: https://doi.org/10.1007/s11222-019-09900-1.
    Same arguments and returns as :func:`~ode.interrogate_rodeo`.

    """
    n_block, n_bmeas, n_bstate = W.shape
    p = int(n_bstate/n_bmeas)
    mean_meas = -fun(mean_state_pred, t, theta)
    jac = jax.jacfwd(fun)(mean_state_pred, t, theta)
    # need to get the diagonal of jac
    jac = jax.vmap(lambda b:
                   jac[b, :, b])(jnp.arange(n_block))
    trans_meas = W - jac
    # var_meas = jax.vmap(lambda wm, vsp:
    #                     jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
    #     trans_meas, var_state_pred
    # )
    var_meas = jnp.zeros((n_block, n_bmeas, n_bmeas))
    return trans_meas, mean_meas, var_meas


def _solve_filter(key, fun,  W,  x0, theta,
                  tmin, tmax, n_steps,
                  trans_state, mean_state, var_state,
                  interrogate):
    r"""
    Forward pass of the ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # lax.scan setup
    # scan function
    def scan_fun(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"])
        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(lambda b:
            predict(
                mean_state_past=mean_state_filt[b],
                var_state_past=var_state_filt[b],
                mean_state=mean_state[b],
                trans_state=trans_state[b],
                var_state=var_state[b]
            )
        )(jnp.arange(n_block))
        # model interrogation
        trans_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        # kalman update
        mean_state_next, var_state_next = jax.vmap(lambda b:
            update(
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                W=W[b],
                x_meas=x_meas[b],
                mean_meas=mean_meas[b],
                trans_meas=trans_meas[b],
                var_meas=var_meas[b]
            )
        )(jnp.arange(n_block))
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "key": key
        }
        stack = {
            "state_filt": (mean_state_next, var_state_next),
            "state_pred": (mean_state_pred, var_state_pred)
        }
        return carry, stack
    # scan initial value
    scan_init = {
        "state_filt": (mean_state_init, var_state_init),
        "key": key
    }
    # scan itself
    _, scan_out = jax.lax.scan(scan_fun, scan_init, jnp.arange(n_steps))
    # append initial values to front
    scan_out["state_filt"] = (
        jnp.concatenate([mean_state_init[None], scan_out["state_filt"][0]]),
        jnp.concatenate([var_state_init[None], scan_out["state_filt"][1]])
    )
    scan_out["state_pred"] = (
        jnp.concatenate([mean_state_init[None], scan_out["state_pred"][0]]),
        jnp.concatenate([var_state_init[None], scan_out["state_pred"][1]])
    )
    return scan_out

def solve_sim(key, fun, W, x0, theta,
              tmin, tmax, n_steps,
              trans_state, mean_state, var_state,
              interrogate=interrogate_rodeo):
    r"""
    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (ndarray(n_steps+1, n_blocks, n_bstate)): Sample solution for :math:`X_t` at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mean_state.shape
    key, *subkeys = jax.random.split(key, num=n_steps*n_block+1)
    subkeys = jnp.reshape(jnp.array(subkeys), newshape=(n_steps, n_block, 2))

    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    # lax.scan setup
    def scan_fun(x_state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        key = smooth_kwargs['key']

        def vmap_fun(b):
            mean_state_sim, var_state_sim = smooth_sim(
                x_state_next=x_state_next[b],
                trans_state=trans_state[b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b]
            )
            return jax.random.multivariate_normal(key[b], mean_state_sim, var_state_sim)

        x_state_curr = jax.vmap(lambda b:
            vmap_fun(b)
        )(jnp.arange(n_block))
        return x_state_curr, x_state_curr
    # initialize
    scan_init = jax.vmap(lambda b:
                         jax.random.multivariate_normal(
                             subkeys[n_steps-1, b], 
                             mean_state_filt[n_steps, b],
                             var_state_filt[n_steps, b]))(jnp.arange(n_block))
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1],
        'key': subkeys[:n_steps-1]
    }
    # Note: initial value x0 is assumed to be known, so we don't
    # sample it.  In fact, doing so would probably fail due to cholesky
    # of a zero variance matrix...
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [x0[None], scan_out, scan_init[None]]
    )
    return x_state_smooth

def solve_mv(key, fun, W, x0, theta,
             tmin, tmax, n_steps,
             trans_state, mean_state, var_state,
             interrogate=interrogate_rodeo):
    r"""
    Mean and variance of the stochastic ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mean_state.shape
    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        mean_state_curr, var_state_curr = jax.vmap(lambda b:
            smooth_mv(
                mean_state_next=state_next["mean"][b],
                var_state_next=state_next["var"][b],
                trans_state=trans_state[b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
            )
        )(jnp.arange(n_block))
        state_curr = {
            "mean": mean_state_curr,
            "var": var_state_curr
        }
        return state_curr, state_curr
    # initialize
    scan_init = {
        "mean": mean_state_filt[n_steps],
        "var": var_state_filt[n_steps]
    }
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    mean_state_smooth = jnp.concatenate(
        [x0[None], scan_out["mean"], scan_init["mean"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_block, n_bstate, n_bstate))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth

def solve(key, fun, W, x0, theta,
          tmin, tmax, n_steps,
          trans_state, mean_state, var_state,
          interrogate=interrogate_rodeo):
    r"""
    Both random draw and mean/variance of the stochastic ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **x_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Sample solution for :math:`X_t` at times :math:`t \in [a, b]`.
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mean_state.shape
    key, *subkeys = jax.random.split(key, num=n_steps * n_block + 1)
    subkeys = jnp.reshape(jnp.array(subkeys), newshape=(n_steps, n_block, 2))

    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax,
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        #z_state = smooth_kwargs['z_state']
        key = smooth_kwargs['key']
        
        def vmap_fun(b):
            mean_state_sim, var_state_sim, mean_state_curr, var_state_curr = smooth(
                x_state_next=state_next["x"][b],
                mean_state_next=state_next["mean"][b],
                var_state_next=state_next["var"][b],
                trans_state=trans_state[b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b]
            )

            x_state_curr = jax.random.multivariate_normal(key[b], mean_state_sim, var_state_sim)
            return x_state_curr, mean_state_curr, var_state_curr

        x_state_curr, mean_state_curr, var_state_curr = jax.vmap(lambda b:
            vmap_fun(b)
        )(jnp.arange(n_block))
        state_curr = {
            "x": x_state_curr,
            "mean": mean_state_curr,
            "var": var_state_curr
        }
        return state_curr, state_curr
    # initialize

    x_init = jax.vmap(lambda b:
                      jax.random.multivariate_normal(
                          subkeys[n_steps-1, b],
                          mean_state_filt[n_steps, b],
                          var_state_filt[n_steps, b]))(jnp.arange(n_block))
    scan_init = {
        "x": x_init,
        "mean": mean_state_filt[n_steps],
        "var": var_state_filt[n_steps]
    }
    # scan arguments
    # Slice these arrays so they are aligned.
    # More precisely, for time step t, want filt[t], pred[t+1], z_state[t-1]
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1],
        'key': subkeys[:n_steps-1]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [x0[None], scan_out["x"], scan_init["x"][None]]
    )
    mean_state_smooth = jnp.concatenate(
        [x0[None], scan_out["mean"], scan_init["mean"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_block, n_bstate, n_bstate))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return x_state_smooth, mean_state_smooth, var_state_smooth
