import rodeo
import rodeo.gauss_markov as gm
import rodeo.kalmantv as ktv
from rodeo.utils import mvncond
from utils import gauss_dens
from double_filter import *
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.config import config
config.update("jax_enable_x64", True)


def print_diff(name, x1, x2):
    ad = np.max(np.abs(x1 - x2))
    print(name + " abs diff = {}".format(ad))
    return ad


key = jax.random.PRNGKey(0)

n_meas = 3
n_obs = 2
n_state = 4
n_tot = 3
n_res = 1

key, *subkeys = jax.random.split(key, 14)
# states
_mu_state = jax.random.normal(subkeys[0], (n_state,))
mu_state = jnp.repeat(_mu_state[jnp.newaxis], n_tot*n_res-1, 0)
_var_state = jax.random.normal(subkeys[1], (n_state, n_state))
_var_state = _var_state.dot(_var_state.T)
var_state = jnp.repeat(_var_state[jnp.newaxis], n_tot*n_res-1, 0)
_wgt_state = jax.random.normal(subkeys[2], (n_state, n_state))
wgt_state = jnp.repeat(_wgt_state[jnp.newaxis], n_tot*n_res-1, 0)
# meas
_mu_meas = jax.random.normal(subkeys[3], (n_meas,))
mu_meas = jnp.repeat(_mu_meas[jnp.newaxis], n_tot*n_res, 0)
# _var_meas = jax.random.normal(subkeys[4], (n_meas, n_meas))
# _var_meas = _var_meas.dot(_var_meas.T)
# _var_meas = 1e-10*jnp.eye(n_meas)
_var_meas = jnp.zeros((n_meas, n_meas))
var_meas = jnp.repeat(_var_meas[jnp.newaxis], n_tot*n_res, 0)
_wgt_meas = jax.random.normal(subkeys[5], (n_meas, n_state))
wgt_meas = jnp.repeat(_wgt_meas[jnp.newaxis], n_tot*n_res, 0)
# z_meas = jax.random.normal(subkeys[6], (n_tot*n_res, n_meas))
z_meas = jnp.zeros((n_tot*n_res, n_meas))
# obs
_mu_obs = jax.random.normal(subkeys[7], (n_obs,))
mu_obs = jnp.repeat(_mu_obs[jnp.newaxis], n_tot*n_res, 0)
_var_obs = jax.random.normal(subkeys[8], (n_obs, n_obs))
_var_obs = _var_obs.dot(_var_obs.T)
var_obs = jnp.repeat(_var_obs[jnp.newaxis], n_tot*n_res, 0)
_wgt_obs = jax.random.normal(subkeys[9], (n_obs, n_state))
wgt_obs = jnp.repeat(_wgt_obs[jnp.newaxis], n_tot*n_res, 0)
y_obs = jax.random.normal(subkeys[10], (n_tot, n_obs))
# Initial x0
x0 = jax.random.normal(subkeys[11], (n_state,))
mu_state_init = x0
# var_state_init = jax.random.normal(subkeys[12], (n_state, n_state))
# var_state_init = var_state_init.dot(var_state_init.T)
var_state_init = jnp.zeros((n_state, n_state))

mu_state = jnp.vstack([mu_state_init, mu_state])
var_state = jnp.concatenate([var_state_init[jnp.newaxis], var_state])
# Concatenate variables for (Y, Z)
wgt_obs_meas = jax.vmap(lambda t: jnp.concatenate([wgt_obs[t], wgt_meas[t]]))(jnp.arange(n_tot*n_res))
mu_obs_meas = jax.vmap(lambda t: jnp.append(mu_obs[t], mu_meas[t]))(jnp.arange(n_tot*n_res))
var_obs_meas = jax.vmap(lambda t: jsp.linalg.block_diag(var_obs[t], var_meas[t]))(jnp.arange(n_tot*n_res))

# --- Y_{0:1} | Z_{0:} -----------------------------------------------------------------
A_gm, b_gm, C_gm = gm.kalman2gm(
    wgt_state=wgt_state,
    mu_state=mu_state,
    var_state=var_state,
    wgt_meas=wgt_obs_meas,
    mu_meas=mu_obs_meas,
    var_meas=var_obs_meas
)

mu_gm, var_gm = gm.gauss_markov_mv(A=A_gm, b=b_gm, C=C_gm)

logdens = double_filter(
    x0, mu_state[1], wgt_state[0], var_state[1],
    mu_meas[0], wgt_meas[0], var_meas[0], z_meas,
    mu_obs[0], wgt_obs[0], var_obs[0], y_obs
)
mu_yz = mu_gm[:, n_state:]
var_yz = var_gm[:, n_state:, :, n_state:]
logdens2 = gauss_dens(
    z=z_meas[1:], y=y_obs, mu=mu_yz, Sigma=var_yz, n_res=n_res
)
print_diff("gss logdens", logdens, logdens2)

# --- n_res > 1 ------------------------------------------------------------------------
n_res = 2

# states
mu_state = jnp.repeat(_mu_state[jnp.newaxis], n_tot*n_res-1, 0)
var_state = jnp.repeat(_var_state[jnp.newaxis], n_tot*n_res-1, 0)
wgt_state = jnp.repeat(_wgt_state[jnp.newaxis], n_tot*n_res-1, 0)
# meas
mu_meas = jnp.repeat(_mu_meas[jnp.newaxis], n_tot*n_res, 0)
var_meas = jnp.repeat(_var_meas[jnp.newaxis], n_tot*n_res, 0)
wgt_meas = jnp.repeat(_wgt_meas[jnp.newaxis], n_tot*n_res, 0)
# z_meas = jax.random.normal(subkeys[6], (n_tot*n_res, n_meas))
z_meas = jnp.zeros((n_tot*n_res, n_meas))
# obs
mu_obs = jnp.repeat(_mu_obs[jnp.newaxis], n_tot*n_res, 0)
var_obs = jnp.repeat(_var_obs[jnp.newaxis], n_tot*n_res, 0)
wgt_obs = jnp.repeat(_wgt_obs[jnp.newaxis], n_tot*n_res, 0)
y_obs = jax.random.normal(subkeys[10], (n_tot, n_obs))
y_out = jnp.ones((n_tot*n_res, n_obs))*jnp.nan
for i in range(n_tot):
    y_out = y_out.at[i*n_res].set(y_obs[i])

mu_state = jnp.vstack([mu_state_init, mu_state])
var_state = jnp.concatenate([var_state_init[jnp.newaxis], var_state])
# Concatenate variables for (Y, Z)
wgt_obs_meas = jax.vmap(lambda t: jnp.concatenate([wgt_obs[t], wgt_meas[t]]))(jnp.arange(n_tot*n_res))
mu_obs_meas = jax.vmap(lambda t: jnp.append(mu_obs[t], mu_meas[t]))(jnp.arange(n_tot*n_res))
var_obs_meas = jax.vmap(lambda t: jsp.linalg.block_diag(var_obs[t], var_meas[t]))(jnp.arange(n_tot*n_res))

A_gm, b_gm, C_gm = gm.kalman2gm(
    wgt_state=wgt_state,
    mu_state=mu_state,
    var_state=var_state,
    wgt_meas=wgt_obs_meas,
    mu_meas=mu_obs_meas,
    var_meas=var_obs_meas
)

mu_gm, var_gm = gm.gauss_markov_mv(A=A_gm, b=b_gm, C=C_gm)

logdens = double_filter(
    x0, mu_state[1], wgt_state[0], var_state[1],
    mu_meas[0], wgt_meas[0], var_meas[0], z_meas,
    mu_obs[0], wgt_obs[0], var_obs[0], y_out
)

mu_yz = mu_gm[:, n_state:]
var_yz = var_gm[:, n_state:, :, n_state:]
logdens2 = gauss_dens(
    z=z_meas[1:], y=y_obs, mu=mu_yz, Sigma=var_yz, n_res=n_res
)
print_diff("gss logdens, n_res>1", logdens, logdens2)

# # --- ODE ------------------------------------------------------------------------
tmin = 0.
tmax = 1.
x0 = mu_state_init
W = wgt_meas[0]
theta = jnp.zeros((n_state,))

def ode_fun(X, t, theta):
    return -mu_meas[0]

logdens3 = double_ode_filter(key, ode_fun, x0, theta,
                             tmin, tmax, W,
                             mu_state[1], wgt_state[0], var_state[1],
                             mu_obs[0], wgt_obs[0], var_obs[0], y_out)

print_diff("ode logdens", logdens, logdens3)
