import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from inference.fitzinf import fitzinf as inference
from rodeo.ibm_init import ibm_init
from rodeo.ode_solve import *

def fitz(X_t, t, theta):
    "Fitz ODE written for jax"
    a, b, c = theta
    V, R = X_t[:, 0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                      [-1/c*(V - a + b*R)]])

def fitzpad(X_t, t, theta):
    a, b, c = theta
    p = len(X_t)//2
    V, R = X_t[0], X_t[p]
    return jnp.array([[V, c*(V - V*V*V/3 + R), 0],
                      [R, -1/c*(V - a + b*R), 0]])


def fitz_example(load_calcs=False):
    "Perform parameter inference using the FitzHugh-Nagumo function."
    # These parameters define the order of the ODE and the CAR(p) process
    n_deriv = 1 # number of derivatives in IVP
    n_obs = 2 # number of observations.
    n_deriv_prior = 3 # number of derivatives in IBM prior

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    tmin = 0.
    tmax = 40.

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    sigma = jnp.array([.1]*n_obs)
    n_order = jnp.array([n_deriv_prior]*n_obs)

    # Initial value, x0, for the IVP
    x0 = np.array([-1., 1.])
    v0 = np.array([1, 1/3])
    X0 = np.ravel([x0, v0], 'F')

    # pad the inputs
    W_mat = np.zeros((n_obs, 1, n_deriv_prior))
    W_mat[:, :, 1] = 1
    W = jnp.array(W_mat)

    # logprior parameters
    theta_true = np.array([0.2, 0.2, 3]) # True theta
    n_theta = len(theta_true)
    phi_mean = np.zeros(n_theta)
    phi_sd = np.log(10)*np.ones(n_theta) 

    # Observation noise
    gamma = 0.2

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    key = jax.random.PRNGKey(0)
    inf = inference(key, tmin, tmax, fitz)
    inf.funpad = fitzpad
    tseq = np.linspace(tmin, tmax, 41)
    Y_t, X_t = inf.simulate(x0, theta_true, gamma, tseq)

    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    axs[0].plot(tseq, X_t[:,0], label = 'X_t')
    axs[0].scatter(tseq, Y_t[:,0], label = 'Y_t', color='orange')
    axs[0].set_title("$V^{(0)}_t$")
    axs[1].plot(tseq, X_t[:,1], label = 'X_t')
    axs[1].scatter(tseq, Y_t[:,1], label = 'Y_t', color='orange')
    axs[1].set_title("$R^{(0)}_t$")
    axs[1].legend(loc='upper left', bbox_to_anchor=[1, 1])
    fig.savefig('figures/fitzsim.pdf')
    
    dtlst = np.array([0.1, 0.05, 0.02, 0.01])
    obs_t = 1
    if load_calcs:
        theta_euler = np.load('saves/fitz_theta_euler.npy')
        theta_kalman = np.load('saves/fitz_theta_kalman.npy')
        theta_diffrax = np.load('saves/fitz_theta_diffrax.npy')
    else:
        # Parameter inference using Euler's approximation
        theta_euler = np.zeros((len(dtlst), n_samples, n_theta+n_obs))
        phi_init = np.append(np.log(theta_true), x0)
        for i in range(len(dtlst)):
            n_eval = int((tmax-tmin)/dtlst[i])
            inf.n_eval = n_eval
            phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None]), dtlst[i], obs_t, phi_mean, phi_sd, inf.euler_nlpost,
                                           gamma, phi_init=phi_init)
            theta_euler[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_euler[i, :, :n_theta] = np.exp(theta_euler[i, :, :n_theta])
            
        np.save('saves/fitz_theta_euler.npy', theta_euler)
        
        # Parameter inference using Kalman solver
        theta_kalman = np.zeros((len(dtlst), n_samples, n_theta+n_obs))
        for i in range(len(dtlst)):
            kinit = ibm_init(dtlst[i], n_order, sigma)
            n_eval = int((tmax-tmin)/dtlst[i])
            inf.n_eval = n_eval
            inf.kinit = kinit
            inf.W = W
            phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None]), dtlst[i], obs_t, phi_mean, phi_sd, inf.kalman_nlpost,
                                           gamma, phi_init = phi_init)
            theta_kalman[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_kalman[i, :, :n_theta] = np.exp(theta_kalman[i, :, :n_theta])
        np.save('saves/fitz_theta_kalman.npy', theta_kalman)

        # Parameter inference using diffrax
        phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None]), obs_t, obs_t, phi_mean, phi_sd, inf.diffrax_nlpost,
                                      gamma, phi_init = phi_init)
        theta_diffrax = inf.phi_sample(phi_hat, phi_var, n_samples)
        theta_diffrax[:, :n_theta] = np.exp(theta_diffrax[:, :n_theta])
        np.save('saves/fitz_theta_diffrax.npy', theta_diffrax)
        
    # Produces the graph in Figure 3
    plt.rcParams.update({'font.size': 20})
    var_names = ['a', 'b', 'c', r"$V_0^{(0)}$", r"$R_0^{(0)}$"]
    param_true = np.append(theta_true, np.array([-1, 1]))
    figure = inf.theta_plot(theta_euler, theta_kalman, theta_diffrax, param_true, dtlst, var_names, clip=[None, (0, 0.5), None, None, None], rows=1)
    figure.savefig('figures/fitzfigure.pdf')
    plt.show()
    return

if __name__ == '__main__':
    fitz_example(False)
