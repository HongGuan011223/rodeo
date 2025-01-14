r"""
Util functions for Jax kalmantv.

"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as jsl

def block_diag(X):
    r"""
    Convert an array with blocks to a block diagonal matrix.

    Args:
        X (ndarray(n_eval, n_block, n_dim, n_dim)): Array containing blocks of matrices.
    
    Returns:
        (ndarray(n_eval, n_block * n_dim, n_block * n_dim)): Block diagonal matrix created from the blocks.

    """
    n_eval = X.shape[0]
    mat = jax.vmap(lambda t:
                   jsp.linalg.block_diag(*X[t]))(jnp.arange(n_eval))
        
    return mat

def mvncond(mu, Sigma, icond):
    """
    Calculates A, b, and V such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)`.

    Args:
        mu (ndarray(2*n_dim)): Mean of y.
        Sigma (ndarray(2*n_dim, 2*n_dim)): Covariance of y. 
        icond (ndarray(2*nd_dim)): Conditioning on the terms given.

    Returns:
        (tuple):
        - **A** (ndarray(n_dim, n_dim)): For :math:`y \sim N(\mu, \Sigma)` 
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate A.
        - **b** (ndarray(n_dim)): For :math:`y \sim N(\mu, \Sigma)` 
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate b.
        - **V** (ndarray(n_dim, n_dim)): For :math:`y \sim N(\mu, \Sigma)`
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate V.

    """
    # if y1 = y[~icond] and y2 = y[icond], should have A = Sigma12 * Sigma22^{-1}
    ficond = jnp.nonzero(~icond)
    ticond = jnp.nonzero(icond)
    A = jnp.dot(Sigma[jnp.ix_(ficond[0], ticond[0])], jsl.cho_solve(
        jsl.cho_factor(Sigma[jnp.ix_(ticond[0], ticond[0])]), jnp.identity(sum(icond))))
    b = mu[~icond] - jnp.dot(A, mu[icond])  # mu1 - A * mu2
    V = Sigma[jnp.ix_(ficond[0], ficond[0])] - jnp.dot(A, Sigma[jnp.ix_(ticond[0], ficond[0])])  # Sigma11 - A * Sigma21
    return A, b, V
