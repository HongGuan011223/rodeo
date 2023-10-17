import jax
import jax.numpy as jnp
import jax.scipy as jsp

def add_sqrt(sqrt_A,sqrt_B):
    """
    Transform any “square-root” of the matrices A and B math:`(A^(1/2),B^(1/2))` to the squre root of the sum A and B math:`(A+B)^(1/2)`.

    Args:
        srqt_A (jax.numpy.ndarray): The square root of matrix A.
        sqrt_B (jax.numpy.ndarray): The square root of matrix B.
        
    Returns:
    jax.numpy.ndarray: The square root of the sum of A and B.
    """
    sqrt_sum = jnp.vstack([sqrt_A.T,sqrt_B.T])
    Q,R = jnp.linalg.qr(sqrt_sum)

    return R.T


