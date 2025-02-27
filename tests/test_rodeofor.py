import unittest
import jax
from rodeo.ode import *
import ode_block_solve_for as bfor
import utils
# from jax.config import config
# config.update("jax_enable_x64", True)

class TestrodeoFor(unittest.TestCase):
    """
    Test if lax scan version of rodeo gives the same results as for-loop version.

    """
    setUp = utils.fitz_setup 

    def test_interrogate_rodeo(self):
        trans_meas1, mean_meas1, var_meas1 = interrogate_rodeo(
            key=self.key,
            fun=self.fitz_jax,
            W=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.var_block
        )
        # for
        trans_meas2, mean_meas2, var_meas2 = bfor.interrogate_rodeo(
            key=self.key,
            fun=self.fitz_jax,
            W = self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.var_block
        )
        
        self.assertAlmostEqual(utils.rel_err(trans_meas1, trans_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_meas1, mean_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)
    
    def test_interrogate_chkrebtii(self):
        trans_meas1, mean_meas1, var_meas1 = interrogate_chkrebtii(
            key=self.key,
            fun=self.fitz_jax,
            W=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.var_block
        )
        # for
        trans_meas2, mean_meas2, var_meas2 = bfor.interrogate_chkrebtii(
            key=self.key,
            fun=self.fitz_jax,
            W=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.var_block
        )
        
        self.assertAlmostEqual(utils.rel_err(trans_meas1, trans_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_meas1, mean_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)

    def test_solve_sim(self):
        sim1 = solve_sim(key=self.key, fun=self.fitz_jax, W=self.W_block,
                         x0=self.x0_block, theta=self.theta,
                         tmin=self.tmin, tmax=self.tmax,
                         n_steps=self.n_steps, **self.ode_init)
        # for
        sim2 = bfor.solve_sim(key=self.key, fun=self.fitz_jax, W=self.W_block,
                              x0=self.x0_block, theta=self.theta,
                              tmin=self.tmin, tmax=self.tmax,
                              n_steps=self.n_steps, **self.ode_init)
        self.assertAlmostEqual(utils.rel_err(sim1, sim2), 0.0)
    
    def test_solve_mv(self):
        mu1, var1 = solve_mv(key=self.key, fun=self.fitz_jax, W=self.W_block,
                             x0=self.x0_block, theta=self.theta,
                             tmin=self.tmin, tmax=self.tmax, 
                             n_steps=self.n_steps, **self.ode_init)
        # for
        mu2, var2 = bfor.solve_mv(key=self.key, fun=self.fitz_jax, W=self.W_block,
                                  x0=self.x0_block, theta=self.theta,
                                  tmin=self.tmin, tmax=self.tmax,
                                  n_steps=self.n_steps, **self.ode_init)
        self.assertAlmostEqual(utils.rel_err(mu1, mu2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var1, var2), 0.0)
    
    def test_solve(self):
        sim1, mu1, var1 = solve(key=self.key, fun=self.fitz_jax, W=self.W_block,
                                x0=self.x0_block, theta=self.theta,
                                tmin=self.tmin, tmax=self.tmax,
                                n_steps=self.n_steps, **self.ode_init)
        # for
        sim2, mu2, var2 = bfor.solve(key=self.key, fun=self.fitz_jax, W=self.W_block,
                                     x0=self.x0_block, theta=self.theta,
                                     tmin=self.tmin, tmax=self.tmax,
                                     n_steps=self.n_steps, **self.ode_init)
        self.assertAlmostEqual(utils.rel_err(mu1, mu2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var1, var2), 0.0)
        self.assertAlmostEqual(utils.rel_err(sim1, sim2), 0.0)

if __name__ == '__main__':
    unittest.main()
