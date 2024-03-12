import jax
import jax.numpy as jnp

from functools import partial

####################################################################################
# MCMC: Markov Chain Monte Carlo sampling algorithm
from .mcmc import mcmcw

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, 0,
                            None, 0, 0, 
                            None, None, None),
                   static_broadcasted_argnums=(2))
def sample_x_mcmc(key, state_indices,
                  logp, x, params_flow,
                  mc_steps, mc_stddev, invsqrtw):
    """
        Generate new x.shape(batch, n, dim).
    """
    key, key_MCMC = jax.random.split(key, 2)
    x, accept_rate = mcmcw(lambda x: logp(x, params_flow, state_indices), 
                          x, key_MCMC, mc_steps, mc_stddev, invsqrtw)
    return key, x, accept_rate

####################################################################################

def make_loss(logpsi, logpsi_grad_laplacian, potential_energy):

    def observable_and_lossfn(params_flow, state_indices, x, key):

        grad, laplacian = logpsi_grad_laplacian(x, params_flow, state_indices, key)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)
        
        kinetic = (- 0.5 * (laplacian + (grad**2).sum(axis=(-2, -1)))).real
        potential = potential_energy(x).real

        Eloc = kinetic + potential

        K_mean, K2_mean, V_mean, V2_mean, E_mean, E2_mean = \
        jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                     (kinetic.real.mean(), (kinetic.real**2).mean(),
                      potential.mean(), (potential**2).mean(),
                      Eloc.real.mean(), (Eloc.real**2).mean(),))
        observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                      "V_mean": V_mean, "V2_mean": V2_mean,
                      "E_mean": E_mean, "E2_mean": E2_mean,}

        return observable

    return observable_and_lossfn
