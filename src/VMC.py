import jax
import jax.numpy as jnp

from functools import partial

####################################################################################
# MCMC: Markov Chain Monte Carlo sampling algorithm
# from .mcmc import mcmcw

# # Only sample state_indices and x
# @partial(jax.pmap, axis_name="p",
#                    in_axes=(0, None, 0,
#                             None, 0, 0, 
#                             None, None, None),
#                    static_broadcasted_argnums=(1, 3))
# def sample_stateindices_and_x_mcmc(key, sampler, params_prob,
#                                    logp, x, params_flow,
#                                    mc_steps, mc_stddev, invsqrtw):
#     """
#         Generate new state_indices.shape(batch, ) and x.shape(batch, n, dim).
#     """
#     key, key_sampler, key_MCMC = jax.random.split(key, 3)
#     batch = x.shape[0]
#     state_indices = sampler(params_prob, key_sampler, batch)
#     x, accept_rate = mcmcw(lambda x: logp(x, params_flow, state_indices), 
#                            x, key_MCMC, mc_steps, mc_stddev, invsqrtw)
#     return key, x, state_indices, accept_rate

# Only sample x
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

def make_loss(logpsi, logpsi_grad_laplacian, potential_energy, clip_factor, 
              batch_per_device, num_orb, print_levels=0):

    def observable_and_lossfn(params_flow, state_indices, x, key):
        #========== calculate Eloc & E_mean ==========
        grad, laplacian = logpsi_grad_laplacian(x, params_flow, state_indices, key)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)
        
        kinetic = (- 0.5 * (laplacian + (grad**2).sum(axis=(-2, -1)))).real
        potentl = (potential_energy(x)).real
        Eloc = jax.lax.stop_gradient(kinetic + potentl)
        print("K.shape:", kinetic.shape)
        print("V.shape:", potentl.shape)
        print("Eloc.shape:", Eloc.shape)

        LossK = (kinetic.reshape(batch_per_device, num_orb)).sum(axis=1)
        LossV = (potentl.reshape(batch_per_device, num_orb)).sum(axis=1)
        LossE = (Eloc.reshape(batch_per_device, num_orb)).sum(axis=1)
        print("LossK.shape", LossK.shape)
        print("LossV.shape", LossV.shape)
        print("LossE.shape", LossE.shape)
        
        if print_levels:
            Levels = Eloc.reshape(batch_per_device, num_orb)
            print("Levels.shape", Levels.shape)
            Levels = jax.lax.pmean(Levels.mean(axis=0), axis_name="p")
        
        K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, \
        LK_mean, LK2_mean, LV_mean, LV2_mean, LE_mean, LE2_mean = \
        jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                     (kinetic.mean(), (kinetic**2).mean(),
                      potentl.mean(), (potentl**2).mean(),
                      Eloc.mean(),    (Eloc**2).mean(),
                      LossK.mean(),   (LossK**2).mean(),
                      LossV.mean(),   (LossV**2).mean(),
                      LossE.mean(),   (LossE**2).mean(),
                      ))

        if print_levels:
            observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                        "V_mean": V_mean, "V2_mean": V2_mean,
                        "E_mean": E_mean, "E2_mean": E2_mean,
                        "LK_mean": LK_mean, "LK2_mean": LK2_mean,
                        "LV_mean": LV_mean, "LV2_mean": LV2_mean,
                        "LE_mean": LE_mean, "LE2_mean": LE2_mean,
                        "Levels": Levels}
        else:
            observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                        "V_mean": V_mean, "V2_mean": V2_mean,
                        "E_mean": E_mean, "E2_mean": E2_mean,
                        "LK_mean": LK_mean, "LK2_mean": LK2_mean,
                        "LV_mean": LV_mean, "LV2_mean": LV2_mean,
                        "LE_mean": LE_mean, "LE2_mean": LE2_mean}

        def quant_lossfn(params_flow):
            logpsix = logpsi(x, params_flow, state_indices)

            tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E_mean - clip_factor*tv, E_mean + clip_factor*tv)
            gradF_theta = 2 * (logpsix * Eloc_clipped.conj()).real.mean()
            quantum_score = 2 * logpsix.real.mean()
            return gradF_theta, quantum_score

        return observable, quant_lossfn

    return observable_and_lossfn
