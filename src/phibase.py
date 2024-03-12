import jax
import jax.numpy as jnp
from functools import partial

####################################################################################
def logphi_base(sp_orbitals, orb_state_indices, w_indices, x):
    """
        Base wavefunction of the system.
            log(psi) = log(psi_1(x_1)) + log(psi_2(x_2)) + ...
        Variable name:
            psi_values = [psi_1(x_1), psi_2(x_2), ...])]
            logpsi_values = [log(psi_1(x_1)), log(psi_2(x_2)), ...])]
            logpsi_base = log(psi_1(x_1)) + log(psi_2(x_2)) + ...
    """
    phi_values = [sp_orbitals(jnp.array([state]), x[i], jnp.array([w])) 
                for i, (state, w) in enumerate(zip(orb_state_indices, w_indices))]
    logphi_values = jnp.log(jnp.abs(jnp.array(phi_values)))
    return jnp.sum(logphi_values)
