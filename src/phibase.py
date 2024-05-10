import jax
import jax.numpy as jnp
from functools import partial

####################################################################################
# def logphi_base(sp_orbitals, orb_state_indices, w_indices, x):
#     """
#         Base wavefunction of the system.
#             log(psi) = log(psi_1(x_1)) + log(psi_2(x_2)) + ...
#         Variable name:
#             psi_values = [psi_1(x_1), psi_2(x_2), ...])]
#             logpsi_values = [log(psi_1(x_1)), log(psi_2(x_2)), ...])]
#             logpsi_base = log(psi_1(x_1)) + log(psi_2(x_2)) + ...
#     """
#     phi_values = [sp_orbitals(jnp.array([state]), x[i], jnp.array([w])) 
#                 for i, (state, w) in enumerate(zip(orb_state_indices, w_indices))]
#     logphi_values = jnp.log(jnp.abs(jnp.array(phi_values)))
#     return jnp.sum(logphi_values)

####################################################################################
def logphi_base(sp_orbitals, orb_state_indices, w_indices, x):
    
    n = x.shape[0]
    phi_values = [sp_orbitals(jnp.array([orb_state_indices[i]]), x[i], w_indices[i])
                    for i in range(n)]
    logphi_values = jnp.log(jnp.abs(jnp.array(phi_values)))
    
    return jnp.sum(logphi_values)

####################################################################################
def logphi_base_vscf(sp_orbitals, orb_state_indices, w_indices, C_vscf, x):

    n, nlev = C_vscf.shape[0], C_vscf.shape[1]
    phi_values = jnp.array([[C_vscf[i, j, orb_state_indices[i]] * sp_orbitals(jnp.array([j]), x[i], w_indices[i]) 
                            for j in range(nlev)] 
                            for i in range(n)])
    phi_values = jnp.sum(jnp.reshape(phi_values, (n, nlev)), axis=1)
    logphi_values = jnp.log(jnp.abs(phi_values))
    
    return jnp.sum(logphi_values)