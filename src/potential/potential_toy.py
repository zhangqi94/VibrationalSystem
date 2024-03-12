import jax
import jax.numpy as jnp

from functools import partial

####################################################################################
"""
    For two-mode system: 
    Ref: Self-consistent field energies and wavefunctions for coupled oscillators
         Joel M. Bowman
         J. Chem. Phys. 68, 608-610 (1978)
         https://doi.org/10.1063/1.435782
"""
def get_potential_energy_mode2(lam, wsquare_indices):
    w1square, w2square = wsquare_indices
    w_indices = jnp.sqrt(wsquare_indices)
    lam1, lam2 = lam
    
    @partial(jax.jit)
    @partial(jax.vmap, in_axes=0, out_axes=0)
    def potential_energy(x):
        x1, x2 = x[0, 0], x[1, 0]
        V = 0.5 * w1square * (x1**2) + 0.5 * w2square * (x2**2) \
          + lam1 * ((x2**2) * x1 + lam2 * (x1**3))
        return V
    return potential_energy, w_indices


####################################################################################
"""
    For three-mode system: 
    Ref: Investigations of self-consistent field, scf ci and virtual state configuration 
         interaction vibrational energies for a model three-mode system.
         Kurt M. Christoffel and Joel M. Bowman
         chemical physics letters, volume 85, number 2, 1982
"""
def get_potential_energy_mode3(lam, wsquare_indices):
    w1square, w2square, w3square = wsquare_indices
    w_indices = jnp.sqrt(wsquare_indices)
    lam1, lam2, lam3, lam4 = lam
    
    @partial(jax.jit)
    @partial(jax.vmap, in_axes=0, out_axes=0)
    def potential_energy(x):
        x1, x2, x3 = x[0, 0], x[1, 0], x[2, 0]
        V = 0.5 * w1square * (x1**2) + 0.5 * w2square * (x2**2) + 0.5 * w3square * (x3**2) \
          + lam1 * lam3 * (x1 **3) + lam2 * lam4 * (x2 **3) + lam1 * x1 * (x2 **2) + lam2 * x2 * (x3 **2)
        return V
    return potential_energy, w_indices

