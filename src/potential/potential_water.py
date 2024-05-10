import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

####################################################################################
def get_w0():
    # get frequency
    w0 = jnp.array([1639.13, 3799.52, 3899.80])
    return w0

def get_k30():
    # get cubic force constant
    datastr = """
    1 1 1  -58.7
    1 1 2  -46.4
    2 2 1  60.2
    2 2 2  -301.3
    3 3 1  116.5
    3 3 2  -906.7
    """
    k30 = np.zeros((3,3,3), dtype=np.float64)
    lines = datastr.strip().split('\n')
    for line in lines:
        i, j, k, value = map(float, line.split())
        k30[int(i)-1, int(j)-1, int(k)-1] = value
    return k30

def get_k40():
    # get quartic force constants k40
    datastr = """
    1 1 1 1  -4.3
    1 1 1 2  7.8
    1 1 2 2  3.3
    1 1 3 3  -2.6
    2 2 2 1  -14
    2 2 2 2  31
    2 2 3 3  192.7
    3 3 1 2  -45.8
    3 3 3 3  32.3
    """
    k40 = np.zeros((3,3,3,3), dtype=np.float64)
    lines = datastr.strip().split('\n')
    for line in lines:
        i, j, k, l, value = map(float, line.split())
        k40[int(i)-1, int(j)-1, int(k)-1, int(l)-1] = value
    return k40
    
####################################################################################
def get_potential_energy_water(alpha=1000):
    """
    Potential of water molecule (H2O), Ref: 
        [1] Chemical Physics 54, 365 (1981)
        [2] Chemical Physics 300 (2004) 41-51
    Input:
        alpha: scaling factor, default=1000
    """
    
    w0 = get_w0()
    print("w0, harmonic constants:", np.count_nonzero(w0))
    w = w0 / alpha
    sqrtw = np.sqrt(w)
    
    k2 = (1/2) * jnp.diag(w**2)
    
    k30 = get_k30()
    print("k30, non-zero terms:", np.count_nonzero(k30))
    k3 = jnp.einsum('ijk,i,j,k->ijk', k30, sqrtw, sqrtw, sqrtw) / alpha

    k40 = get_k40()
    print("k40, non-zero terms:", np.count_nonzero(k40))
    k4 = jnp.einsum('ijkl,i,j,k,l->ijkl', k40, sqrtw, sqrtw, sqrtw, sqrtw) / alpha

    @partial(jax.jit)
    @partial(jax.vmap, in_axes=0, out_axes=0)
    def potential_energy(x):
        q = x[:, 0]
        V =   (1/2) * jnp.einsum('i,i->', w**2, q**2) \
            + jnp.einsum('ijk,i,j,k->', k3, q, q, q) \
            + jnp.einsum('ijkl,i,j,k,l->', k4, q, q, q, q)
            
        return V

    return potential_energy, jnp.array(w, dtype=jnp.float64), \
        jnp.array(k2, dtype=jnp.float64), jnp.array(k3, dtype=jnp.float64), jnp.array(k4, dtype=jnp.float64)

####################################################################################
# test
if __name__ == "__main__":
    batch, n, dim =10, 3, 1
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch, n, dim))
    
    alpha = 1000
    potential_energy, w_indices = get_potential_energy_water(alpha=alpha)
    print("w_indices:", w_indices)
    print("x.shape:", x.shape)
    print("V(x):", potential_energy(x))
    
    import orbitals
    num_orb = 14
    num_orb = 80
    # method 1
    orb_index, orb_state, orb_Es = orbitals.get_orbitals_indices_first(w_indices, max_orb=1000, num_orb = num_orb)
    orb_Es = orb_Es * alpha
    print("Total number of orbitals:", num_orb)
    for ii in range(num_orb): 
        print("    %d, E: %.3f, level:" %(ii, orb_Es[ii]), orbitals.orbitals_array2str(orb_state[ii]))
    
    # method 2
    orb_index, orb_state, orb_Es = orbitals.get_orbitals_indices(n, w_indices, max_idx=15, num_orb = num_orb)
    orb_Es = orb_Es * alpha
    print("Total number of orbitals:", num_orb)
    for ii in range(num_orb): 
        print("    %d, E: %.3f, level:" %(ii, orb_Es[ii]), orbitals.orbitals_array2str(orb_state[ii]))