import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial


####################################################################################
def calculate_frequency(D):
    
    #========== calculate exact eigenvalues ==========
    wsquare_indices = 0.5 * np.arange(1, D+1)
    w_indices = np.sqrt(wsquare_indices)

    a = 0.1
    Aij = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            if i == j:
                Aij[i,j] = w_indices[i]**2
            else:
                Aij[i,j] = a * np.sqrt(w_indices[i] * w_indices[j])
    Aij = jnp.array(Aij)
                
    nu, _ = jax.numpy.linalg.eigh(Aij)
    w_indices = jnp.sqrt(nu)

    return w_indices

####################################################################################
def get_potential_energy_harmonic(D=64):
    """
    For N-D bilinearly coupled oscillator,
    Ref: [1] Calculating vibrational spectra of molecules using tensor train decomposition.
                https://doi.org/10.1063/1.4962420
         [2] Using Nested Contractions and a Hierarchical Tensor Format To Compute 
                Vibrational Spectra of Molecules with Seven Atoms
    """
    
    w_indices = calculate_frequency(D)

    Vij = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            if i == j:
                Vij[i,j] = 0.5 * w_indices[i]**2

    Vij = jnp.array(Vij)

    @partial(jax.jit)
    @partial(jax.vmap, in_axes=0, out_axes=0)
    def potential_energy(x):
        q = x[:, 0]
        V = jnp.einsum('i,ij,j->', q, Vij, q)
        return V

    return potential_energy, jnp.array(w_indices)

####################################################################################
def get_first_N_orbitals(nu, num_orb):
    """
        Finds first N elemens of a sum:
            const + nu[0]*n_0 + nu[1]*n_1 + ... + nu[d]*n_d, where n_i = 0, 1, 2, ...
    """

    nu = np.array(nu, dtype=np.float64)
    n = nu.size
    const = np.sum(nu)*0.5
    V = set([tuple(np.zeros(n))])
    ans = np.zeros((num_orb, n))
    values = np.zeros(num_orb)
    for iter_idx in range(1, num_orb):
        best_val = float("+inf")
        for i in range(iter_idx):
            for j in range(n):
                curr_seq = ans[i, :].copy()
                curr_seq[j] += 1
                if tuple(curr_seq) not in V:
                    curr_val = values[i] + nu[j]
                    if curr_val < best_val:
                        best_seq = curr_seq
                        best_val = curr_val
                    break
        ans[iter_idx, :] = best_seq
        values[iter_idx] = best_val
        V.add(tuple(best_seq))
        
    orb_Es = jnp.array(values + const, dtype=jnp.float64)
    orb_state = jnp.array(ans, dtype=jnp.int64)  
    orb_index = jnp.arange(num_orb)
        
    return orb_index, orb_state, orb_Es

####################################################################################
def calculate_exact_energy_harmonic(D=64, num_orb=80):
    
    nu = calculate_frequency(D)
    
    #========== calculate excited state orbitals ==========
    orb_index, orb_state, orb_Es = get_first_N_orbitals(nu, num_orb)

    return orb_index, orb_state, orb_Es, nu

####################################################################################
# test
if __name__ == "__main__":
    #==== test for potential ====
    print("\n==== test potential ====")
    batch, n, dim = 10, 20, 1
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch, n, dim))
    
    potential_energy, w_indices = get_potential_energy_harmonic(D=n)
    print("n:", n)
    print("w:", w_indices)
    print("w^2:", w_indices**2)
    print("x.shape:", x.shape)
    print("V(x):", potential_energy(x))
    
    #==== exact energy levels ====
    num_orb = 21
    orb_idx, orb_state, orb_Es, nu = calculate_exact_energy_harmonic(D=n, num_orb=num_orb)
    print("\n==== exact values ====")
    print("nu:", nu)
    import orbitals
    for ii in range(num_orb): 
        print("    %d, E: %.9f, orbital:" %(ii, orb_Es[ii]), orbitals.orbitals_array2str(orb_state[ii]))
    
    # beta = 1.0
    # lnZ = np.logaddexp.reduce(-beta * orb_Es)
    # F = - lnZ / beta
    # print("free energy at beta: %.4f,    F:%.9f" %(beta, F))
