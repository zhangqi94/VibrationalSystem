import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

####################################################################################
"""
    https://code.itp.ac.cn/-/snippets/20 
    hermite(n, x) = 1/sqrt(2^n * n!) / pi^(1/4) * H_n(x)
"""

@jax.vmap
def hermite0(indices, x):
    h0 = 1. / jnp.pi**(1/4)
    h1 = jnp.sqrt(2.)*x / jnp.pi**(1/4)

    def body_fun(i, val):
        valm2, valm1 = val
        return valm1, jnp.sqrt(2./i)*x*valm1 - jnp.sqrt((i-1)/i)*valm2
    _, hn = jax.lax.fori_loop(2, indices+1, body_fun, (h0, h1))

    return jax.lax.cond(indices>0, lambda: hn, lambda: h0)

hermite = jax.custom_jvp(hermite0, nondiff_argnums=(0,))

@hermite.defjvp
def hermite_jvp(indices, primals, tangents):
    #print("hermite_jvp...")
    x, = primals
    dx, = tangents
    hn = hermite(indices, x)
    dhn = jnp.sqrt(2*indices) * hermite((indices-1)*(indices>0), x) * dx
    primals_out, tangents_out = hn, dhn
    return primals_out, tangents_out


####################################################################################
## get 1d orbitals
def get_orbitals_1d(m=1.0):
    """
    Computes the 1D harmonic oscillator wavefunction.

    Parameters:
        indices: Quantum numbers.
        m (float): Mass parameter (default is 1.0).
        x (float): Position coordinate.
        w (float): Oscillator frequency (default is 1.0).

    Returns:
        sp_orbital (function): Function for 1D harmonic oscillator wavefunction.
            psi_n(x) = ((m*w)**(1/4)) * exp(- 0.5 * m*w * (x**2) ) * H_n(sqrt(m*w) * x)
        sp_energy (function): Function for 1D harmonic oscillator energy.
            E_nx = (nx + 0.5) * w
    """
    
    def sp_orbital(indices, x, w=1.0):
        return ((m*w)**(1/4)) * jnp.exp(- 0.5 * m*w * (x**2) ) * hermite(indices, jnp.sqrt(m*w) * x)

    def sp_energy(indices, w):
        return w * (indices + 0.5)

    return sp_orbital, sp_energy

####################################################################################
## get orbitals indices
def get_orbitals_indices(w_indices, max_idx=10, num_orb=6):
    """
    Generates and sorts indices for 1D harmonic oscillator orbitals.

    Parameters:
        n (int): Principal quantum number.
        w_indices (float): Oscillator frequencies for each dimension.
        max_idx (int): Maximum index value (default is 5).
        num_orb (int): Number of orbitals to consider (default is 6).

    Returns:
        orb_idx (array): Sorted orbital indices.
        orb_state (array): Quantum number states corresponding to orbitals.
        orb_Es (array): Energies of the orbitals.
    """
    w_indices = np.array(w_indices, dtype=np.float64)
    n = w_indices.size
    max_idx = max_idx + 1
    
    def generate_indices(n):
        if n == 0:
            return [[]]
        prev = generate_indices(n - 1)
        return [rest + [i] for i in range(max_idx) for rest in prev]
    orb_state = jnp.array(generate_indices(n))
    orb_Es = ((orb_state+0.5) * w_indices).sum(axis=(-1))
    
    sort_idx = orb_Es.argsort()
    orb_state, orb_Es = orb_state[sort_idx], orb_Es[sort_idx]
    orb_state, orb_Es = orb_state[:num_orb], orb_Es[:num_orb]
    orb_index = jnp.arange(num_orb)
    return orb_index, orb_state, orb_Es

####################################################################################
## get orbitals indices from NCDO method
def get_orbitals_indices_first(nu, max_orb=1000, num_orb=6):
    """
        Finds first N elemens of a sum:
            const + nu[0]*n_0 + nu[1]*n_1 + ... + nu[d]*n_d, where n_i = 0, 1, 2, ...
    """

    nu = np.array(nu, dtype=np.float64)
    n = nu.size
    const = np.sum(nu)*0.5
    V = set([tuple(np.zeros(n))])
    ans = np.zeros((max_orb, n))
    values = np.zeros(max_orb)
    for iter_idx in range(1, max_orb):
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
    
    sort_idx = orb_Es.argsort()
    orb_state, orb_Es = orb_state[sort_idx], orb_Es[sort_idx]
    orb_state, orb_Es = orb_state[:num_orb], orb_Es[:num_orb]
    orb_index = jnp.arange(num_orb)
    return orb_index, orb_state, orb_Es

####################################################################################
## choose some of orbitals
def choose_orbitals(orb_state, orb_Es, choose_orb):
    choose_orb = jnp.array(choose_orb, dtype=jnp.int64)
    orb_state = orb_state[choose_orb]
    orb_Es    = orb_Es[choose_orb]
    orb_index = jnp.arange(len(choose_orb))
    return orb_index, orb_state, orb_Es

####################################################################################
## get orbitals energy
def get_orbitals_energy(orb_state, w_indices):
    orb_Es = ((orb_state+0.5) * w_indices).sum(axis=(-1))
    return orb_Es

####################################################################################
## array to string
def orbitals_array2str(arr, type="code"):
    ## type: code, latex
    if type == "code":
        result = ""
        for i, value in enumerate(arr):
            if value != 0:
                if result:
                    result += " + "
                if value == 1:
                    result += f"v{i+1}"
                if value != 1:
                    result += f"{value}v{i+1}"
        if result == "":
            result = "ZPE"
    
    elif type == "latex":
        result = ""
        for i, value in enumerate(arr):
            if value != 0:
                if result:
                    result += " + "
                if value == 1:
                    result += f"\\nu_{{{i+1}}}"
                if value != 1:
                    result += f"{value}\\nu_{{{i+1}}}"
        result = "$" + result + "$"
        if result == "$$":
            result = "ZPE"  
        
    return result


####################################################################################
# An example of how to generate orbitals
if __name__ == "__main__":
    n = 3
    wsquare_indices = jnp.array([0.49, 1.69, 1.0], dtype=jnp.float64)
    w_indices = jnp.sqrt(wsquare_indices)
    num_orb = 6
    orb_index, orb_state, orb_Es = get_orbitals_indices(n, w_indices, num_orb = num_orb)
    print("\nn =", n, ",  w^2 =", w_indices**2)
    print("Total number of orbitals:", num_orb)
    for ii in range(num_orb): print("    %d, E: %.3f, orbital:" %(ii, orb_Es[ii]), orb_state[ii])
        
    n = 2
    wsquare_indices = jnp.array([0.29375, 2.12581], dtype=jnp.float64)
    w_indices = jnp.sqrt(wsquare_indices)
    num_orb = 4
    orb_index, orb_state, orb_Es = get_orbitals_indices(n, w_indices, num_orb = num_orb)
    print("\nn =", n, ",  w^2 =", w_indices**2)
    print("Total number of orbitals:", num_orb)
    for ii in range(num_orb): print("    %d, E: %.3f, orbital:" %(ii, orb_Es[ii]), orb_state[ii])
    
    