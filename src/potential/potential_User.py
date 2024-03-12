"""The potential api for user specified files."""

import os 
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def get_w0(w0_file:str) -> np.ndarray:
    """Get w0 from user specified file.

    Args: 
        w0_file: the file storing w0 constants
    Returns:
        w0: the harmonic terms read from provided txt file.
    """
    with open(w0_file, "r") as f:
        datastr = f.read()
   
    lines = datastr.strip().split('\n')
    data_array = np.array([list(map(float, line.split()[2:])) for line in lines])
    w0 = np.array(data_array[:, 0], dtype=np.float64) * 2
    return w0


def get_k30(
    cubic_file:str,
    modes_num:int,
) -> np.ndarray:
    """Get k30 from user specified file
    
    Args:
        cubic_file: the file storing k30 constants
        modes_num: the total modes number in the molecule.
    Returns:
        k30: the cubic terms read from provided txt file.
    """
    with open(cubic_file, "r") as f:
        datastr = f.read()
    
    k30 = np.zeros((modes_num,modes_num,modes_num), dtype=np.float64)
    lines = datastr.strip().split('\n')
    for line in lines:
        i, j, k, value = map(float, line.split())
        indices = (int(i)-1, int(j)-1, int(k)-1)
        k30[indices] = value
    return k30


def get_k40(
    quartic_file:str,
    modes_num:int
):
    """Get k40 from user specified file
    
    Args:
        quartic_file: the file storing k30 constants
        modes_num: the total modes number in the molecule.
    Returns:
        k40: the quartic terms read from provided txt file.
    """
    with open(quartic_file, "r") as f:
        datastr = f.read()
       
    k40 = np.zeros((modes_num,modes_num,modes_num,modes_num), dtype=np.float64)
    lines = datastr.strip().split('\n')
    for line in lines:
        i, j, k, l, value = map(float, line.split())
        indices = (int(i)-1, int(j)-1, int(k)-1, int(l)-1)
        k40[indices] = value
    return k40

####################################################################################
def get_potential_energy_User(
        user_potential_file_dir:str,
        modes_num:int,
        alpha:int = 1000,
    ) -> [callable, jnp.ndarray]:
    """Get potential energy function for User specified potential.

    Args:
        user_potential_file_dir: the directory storing the user
            specified potential file, including w0.txt,
            cubic.txt and quartic.txt
        modes_num: the total modes number in the molecule.
        alpha: the int for scaling.

    Returns:
        potential_energy: a jit function for solving the 
            potential energy in training.
        w_indices: harmonic terms of the system.
    """
    print(f"Reading from user specified file under {user_potential_file_dir}")
    w0_file = os.path.join(
        user_potential_file_dir,
        "w0.txt",
    )
    cubic_file = os.path.join(
        user_potential_file_dir,
        "cubic.txt",
    )
    quartic_file = os.path.join(
        user_potential_file_dir,
        "quartic.txt",
    )
    
    w0 = get_w0(w0_file=w0_file)
    print("w0, harmonic constants:", np.count_nonzero(w0))
    w = w0 / alpha
    sqrtw = np.sqrt(w)
    
    k30 = get_k30(cubic_file=cubic_file,modes_num=modes_num)
    print("k30, non-zero terms:", np.count_nonzero(k30))
    k3 = jnp.einsum('ijk,i,j,k->ijk', k30, sqrtw, sqrtw, sqrtw) / alpha

    k40 = get_k40(quartic_file=quartic_file,modes_num=modes_num)
    print("k40, non-zero terms:", np.count_nonzero(k40))
    k4 = jnp.einsum('ijkl,i,j,k,l->ijkl', k40, sqrtw, sqrtw, sqrtw, sqrtw) / alpha

    #==== get potential energy ====
    @partial(jax.jit)
    @partial(jax.vmap, in_axes=0, out_axes=0)
    def potential_energy(x):
        q = x[:, 0]
        V =  (1/2) * jnp.einsum('i,i->', w**2, q**2) \
           + jnp.einsum('ijk,i,j,k->', k3, q, q, q) \
           + jnp.einsum('ijkl,i,j,k,l->', k4, q, q, q, q)
           
        return V

    return potential_energy, jnp.array(w)