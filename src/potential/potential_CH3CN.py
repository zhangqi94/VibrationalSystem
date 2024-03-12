import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from itertools import permutations

####################################################################################
def get_w0():
    # get frequency
    datastr = """
    1   1          3065
    2   2          2297
    3   3          1413
    4   4          920
    5   5          3149
    6   6          3149
    7   7          1487
    8   8          1487
    9   9          1061
    10  10         1061
    11  11         361
    12  12         361
    """
    
    lines = datastr.strip().split('\n')
    data_array = np.array([list(map(float, line.split()[2:])) for line in lines])
    w0 = np.array(data_array[:, 0], dtype=np.float64) 
    return w0


def get_k30():
    ### get cubic force constants k30
    datastr = """
    1  1  1    -176.0000
    4  4  4     -41.8167
    1  1  2     -10.5500
    4  5  5       4.3500
    4  6  6       4.3500
    1  2  2       2.2000
    4  5  9     -41.2000
    4  6  10     -41.2000
    1  3  3     -10.5000
    4  7  7       4.3500
    4  8  8       4.3500
    1  3  4     -19.3000
    4  7  9     -13.4000
    4  8  10     -13.4000
    1  5  5    -571.9000
    1  6  6    -571.9000
    4  9  9     -14.0500
    4  10  10     -14.0500
    1  5  7     -41.6000
    1  6  8     -41.6000
    4  9  11      13.4000
    4  10  12      13.4000
    1  5  9      65.1000
    1  6  10      65.1000
    4  11  11     -16.4000
    4  12  12     -16.4000
    1  5  11      17.8000
    1  6  12      17.8000
    5  5  5    -137.2833
    5  6  6     411.8499
    1  7  7     -24.4000
    1  8  8     -24.4000
    5  7  7       8.8000
    5  8  8      -8.8000
    6  7  8     -17.6000
    1  7  9      42.6000
    1  8  10      42.6000
    5  7  9     -16.4000
    5  8  10      16.4000
    6  7  10      16.4000
    6  8  9      16.4000
    1  7  11       9.7000
    1  8  12       9.7000
    5  9  9      10.5000
    5  10  10     -10.5000
    6  9  10     -21.0000
    1  9  9     -51.6500
    1  10  10     -51.6500
    5  9  11       9.0000
    5  10  12      -9.0000
    6  10  11      -9.0000
    6  9  12      -9.0000
    1  9  11     -15.9000
    1  10  12     -15.9000
    7  7  7      23.1167
    7  8  8     -69.3501
    2  2  2    -101.0000
    7  7  9     -20.2000
    8  8  9      20.2000
    7  8  10      40.4000
    2  2  3       8.5500
    7  9  9     -20.7000
    7  10  10      20.7000
    8  9  10      41.4000
    2  2  4     -96.4500
    7  9  11      -7.1000
    7  10  12       7.1000
    8  10  11       7.1000
    8  9  12       7.1000
    2  3  3      10.3500
    7  11  11      -3.8000
    7  12  12       3.8000
    8  11  12       7.6000
    2  3  4     -37.8000
    9  9  9     -12.4333
    9  10  10      37.2999
    2  4  4      49.6000
    9  9  11      -9.2500
    10  10  11       9.2500
    9  10  12      18.5000
    2  5  5     -12.3000
    2  6  6     -12.3000
    9  11  11      -2.8500
    9  12  12       2.8500
    10  11  12       5.7000
    2  5  9      11.0000
    2  6  10      11.0000
    2  7  7      -5.0500
    2  8  8      -5.0500
    2  9  11       7.8000
    2  10  12       7.8000
    2  11  11      -5.6000
    2  12  12      -5.6000
    3  3  3      20.0000
    3  3  4      21.3000
    3  4  4      39.6000
    3  5  5      -9.5500
    3  6  6      -9.5500
    3  5  9     -38.6000
    3  6  10     -38.6000
    3  5  11      -6.8000
    3  6  12      -6.8000
    3  7  7     -10.7500
    3  8  8     -10.7500
    3  9  11       9.4000
    3  10  12       9.4000
    3  11  11       5.9000
    3  12  12       5.9000
    """
    
    k30 = np.zeros((12,12,12), dtype=np.float64)
    lines = datastr.strip().split('\n')
    for line in lines:
        i, j, k, value = map(float, line.split())
        indices = (int(i)-1, int(j)-1, int(k)-1)
        k30[indices] = value
    return k30


def get_k40():
    ### get quartic force constants k40
    datastr = """
    1  4  5  5       3.7500
    1  4  6  6       3.7500
    3  5  7  11       7.6000
    3  5  8  12      -7.6000
    3  6  7  12      -7.6000
    3  6  8  11      -7.6000
    1  4  5  9     -10.4000
    1  4  6  10     -10.4000
    3  5  9  11     -11.6000
    3  5  10  12      11.6000
    3  6  9  12      11.6000
    3  6  10  11      11.6000
    1  4  7  7       7.4000
    1  4  8  8       7.4000
    3  7  7  7      -1.2000
    3  7  8  8       3.6000
    1  4  9  11     -11.8000
    1  4  10  12     -11.8000
    3  7  9  9      -6.4000
    3  7  10  10       6.4000
    3  8  9  10      12.8000
    1  5  5  5      44.73333333333333333
    1  5  6  6    -134.2000
    3  9  9  9       4.0000
    3  9  10  10     -12.0000
    1  5  7  7       6.1000
    1  5  8  8      -6.1000
    1  6  7  8     -12.2000
    4  4  4  4       1.1625
    1  5  7  11       8.6000
    1  5  8  12      -8.6000
    1  6  7  12      -8.6000
    1  6  8  11      -8.6000
    4  4  5  9      -4.5500
    4  4  6  10      -4.5500
    1  5  9  11     -16.2000
    1  5  10  12      16.2000
    1  6  9  12      16.2000
    1  6  10  11      16.2000
    4  4  5  11      -4.3000
    4  4  6  12      -4.3000
    1  7  7  11       3.5500
    1  8  8  11      -3.5500
    1  7  8  12      -7.1000
    4  5  7  7       6.2000
    4  5  8  8      -6.2000
    4  6  7  8     -12.4000
    1  9  9  9       2.3500
    1  9  10  10      -7.0500
    4  5  7  11      10.8000
    4  5  8  12     -10.8000
    4  6  7  12     -10.8000
    4  6  8  11     -10.8000
    2  2  2  2       4.5792
    4  5  9  11     -19.5000
    4  5  10  12      19.5000
    4  6  9  12      19.5000
    4  6  10  11      19.5000
    2  2  2  4       5.3167
    4  7  7  11       3.5000
    4  8  8  11      -3.5000
    4  7  8  12      -7.0000
    2  2  3  3      -2.0250
    4  9  9  9       1.9000
    4  9  10  10      -5.7000
    2  2  4  4       1.8000
    5  5  5  5      24.7750
    6  6  6  6      24.7750
    5  5  6  6      49.5500
    2  2  7  7      -3.1250
    2  2  8  8      -3.1250
    5  5  5  7       2.1333
    5  6  6  7       2.1333
    5  5  6  8       2.1333
    6  6  6  8       2.1333
    2  2  9  9      -2.7500
    2  2  10  10      -2.7500
    5  5  5  9      -3.2667
    5  5  6  10      -3.2667
    5  6  6  9      -3.2667
    6  6  6  10      -3.2667
    2  2  9  11       3.8000
    2  2  10  12       3.8000
    5  5  9  9      -2.8000
    5  5  10  10      -2.8000
    6  6  9  9      -2.8000
    6  6  10  10      -2.8000
    2  4  4  4      -1.2000
    5  7  7  7       2.4333
    5  7  8  8       2.4333
    6  7  7  8       2.4333
    6  8  8  8       2.4333
    2  4  5  9      -7.5000
    2  4  6  10      -7.5000
    5  7  7  9      -4.2500
    6  7  7  10      -4.2500
    5  8  8  9      -4.2500
    6  8  8  10      -4.2500
    2  4  7  7       4.5500
    2  4  8  8       4.5500
    5  7  9  9       3.9500
    6  8  9  9       3.9500
    5  7  10  10       3.9500
    6  8  10  10       3.9500
    2  4  9  9       4.3000
    2  4  10  10       4.3000
    5  7  9  11       8.1000
    6  8  10  12       8.1000
    5  7  10  12       8.1000
    6  8  9  11       8.1000
    1  1  1  1      12.4208
    2  5  9  11      -8.2000
    2  5  10  12       8.2000
    2  6  9  12       8.2000
    2  6  10  11       8.2000
    5  9  9  9       1.8333
    5  9  10  10       1.8333
    6  9  9  10       1.8333
    6  10  10  10       1.8333
    1  1  1  2       1.2167
    2  9  9  9       1.3500
    2  9  10  10      -4.0500
    5  9  9  11      -6.6500
    6  9  9  12      -6.6500
    5  10  10  11      -6.6500
    6  10  10  12      -6.6500
    1  1  1  3       1.8833
    3  3  3  3      -3.2417
    5  9  11  11      -4.1000
    5  9  12  12      -4.1000
    6  10  11  11      -4.1000
    6  10  12  12      -4.1000
    1  1  5  5      87.5500
    1  1  6  6      87.5500
    3  3  3  4       1.7000
    5  11  11  11      -1.3667
    5  11  12  12      -1.3667
    6  11  11  12      -1.3667
    6  12  12  12      -1.3667
    1  1  5  7       7.5000
    1  1  6  8       7.5000
    3  3  7  9       7.3000
    3  3  8  10       7.3000
    7  7  7  7       0.7708
    8  8  8  8       0.7708
    7  7  8  8       1.5416
    1  1  5  9      -5.9500
    1  1  6  10      -5.9500
    3  3  9  9      -4.4000
    3  3  10  10      -4.4000
    7  7  7  9      -2.1667
    7  8  8  9      -2.1667
    7  7  8  10      -2.1667
    8  8  8  10      -2.1667
    1  1  5  11      -3.5500
    1  1  6  12      -3.5500
    3  4  5  9      -8.8000
    3  4  6  10      -8.8000
    7  7  9  9       4.5000
    7  7  10  10       4.5000
    8  8  9  9       4.5000
    8  8  10  10       4.5000
    1  2  5  5       4.8500
    1  2  6  6       4.8500
    3  4  5  11      -7.3000
    3  4  6  12      -7.3000
    9  9  9  9      -0.9875
    10  10  10  10      -0.9875
    9  9  10  10      -1.9750
    1  3  3  3      -1.1667
    3  4  7  7       4.8500
    3  4  8  8       4.8500
    11  11  11  11       0.8042
    12  12  12  12       0.8042
    11  11  12  12       1.6084
    1  3  5  5       6.8000
    1  3  6  6       6.8000
    3  4  9  9       5.7000
    3  4  10  10       5.7000
    1  3  5  9      -7.8000
    1  3  6  10      -7.8000
    3  4  9  11      -7.7000
    3  4  10  12      -7.7000
    1  3  7  7       5.4000
    1  3  8  8       5.4000
    3  5  5  5       1.633333333333333333333333333
    3  5  6  6      -4.9000
    1  4  4  4       1.5000
    3  5  7  7       4.3000
    3  5  8  8      -4.3000
    3  6  7  8      -8.6000
    """
    
    k40 = np.zeros((12,12,12,12), dtype=np.float64)
    lines = datastr.strip().split('\n')
    for line in lines:
        i, j, k, l, value = map(float, line.split())
        indices = (int(i)-1, int(j)-1, int(k)-1, int(l)-1)
        k40[indices] = value
    return k40

####################################################################################
def get_potential_energy_CH3CN(alpha=1000):
    """
        Get potential energy function for CH3CN(acetonitrile).
        Ref: Using Nested Contractions and a Hierarchical Tensor Format To
            Compute Vibrational Spectra of Molecules with Seven Atoms
    """

    w0 = get_w0()
    print("w0, harmonic constants:", np.count_nonzero(w0))
    w = w0 / alpha
    sqrtw = np.sqrt(w)
    
    k30 = get_k30()
    print("k30, non-zero terms:", np.count_nonzero(k30))
    k3 = jnp.einsum('ijk,i,j,k->ijk', k30, sqrtw, sqrtw, sqrtw) / alpha

    k40 = get_k40()
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
