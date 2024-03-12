import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

####################################################################################

shard = jax.pmap(lambda x: x)

def replicate(pytree, num_devices):
    dummy_input = jnp.empty(num_devices)
    return jax.pmap(lambda _: pytree)(dummy_input)


####################################################################################
## get radius distribution function (RDF)
## example: rmesh, gr = get_gr(x[:, :n//2], x[:, :n//2], L, nums=100)
def get_gr(x, y, L, nums=500): 
    batchsize, n, dim = x.shape[0], x.shape[1], x.shape[2]
        
    i,j = jnp.triu_indices(n, k=1)
    rij = (jnp.reshape(x, (-1, n, 1, dim)) - jnp.reshape(y, (-1, 1, n, dim)))[:,i,j]
    rij = rij - L*jnp.rint(rij/L)
    dij = np.linalg.norm(rij, axis=-1)  # shape: (batchsize, n*(n-1)/2)
    
    hist, bin_edges = np.histogram(dij.reshape(-1,), range=[0, L/2], bins=nums)
    
    dr = bin_edges[1] - bin_edges[0]
    hist = hist*2/(n * batchsize)

    rmesh = bin_edges[0:-1] + dr/2
    
    h_id = 4/3*np.pi*n/(L**3)* ((rmesh+dr)**3 - rmesh**3 )
    gr = hist/h_id
    return rmesh, gr

####################################################################################
## get title from file_path
## example: title_str = wrap_text(file_name, 50)
def wrap_text(text, width):
    return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])

####################################################################################
## get colors for plot
## example: colors = get_colors(colornums = 8)
def get_colors(colornums = 8):
    cmap = plt.get_cmap('jet')
    colors = [cmap(val) for val in np.linspace(0, 1, colornums)]
    colors = np.array(colors)[:, 0:3]
    return colors

####################################################################################
