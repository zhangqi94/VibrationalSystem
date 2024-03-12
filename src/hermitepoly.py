import jax
import jax.numpy as jnp

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
