'''
taken from https://github.com/deepmind/distrax/blob/master/examples/flow.py
'''
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
import distrax
import haiku as hk
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

####################################################################################
def make_conditioner(event_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int) -> hk.Sequential:
    """Creates an MLP conditioner for each layer of the flow."""
    num_bijector_params = 3 * 1 + 1 
    return hk.Sequential([
        hk.Flatten(preserve_dims=-len(event_shape)),
        hk.nets.MLP(hidden_sizes,
                    activation=jax.nn.tanh, activate_final=True),
        hk.Linear(np.prod(event_shape) * num_bijector_params,
            w_init=hk.initializers.TruncatedNormal(stddev=0.01),
            b_init=jnp.zeros),
        hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
        ])

####################################################################################
def make_chain(key, 
                event_shape: Sequence[int],
                num_layers: int,
                hidden_sizes: Sequence[int],
                num_bins: int) -> distrax.Transformed:

    """Creates the flow model."""
    def bijector_fn(params: jnp.ndarray):
        return distrax.RationalQuadraticSpline(params, range_min=0., range_max=1.)
    
    num_bijector_params = 3 * num_bins + 1
    
    layers = []
    for _ in range(num_layers):
        mask = jnp.arange(0, np.prod(event_shape)) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)
        key, subkey = jax.random.split(key)
        mask = jax.random.permutation(subkey, mask)

        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape, hidden_sizes, num_bijector_params)
            )
        layers.append(layer)
        
    flow = distrax.Chain(layers)
    base_distribution = distrax.Independent(
            distrax.MultivariateNormalDiag(
                loc=jnp.zeros(event_shape),
                scale_diag=jnp.ones(event_shape)),
            reinterpreted_batch_ndims=None)
    
    return distrax.Transformed(base_distribution, flow)

####################################################################################
def make_bijector(key, 
                  event_shape: Sequence[int],
                  num_layers: int,
                  hidden_sizes: Sequence[int],
                  num_bins: int
                  ):

    @hk.without_apply_rng
    @hk.transform
    def bijector(data: jnp.ndarray, sign) -> jnp.ndarray:
        model = make_chain(key, 
                           event_shape, 
                           num_layers, 
                           hidden_sizes, 
                           num_bins)
        
        return model.bijector.forward_and_log_det(data) \
            if sign==1 else model.bijector.inverse_and_log_det(data)
        
    return bijector
