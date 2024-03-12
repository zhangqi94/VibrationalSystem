import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk


class RealNVP(hk.Module):
    """
        Real-valued non-volume preserving (real NVP) transform. 
        The implementation follows the paper "arXiv:1605.08803."
    """
    def __init__(self, maskflow, nvp_depth, mlp_width, mlp_depth, event_size):
        super().__init__()
        self.maskflow = maskflow
        self.nvp_depth = nvp_depth
        self.event_size = event_size
        
        # MLP (Multi-Layer Perceptron) layers for the real NVP.
        self.fc_mlp = [hk.nets.MLP([mlp_width]*mlp_depth, activation=jax.nn.tanh, activate_final=True)
                        for _ in range(nvp_depth)]
        
        # last linear layer initialized close to zero for the flow initialized near the identity function.
        self.fc_lin = [hk.Linear(event_size * 2,
                        w_init=hk.initializers.TruncatedNormal(stddev=0.0001), b_init=jnp.zeros)
                        for _ in range(nvp_depth)]
        self.zoom = hk.get_parameter("zoom", [event_size, ], init=jnp.ones, dtype=jnp.float64)

    ####################################################################################
    def coupling_forward(self, x1, x2, l):
        ## get shift and log(scale) from x1
        shift_and_logscale = self.fc_lin[l](self.fc_mlp[l](x1))
        shift, logscale = jnp.split(shift_and_logscale, 2, axis=-1)
        logscale = jnp.where(self.maskflow[l], 0, jnp.tanh(logscale)*self.zoom)

        ## transform: y2 = x2 * scale + shift
        y2 = x2 * jnp.exp(logscale) + shift
        
        ## calculate: logjacdet for each layer
        sum_logscale = jnp.sum(logscale)
        
        return y2, sum_logscale
    
    ####################################################################################   
    def __call__(self, x):
        #========== Real NVP (forward) ==========
        n, dim = x.shape  
        
        ## initial x and logjacdet
        x_flatten = jnp.reshape(x, (n*dim, ))
        logjacdet = 0
        
        for l in range(self.nvp_depth):
            ## split x into two parts: x1, x2
            x1 = jnp.where(self.maskflow[l], x_flatten, 0)
            x2 = jnp.where(self.maskflow[l], 0, x_flatten)
            
            ## get y2 from fc(x1), and calculate logjacdet = sum_l log(scale_l)
            y2, sum_logscale = self.coupling_forward(x1, x2, l)
            logjacdet += sum_logscale

            ## update: [x1, x2] -> [x1, y2]
            x_flatten = jnp.where(self.maskflow[l], x_flatten, y2)
            
        x = jnp.reshape(x_flatten, (n, dim))
        return x, logjacdet


####################################################################################
def get_maskflow(key, nvp_depth, event_size):
    
    mask = jnp.arange(0, jnp.prod(event_size)) % 2 == 0
    mask = (jnp.reshape(mask, event_size)).astype(bool)
    
    maskflow = []
    for _ in range(nvp_depth):
        mask_new = mask
        ## make sure mask_new is different from the old one
        while jnp.array_equal(mask_new, mask):
            key, subkey = jax.random.split(key)
            mask_new = jax.random.permutation(subkey, mask) 
        mask = mask_new
        maskflow += [mask]
    return maskflow

####################################################################################
def make_flow(key, nvp_depth, mlp_width, mlp_depth, event_size):
    
    maskflow = get_maskflow(key, nvp_depth, event_size)
    
    def forward_fn(x):
        model = RealNVP(maskflow, nvp_depth, mlp_width, mlp_depth, event_size)
        return model(x)
    
    flow = hk.transform(forward_fn)
    return flow


####################################################################################
# An example of how to use the flow
if __name__ == "__main__":
    import time
    
    key = jax.random.PRNGKey(42)
    nvp_depth = 4
    mlp_width = 16
    mlp_depth = 2
    n, dim = 6, 1
    
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    flow = make_flow(subkey1, nvp_depth, mlp_width, mlp_depth, n*dim)
    params = flow.init(subkey2, jnp.zeros((n, dim)))
    from jax.flatten_util import ravel_pytree
    raveled_params_flow, _ = ravel_pytree(params)
    print("parameters in the flow model: %d" % raveled_params_flow.size)

    # get logjacdet from: logjacdet = sum_l log(scale_l)
    t1 = time.time()    
    x = jax.random.normal(subkey3, (n, dim))
    y, logjacdet = flow.apply(params, None, x)
    t2 = time.time()
    print("x:", x.reshape(-1))
    print("y:", y.reshape(-1))
    print("logjacdet:", logjacdet, ",  time used:", t2-t1)

    # get logjacdet from: jacrev(y)(x)
    t1 = time.time()
    x = jax.random.normal(subkey3, (n, dim))
    x_flatten = x.reshape(-1)
    flow_flatten = lambda x: flow.apply(params, None, x.reshape(n, dim))[0].reshape(-1)
    jac = jax.jacrev(flow_flatten)(x_flatten)
    _, logjacdet = jnp.linalg.slogdet(jac)
    t2 = time.time()
    print("logjacdet:", logjacdet, ",  time used:",t2-t1)

