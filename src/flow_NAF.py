import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk

####################################################################################
class NeuralAutoregressiveFlow(hk.Module):
    """
        Neural Autoregressive Flow (NAF) with deep dense sigmoid flow (DDSF).
        Reference: "arXiv:1804.00779"
    """
    def __init__(self, maskflow, naf_depth, mlp_width, mlp_depth, dsf_width, dsf_depth, event_size):
        
        super().__init__()
        self.maskflow = maskflow
        self.naf_depth = naf_depth
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.dsf_width = dsf_width
        self.dsf_depth = dsf_depth
        self.event_size = event_size

        #========== paramaters of neural autoregressive flow ========== 
        # MLP (Multi-Layer Perceptron) layers for the neural autoregressive flow.
        self.fc_mlp = [[hk.nets.MLP([mlp_width]*mlp_depth, activation=jax.nn.tanh, activate_final=True)
                          for _ in range(dsf_depth)] for _ in range(naf_depth)]
        
        # Linear layer initialized close to zero for the flow initialized near the identity function.
        self.fc_lin = [[hk.Linear((2*dsf_width + 2*dsf_width**2) * event_size,
                          w_init=hk.initializers.TruncatedNormal(stddev=0.001), b_init=jnp.zeros)
                          for _ in range(dsf_depth-2)] for _ in range(naf_depth)]
        
        self.fc_lin_first = [hk.Linear((3*dsf_width + dsf_width**2) * event_size,
                          w_init=hk.initializers.TruncatedNormal(stddev=0.001), b_init=jnp.zeros)
                          for _ in range(naf_depth)]
        
        self.fc_lin_last = [hk.Linear((3+dsf_width) * event_size,
                          w_init=hk.initializers.TruncatedNormal(stddev=0.001), b_init=jnp.zeros)
                          for _ in range(naf_depth)]

        #========== activate functions ==========  
        self.act_a = lambda x: jax.nn.softplus(x + 0.5413)
        self.act_b = lambda x: x
        self.act_w = lambda x: jax.nn.softmax(x, axis=1)
        self.act_u = lambda x: jax.nn.softmax(x, axis=1)
        self.sigmoid = lambda x: jax.nn.sigmoid(x)
        self.inv_sigmoid = lambda x: - jnp.log(1 / (x + 1e-50) - 1)
        self.ones = jnp.ones((event_size, ), dtype=jnp.float64)

    ####################################################################################
    #========== deep dense sigmoid flow ==========  
    def DSF_block(self, h0, d0, d1, params_abwu):
        event_size = self.event_size
        a, b, w, u, _ = jnp.split(params_abwu, [d1, 2*d1, 2*d1+d1**2, 2*d1+d1**2+d1*d0], axis=0)
        a = self.act_a(a).reshape(d1, event_size)
        b = self.act_b(b).reshape(d1, event_size)
        w = self.act_w(w.reshape(d1, d1, event_size))
        u = self.act_u(u.reshape(d1, d0, event_size))
        
        C = jnp.einsum("kn,kjn,jn->kn", a, u, h0) + b
        D = jnp.einsum("ijn,jn->in", w, self.sigmoid(C))
        h1 = self.inv_sigmoid(D)
        return h1

    def DenseSigmoidFlow(self, x_transform, params_abwus):
        h = x_transform.reshape(1, self.event_size)
        for jj in range(self.dsf_depth):
            d0, d1 = self.dsf_width, self.dsf_width
            if jj == 0: d0 = 1
            if jj == self.dsf_depth-1: d1 = 1
            h = self.DSF_block(h, d0, d1, 
                               params_abwus[jj].reshape(2*d1+d1**2+d1*d0, 
                               self.event_size))
        return h.reshape(-1)

    ####################################################################################
    #========== neural autoregressive flow (forward) ==========  
    def __call__(self, x):
        n, dim = x.shape
        
        ## initial x and logjacdet
        x = x.reshape(self.event_size,)
        logjacdet = 0
        
        for ii in range(self.naf_depth):
            # split x into two parts: x1, x2
            x1 = jnp.where(self.maskflow[ii], x, 0)
            x2 = jnp.where(self.maskflow[ii], 0, x)
            
            ## get paramaters for dense sigmoid flow
            params_abwus = []
            for jj in range(self.dsf_depth):
                if jj == 0: 
                    # first layer of dense sigmoid flow
                    params_abwus += [self.fc_lin_first[ii](self.fc_mlp[ii][jj](x1))]
                elif jj == self.dsf_depth-1:
                    # last layer of dense sigmoid flow
                    params_abwus += [self.fc_lin_last[ii](self.fc_mlp[ii][jj](x1))]
                else:
                    params_abwus += [self.fc_lin[ii][jj-1](self.fc_mlp[ii][jj-1](x1))]
                
            ## transform x2 -> y2 and calculate logjacdet
            fy2 = lambda x_transform: self.DenseSigmoidFlow(x_transform, params_abwus)
            y2, grad_y2 = jax.jvp(fy2, (x2, ), (self.ones, ))
            grad_y2 = jnp.where(self.maskflow[ii], 1, grad_y2)
            logjacdet += jnp.sum(jnp.log(grad_y2))
            
            ## update: [x1, x2] -> [x1, y2]
            x = jnp.where(self.maskflow[ii], x, y2)
        
        x = x.reshape(n, dim)
        return x, logjacdet

####################################################################################
def get_maskflow(key, naf_depth, event_size):
    """
        Generates mask matrix for neural autoregressive flow.
    """
    mask = jnp.arange(0, jnp.prod(event_size)) % 2 == 0
    mask = (jnp.reshape(mask, event_size)).astype(bool)
    
    maskflow = []
    for _ in range(naf_depth):
        mask_new = mask
        ## make sure mask_new is different from the old one
        while jnp.array_equal(mask_new, mask):
            key, subkey = jax.random.split(key)
            mask_new = jax.random.permutation(subkey, mask) 
        mask = mask_new
        maskflow += [mask]
    
    return maskflow

####################################################################################
def make_flow(key, naf_depth, mlp_width, mlp_depth, dsf_width, dsf_depth, event_size):
    """
        Make neural autoregressive flow.
        Input:
            key: random key
            naf_depth: layers of neural autoregressive flow
            mlp_width: width of multi-layer perceptron
            mlp_depth: depth of multi-layer perceptron
            dsf_width: width of dense sigmoid flow
            dsf_depth: depth of dense sigmoid flow
            event_size (=n*dim): size of event 
    """
    maskflow = get_maskflow(key, naf_depth, event_size)
    
    def forward_fn(x):
        model = NeuralAutoregressiveFlow(maskflow, naf_depth, mlp_width, mlp_depth, 
                                         dsf_width, dsf_depth, event_size)
        return model(x)
    
    flow = hk.transform(forward_fn)
    return flow

####################################################################################
# An example of how to use the flow
if __name__ == "__main__":
    import time
    
    key = jax.random.PRNGKey(42)
    naf_depth = 4
    mlp_width = 16
    mlp_depth = 2
    dsf_width = 4
    dsf_depth = 4
    n, dim = 6, 1
    
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    flow = make_flow(subkey1, naf_depth, mlp_width, mlp_depth, dsf_width, dsf_depth, n*dim)
    params = flow.init(subkey2, jnp.zeros((n, dim)))
    from jax.flatten_util import ravel_pytree
    raveled_params_flow, _ = ravel_pytree(params)
    print("parameters in the flow model: %d" % raveled_params_flow.size)

    # get logjacdet from: logjacdet = sum_l log(grad_yl)
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