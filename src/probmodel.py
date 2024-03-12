import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk

## state probability table

class statelogitsNet(hk.Module):
    """
    Initialization Parameters:
        num_orb: Number of orbitals (states).
        orb_Es: Energies of the orbitals.
        beta: Inverse temperature parameter for the Boltzmann distribution.
        type: Type of distribution ("boltzmann" or "uniform").
            
    Input: 
        state_index [0, 1, 2, 3, 4, ....]
    Output: 
        state_logits [log(p0), log(p1), log(p2), log(p3), log(p4), ....]
    """
    def __init__(self, num_orb, orb_Es, beta, type):
        super().__init__()
        self.num_orb = num_orb
        self.orb_Es = orb_Es
        self.beta = beta
        self.type = type
        
    def __call__(self, state_index):
        # Initialized the state logits based on the distribution type.
        
        if self.type == "boltzmann":
            # For the Boltzmann distribution, calculate logits using energy values and temperature.
            state_logits = hk.get_parameter("state_logits", shape=(self.num_orb,), 
                        init = hk.initializers.Constant(-self.beta * (self.orb_Es - self.orb_Es[0])), 
                        dtype=jnp.float64)
        
        elif self.type == "uniform":
            # For the uniform distribution, initialize logits with zeros.
            state_logits = hk.get_parameter("state_logits", shape=(self.num_orb,), 
                        init = hk.initializers.Constant(0.0), 
                        dtype=jnp.float64)
        
        # Apply log_softmax to the logits to get the state probabilities.
        state_logits = jax.nn.log_softmax(state_logits)
        return state_logits[state_index]


####################################################################################
def make_probmodel(num_orb, orb_Es, beta=1.0, type="boltzmann"):
    
    def forward_fn(state_index):
        model = statelogitsNet(num_orb, orb_Es, beta, type)
        return model(state_index)
    
    probNet = hk.transform(forward_fn)
    return probNet


####################################################################################
# An example of how to use the statelogistNet
if __name__ == "__main__":
    from jax.flatten_util import ravel_pytree
    import orbitals
        
    n = 3
    w_indices = jnp.array([0.7, 1.3, 1.0])
    num_orb = 10
    beta = 1.0

    sp_orbitals, _ = orbitals.get_orbitals_1d()
    orb_index, orb_state, orb_Es = orbitals.get_orbitals_indices(n, w_indices, max_idx = 5, num_orb = num_orb)
    print("Total number of orbitals:", num_orb)
    for ii in range(num_orb): print("    %d, E: %.3f, orbital:" %(ii, orb_Es[ii]), orb_state[ii])
        
    print("\ninitialzed with boltzmann distribution")    
    probNet = make_probmodel(num_orb, orb_Es, beta, type="boltzmann")
    params_prob = probNet.init(jax.random.PRNGKey(42), orb_index)
    raveled_params_prob, _ = ravel_pytree(params_prob)
    print("    parameters in the prob model: %d" % raveled_params_prob.size)
    logits_init = probNet.apply(params_prob, None, orb_index)
    print("    init logp_state: ", jax.scipy.special.logsumexp(logits_init), logits_init)
    print("    init probability: ", jnp.exp(logits_init).sum(), jnp.exp(logits_init), flush=True)

    print("\ninitialzed with unifrom distribution")  
    probNet = make_probmodel(num_orb, orb_Es, beta, type="uniform")
    params_prob = probNet.init(jax.random.PRNGKey(42), orb_index)
    raveled_params_prob, _ = ravel_pytree(params_prob)
    print("    parameters in the prob model: %d" % raveled_params_prob.size)
    logits_init = probNet.apply(params_prob, None, orb_index)
    print("    init logp_state: ", jax.scipy.special.logsumexp(logits_init), logits_init)
    print("    init probability: ", jnp.exp(logits_init).sum(), jnp.exp(logits_init), flush=True)
    
    