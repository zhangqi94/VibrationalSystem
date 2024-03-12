import jax
import jax.numpy as jnp

####################################################################################
def make_sampler_logprob(probNet, orb_index):

    def sampler(params_prob, key, batch):
        logits = probNet.apply(params_prob, None, orb_index)
        state_indices = jax.random.categorical(key, logits=logits, shape=(batch,))
        return state_indices

    def log_prob_novmap(params_prob, state_index):
        logp = probNet.apply(params_prob, None, state_index)
        return logp
    
    return sampler, log_prob_novmap

make_classical_score = lambda log_prob: jax.vmap(jax.grad(log_prob), (None, 0), 0)
    
####################################################################################
## get state indices with equal probability
# def sampler_equal(key, batch, orb_state):
#     nob = orb_state.shape[0]
#     state_ids = jax.random.randint(key, (batch, ), minval=0, maxval=nob)
#     state_indices = jnp.array(orb_state[state_ids])
#     return state_indices, state_ids

####################################################################################
shard = jax.pmap(lambda x: x)

def init_state_indices(
        orb_index:jnp.ndarray,
        num_orb:int,
        num_devices:int,
        weight_in_sampling:str,
        batch_per_device:int,
        num_ground_total:int = None,
    ) -> [jnp.ndarray, int]:
    """Initialize the state indices to be used in sampling.

    Args:
        orb_index: the orbitals index as if they are in Equal mode, aka,
            each state appears only once in the array, and in ascending order.
        num_orb: the total number of orbits setted in the arguments. 
            NOTE: after changing the weight_in_sampling to non-equal, the
            actual number of orbitals in the following codes would change!
        num_devices: the number of gpus used.
        weight_in_sampling: the weight mode used in sampling. See argparser's help.
        batch_per_device: after batched to parallel devices, the batch number on 
            each device.
        num_ground_total: The TOTAL number of ground states that would be calculated
                            in sampling. Only needed when weight_in_sampling==Manual
    Returns:
        state_indices: the state_indices corresponding to weight_in_sampling.
        real_num_orb_in_state_indices: the actual number of orbitals after 
            suiting to weight_per_device, as stated above.
    """
    def _add_zeros(
            ground_replicate_number:int, 
            orb_index:jnp.ndarray,
            num_devices:int,
            batch_per_device:int,) -> jnp.ndarray:
        """add zeros to the front of the orb_index
        as a way to adjusting the weight of ground states(the appearing times)
        in real orbital index.
        
        Args: 
            ground_replicate_number: the additional time that ground state would 
                be replicated.
            orb_index: the original orb_index to which additional ground states would 
                be added.
            num_devices: the number of gpus used.
            batch_per_device: after batched to parallel devices, the batch number on 
                each device.
        Returns: 
            state_indices: the state_indices corresponding to weight_in_sampling.
        """
        ground_to_append = jnp.zeros(ground_replicate_number,dtype=jnp.int32)
        orb_index_ground_half = jnp.concatenate((ground_to_append,orb_index))
        state_indices = jnp.tile(orb_index_ground_half, (num_devices, batch_per_device))
        state_indices = shard(state_indices)
        return state_indices

    print(f"\n weight_in_sampling chosen to be {weight_in_sampling}\n")
    print("initialize sampler...", flush=True)
    if weight_in_sampling == "Equal":
        state_indices = jnp.tile(orb_index, (num_devices, batch_per_device))
        state_indices = shard(state_indices)
        real_num_orb_in_state_indices = num_orb
        ground_replicate_number = 0
    elif weight_in_sampling == "Ground-half":
        ground_replicate_number = num_orb - 2
        state_indices = _add_zeros(
            ground_replicate_number=ground_replicate_number,
            orb_index=orb_index,
            num_devices=num_devices,
            batch_per_device=batch_per_device,
        )
    elif weight_in_sampling == "Manual":
        # manually set the number of ground states in real orbital index
        if num_ground_total == None or num_ground_total <= 0:
            raise ValueError(f"While set weight_in_sampling==manual, the total number"
                             f"of ground states must be specified as an interger larger than 0."
                             f"get num_ground_total={num_ground_total}")
        ground_replicate_number = num_ground_total - 1
        state_indices = _add_zeros(
            ground_replicate_number=ground_replicate_number,
            orb_index=orb_index,
            num_devices=num_devices,
            batch_per_device=batch_per_device,
        )
    else:
        # Not defined in argparser
        raise ValueError(f"Undefined argument for weight_in_sampling: {weight_in_sampling}")
    real_num_orb_in_state_indices = ground_replicate_number + num_orb
    print("state_indices.shape:", state_indices.shape)
    print(f"real_num_orb_in_state_indices={real_num_orb_in_state_indices}")

    return state_indices, real_num_orb_in_state_indices