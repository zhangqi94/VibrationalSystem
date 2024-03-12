"""
    Second-order optimization algorithm using stochastic reconfiguration.
    The design of API signatures is in parallel with the package `optax`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import math
from jax.flatten_util import ravel_pytree
from optax._src import base

####################################################################################

def svd(A):
    """
        A manual implementation of SVD using the eigen-decomposition of A A^T or A^T A,
    which appears to be much more efficient than jax.scipy.linalg.svd.
    """
    M, N = A.shape
    if M < N:
        s2, U = jax.scipy.linalg.eigh(A.dot(A.T))
        s2, U = s2[::-1], U[:, ::-1]
        s = jnp.sqrt(jnp.abs(s2))
        Vh = (U/s).T.dot(A)
    else:
        s2, V = jax.scipy.linalg.eigh(A.T.dot(A))
        s2, V = s2[::-1], V[:, ::-1]
        s = jnp.sqrt(jnp.abs(s2))
        U = A.dot(V/s)
        Vh = V.T
    return U, s, Vh

def pad_score(ss, divide,):
    np = ss.shape[-1]
    upnp = math.ceil(np / divide) * divide
    np_pad = upnp - np
    if np_pad > 0:
        pad = jnp.zeros( [*ss.shape[:-1], np_pad], dtype=ss.dtype )
        ret = jnp.concatenate([ss, pad], axis=-1)
        return ret
    else:
        return ss

def qr_implicit_naive(A, vec, damping):
    # nb x np(np_loc)
    N, M = A.shape
    nd = jax.device_count()
    real_np = vec.shape[0]
    vec = pad_score(vec, nd)
    assert M == vec.shape[0]//nd, f"{M} != {vec.shape[0]//nd}"
    # nb x nb
    a = jax.lax.psum(A @ A.T, axis_name="p")
    a = a + damping * jnp.eye(N, dtype=A.dtype)
    # np(np_loc)  fetch the local part of vector
    vec = vec.reshape(nd,-1)[jax.lax.axis_index("p")]
    rhs = jax.lax.psum(A @ vec, axis_name="p")
    y = jax.scipy.linalg.solve(a, rhs, assume_a="pos")
    # np(np_loc)
    ret = 1./damping * (vec - A.T.dot(y))
    ret = jax.lax.all_gather(ret, "p", axis=0, tiled=True)
    ret = jnp.split(ret, [real_np], axis=0)[0]
    return ret

def qr_implicit_gatherd_psum(A, vec, damping):
    # nb x np(np_loc)
    N, M = A.shape
    nd = jax.device_count()
    real_np = vec.shape[0]
    vec = pad_score(vec, nd)
    assert M == vec.shape[0]//nd, f"{M} != {vec.shape[0]//nd}"
    # np(np_loc)  fetch the local part of vector
    vec = vec.reshape(nd,-1)[jax.lax.axis_index("p")]
    Atv = jnp.concatenate([A.T, vec.reshape(-1,1)], axis=-1)
    # nb x nb, nb x 1
    a, rhs = jnp.split(
      jax.lax.psum(A @ Atv, axis_name="p"), [N], axis=-1)
    a = a + damping * jnp.eye(N, dtype=A.dtype)
    y = jax.scipy.linalg.solve(a, rhs.reshape(-1), assume_a="pos")
    # np(np_loc)
    ret = 1./damping * (vec - A.T.dot(y))
    ret = jax.lax.all_gather(ret, "p", axis=0, tiled=True)
    ret = jnp.split(ret, [real_np], axis=0)[0]
    return ret

qr_implicit = qr_implicit_gatherd_psum

def _convert_score(score, factor):
    """
        Convert the quantum fisher information matrix of electrons as follows:
                            1/B Re(S^† S) = A^T A,
    where S is the (complex) score function of shape (*batchshape, nparams), and
    B = prod(batchshape) is the total batch size. The output A is a REAL matrix
    of shape (2B, nparams).

        Note this function should be placed in some pmapped subroutines.
    """
    *batchshape, nparams = score.shape
    c = 1. - jnp.sqrt(1. - factor)
    score_center = score - c * score.mean(axis=-2, keepdims=True)
    # nb_loc x np
    score_center = score_center.reshape(-1, nparams)
    
    ss = score_center
    nb_loc,_ = ss.shape
    nd = jax.device_count()
    # nb_loc x np, pad so the np can be divided by nd
    ss = pad_score(ss, nd)
    np_loc = ss.shape[-1] // nd
    ss = ss.reshape(nb_loc, nd, np_loc)
    # due to the bug https://github.com/google/jax/issues/18122
    # we have to do all_to_all for real and imag parts one by one.
    ss_real = jax.lax.all_to_all(ss.real, "p", 1, 0, tiled=True)
    ss_imag = jax.lax.all_to_all(ss.imag, "p", 1, 0, tiled=True)
    ss = ss_real + 1.J * ss_imag
    ss = ss.reshape(nb_loc*nd, np_loc)
    # nb x np_loc
    score_center = ss
    print("transposed score shape:", score_center.shape)

    assert score_center.shape[0] == jax.device_count() * np.prod(batchshape)
    score_center /= jnp.sqrt(score_center.shape[0])
    score_center = jnp.concatenate([score_center.real, score_center.imag], axis=0)
    # nb x np(np_loc)
    return score_center

####################################################################################
def hybrid_fisher_sr(class_score_fn, quant_score_fn, 
                     lr_c, lr_q, decay,
                     damping_c, damping_q, 
                     maxnorm_c, maxnorm_q,
                     acc_steps, sr_type="dense"):
    """
    Hybrid SR for both a classical probabilistic model and a set of quantum basis wavefunction ansatz.
    Params:
        classical_score_fn, quantum_score_fn, 
        lr_c, lr_q, (learning rate for classical and quantum model)
        decay, (decay of the learning rate)
        damping_c, damping_q, (damping for classical and quantum model)
        max_norm_c, max_norm_q, (maximum norm for classical and quantum model)
        acc_steps, (number of steps to accumulate the scores)
    """

    #========== initinal function ==========
    def init_fn(params_prob, params_flow, x):
        batchshape = x.shape[0]
        print("batchshape:", batchshape)
        raveled_params_prob, _ = ravel_pytree(params_prob)
        raveled_params_flow, _ = ravel_pytree(params_flow)
        prob_size = raveled_params_prob.size
        flow_size = raveled_params_flow.size
        
        return {
            "class_score": jnp.empty((acc_steps, batchshape, prob_size), dtype=jnp.float64),
            "quant_score": jnp.empty((acc_steps, batchshape, flow_size), dtype=jnp.float64),
            "acc_step": 0, 
            "step": 0, 
            "last_update": None,
            }

    #========== fisher function ==========
    def fishers_fn(params_prob, params_flow, state_indices, x, state):
        class_score = class_score_fn(params_prob, state_indices)
        class_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(class_score)

        quant_score = quant_score_fn(x, params_flow, state_indices)
        quant_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(quant_score)
        print("fisher_fn_class_score.shape:", class_score.shape)
        print("fisher_fn_quant_score.shape:", quant_score.shape)

        state["class_score"] = state["class_score"].at[state["acc_step"]].set(class_score)
        state["quant_score"] = state["quant_score"].at[state["acc_step"]].set(quant_score)
        state["acc_step"] = (state["acc_step"] + 1) % acc_steps
        return state
    
    #========== update function ==========
    def update_fn(grads, state):
        """
            NOTE: as the computation of (classical and quantum) Fisher information
        metrics calls for the Monte-Carlo sample `state_indices` and `x`, we manually
        place them within the `params` argument.
        """
        #========== get gradients ==========
        grad_params_prob, grad_params_flow = grads
        grad_params_prob_raveled, params_prob_unravel_fn = ravel_pytree(grad_params_prob)
        grad_params_flow_raveled, params_flow_unravel_fn = ravel_pytree(grad_params_flow)
        print("grad_params_prob.shape:", grad_params_prob_raveled.shape)
        print("grad_params_flow.shape:", grad_params_flow_raveled.shape)
        prob_size = grad_params_prob_raveled.shape[0]
        flow_size = grad_params_flow_raveled.shape[0]
        
        if sr_type == "dense":
            #========== calculate scores ==========
            class_score = state["class_score"]
            quant_score = state["quant_score"]
            print("update_fn class_score.shape:", class_score.shape)
            print("update_fn quant_score.shape:", quant_score.shape)
            batch_per_device = class_score.shape[1]
            class_fisher = jax.lax.pmean(
                            jnp.einsum('bij,bik->jk', class_score, class_score) / (batch_per_device), 
                            axis_name="p")
            quant_fisher = jax.lax.pmean(
                            jnp.einsum('bij,bik->jk', quant_score.conj(), quant_score) / (batch_per_device), 
                            axis_name="p")
            quant_score_mean = jax.lax.pmean(quant_score.mean(axis=(0, 1)), axis_name="p")
            print("update_fn quantum_score_mean.shape:", quant_score_mean.shape)
            quant_fisher = quant_fisher - jnp.einsum('j,k->jk', quant_score_mean.conj(), quant_score_mean)
            print("update_fn class_fisher.shape:", class_fisher.shape)
            print("update_fn quant_fisher.shape:", quant_fisher.shape)

            #========== classical update ==========
            class_fisher = class_fisher + damping_c * jnp.eye(prob_size)
            update_params_prob_raveled = jax.scipy.linalg.solve(class_fisher, grad_params_prob_raveled)
            lr_class = lr_c / (1 + decay*state["step"])
            gnorm = jnp.sum(grad_params_prob_raveled * update_params_prob_raveled)
            scale = jnp.minimum(jnp.sqrt(maxnorm_c/gnorm), lr_class)
            update_params_prob_raveled *= -scale
            update_params_prob = params_prob_unravel_fn(update_params_prob_raveled)

            #========== quantum update ==========
            quant_fisher = quant_fisher + damping_q * jnp.eye(flow_size)
            update_params_flow_raveled = jax.scipy.linalg.solve(quant_fisher, grad_params_flow_raveled)
            lr_quant = lr_q / (1 + decay*state["step"])
            gnorm = jnp.sum(grad_params_flow_raveled * update_params_flow_raveled)
            scale = jnp.minimum(jnp.sqrt(maxnorm_q/gnorm), lr_quant)
            update_params_flow_raveled *= -scale
            update_params_flow = params_flow_unravel_fn(update_params_flow_raveled)

            state["step"] += 1
            
        
        elif sr_type == "qr":
            #========== classical update: dense ==========
            class_score = state["class_score"]
            print("update_fn class_score.shape:", class_score.shape)
            batch_per_device = class_score.shape[1]
            class_fisher = jax.lax.pmean(
                            jnp.einsum('bij,bik->jk', class_score, class_score) / (batch_per_device), 
                            axis_name="p")
            class_fisher = class_fisher + damping_c * jnp.eye(prob_size)
            update_params_prob_raveled = jax.scipy.linalg.solve(class_fisher, grad_params_prob_raveled)
            print("update_fn params_prob.shape:", update_params_prob_raveled.shape)
            
            lr_class = lr_c / (1 + decay*state["step"])
            gnorm = jnp.sum(grad_params_prob_raveled * update_params_prob_raveled)
            scale = jnp.minimum(jnp.sqrt(maxnorm_c/gnorm), lr_class)
            update_params_prob_raveled *= -scale
            update_params_prob = params_prob_unravel_fn(update_params_prob_raveled)

            #========== quantum update: qr ==========
            quant_score = state["quant_score"]
            factor = 1. - 1. / (1 + decay*state["step"])
            A = _convert_score(quant_score, factor)
            print("update_fn quant_score.shape:", quant_score.shape)
            
            def quant_fisher(vec, damping_q):
                return qr_implicit(A, vec, damping_q)

            update_params_flow_raveled = quant_fisher(grad_params_flow_raveled, damping_q)
            print("update_fn params_flow.shape:", update_params_flow_raveled.shape)
            
            lr_quant = lr_q / (1 + decay*state["step"])
            gnorm = jnp.sum(grad_params_flow_raveled * update_params_flow_raveled)
            scale = jnp.minimum(jnp.sqrt(maxnorm_q/gnorm), lr_quant)
            update_params_flow_raveled *= -scale
            update_params_flow = params_flow_unravel_fn(update_params_flow_raveled)

            state["step"] += 1            
        
        return (update_params_prob, update_params_flow), state

    return fishers_fn, base.GradientTransformation(init_fn, update_fn)
