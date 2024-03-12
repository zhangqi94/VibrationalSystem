import jax
import jax.numpy as jnp
from functools import partial
from .phibase import logphi_base

####################################################################################
def make_logpsi(flow, sp_orbitals, orb_state, w_indices):
    """
        Computes the logrithm of the wavefunction:
            log_psi = log_phi + 0.5*log_jacdet.
    """
    def logpsi(x, params, state_indices):
        ## calculate logphi & logjacdet
        z, logjacdet = flow.apply(params, None, x)
        log_phi = logphi_base(sp_orbitals, orb_state[state_indices], w_indices, z)
        return jnp.stack([log_phi.real + 0.5*logjacdet, log_phi.imag])
    
    return logpsi

####################################################################################
def make_logphi_logjacdet(flow, sp_orbitals, orb_state, w_indices):
    """              
        The same functionality as `make_logpsi`, but the two terms involving the base
    wavefunction and the jacobian determinant are separated.
    """
    def logphi(x, params, state_indices):   
        z, _ = flow.apply(params, None, x)
        log_phi = logphi_base(sp_orbitals, orb_state[state_indices], w_indices, z)
        return jnp.stack([log_phi.real, log_phi.imag])

    def logjacdet(x, params):
        _, logjacdet = flow.apply(params, None, x)
        return 0.5*logjacdet

    return logphi, logjacdet

####################################################################################
def make_logpsi_grad_laplacian(logpsi, forloop=True, hutchinson=False,
                               logphi=None, logjacdet=None):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def logpsi_vmapped(x, params, state_indices):
        logpsix = logpsi(x, params, state_indices)
        return logpsix[0] + 1j * logpsix[1]

    ##==== grad_logpsi and laplacian_logpsi ====
    @partial(jax.vmap, in_axes=(0, None, 0, None), out_axes=0)
    def logpsi_grad_laplacian(x, params, state_indices, key):
        """
            Computes the gradient and laplacian of logpsi w.r.t. electron coordinates x.
        The final result is in complex form.
            INPUT:  x: (batch, n, dim)  state_indices: (batch, n)
            OUTPUT: grad: (batch, n, dim)   laplacian: (batch,)
        """

        grad = jax.jacrev(logpsi)(x, params, state_indices)
        grad = grad[0] + 1j * grad[1]
        print("Computed gradient...")

        n, dim = x.shape
        x_flatten = x.reshape(-1)
        grad_logpsi = jax.jacrev(lambda x: logpsi(x.reshape(n, dim), params, state_indices))

        def _laplacian(x):
            if forloop:
                print("foriloop version...")
                def body_fun(i, val):
                    _, tangent = jax.jvp(grad_logpsi, (x,), (eye[i],))
                    return val + tangent[0, i] + 1j * tangent[1, i]
                eye = jnp.eye(x.shape[0])
                laplacian = jax.lax.fori_loop(0, x.shape[0], body_fun, 0.+0.j)
            else:
                print("vmap version...")
                def body_fun(x, basevec):
                    _, tangent = jax.jvp(grad_logpsi, (x,), (basevec,))
                    return (tangent * basevec).sum(axis=-1)
                eye = jnp.eye(x.shape[0])
                laplacian = jax.vmap(body_fun, (None, 1), 1)(x, eye).sum(axis=-1)
                laplacian = laplacian[0] + 1j * laplacian[1]
            return laplacian

        laplacian = _laplacian(x_flatten)
        print("Computed laplacian...")

        return grad, laplacian

    ##==== hutchinson: grad_logpsi and laplacian_logpsi ====
    def logpsi_grad_laplacian_hutchinson(x, params, state_indices, key):

        v = jax.random.normal(key, x.shape)

        @partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=0)
        def logpsi_grad_random_laplacian(x, params, state_indices, v):
            """
                Compute the laplacian as a random variable `v^T hessian(ln Psi_n(x)) v`
            using the Hutchinson's trick.

                The argument `v` is a random "vector" that has the same shape as `x`,
            i.e., (after vmapped) (batch, n, dim).
            """

            grad, hvp = jax.jvp( jax.jacrev(lambda x: logpsi(x, params, state_indices)),
                                 (x,), (v,) )

            grad = grad[0] + 1j * grad[1]
            print("Computed gradient...")

            random_laplacian = (hvp * v).sum(axis=(-2, -1))
            random_laplacian = random_laplacian[0] + 1j * random_laplacian[1]
            print("Computed Hutchinson's estimator of laplacian.")

            return grad, random_laplacian

        @partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=0)
        def logpsi_grad_random_logjacdet(x, params, state_indices, v):
            grad_logphi = jax.jacrev(logphi)(x, params, state_indices)
            grad_logphi = grad_logphi[0] + 1j * grad_logphi[1]
            grad_logjacdet, hvp = jax.jvp( jax.grad(lambda x: logjacdet(x, params)),
                                 (x,), (v,) )
            grad = grad_logphi + grad_logjacdet
            print("Computed gradient...")

            n, dim = x.shape
            x_flatten = x.reshape(-1)
            grad_logphi = jax.jacrev(lambda x: logphi(x.reshape(n, dim), params, state_indices))

            def _laplacian(x):
                if forloop:
                    print("foriloop version...")
                    def body_fun(i, val):
                        _, tangent = jax.jvp(grad_logphi, (x,), (eye[i],))
                        return val + tangent[0, i] + 1j * tangent[1, i]
                    eye = jnp.eye(x.shape[0])
                    laplacian = jax.lax.fori_loop(0, x.shape[0], body_fun, 0.+0.j)
                return laplacian

            laplacian_logphi = _laplacian(x_flatten)
            print("Computed exact laplacian of logphi.")

            random_logjacdet = (hvp * v).sum(axis=(-2, -1))
            print("Computed Hutchinson's estimator of logjacdet.")
            laplacian = laplacian_logphi + random_logjacdet

            return grad, laplacian

        logpsi_grad_laplacian = logpsi_grad_random_laplacian \
                                if (logphi is None and logjacdet is None) else \
                                logpsi_grad_random_logjacdet
        return logpsi_grad_laplacian(x, params, state_indices, v)

    return logpsi_vmapped, \
           (logpsi_grad_laplacian_hutchinson if hutchinson else logpsi_grad_laplacian)

####################################################################################
def make_logp(logpsi):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def logp(x, params, state_indices):
        """ logp = logpsi + logpsi* = 2 Re logpsi """
        return 2 * logpsi(x, params, state_indices)[0]

    return logp

####################################################################################
def make_quantum_score(logpsi):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def quantum_score_fn(x, params, state_indices):
        grad_params = jax.jacrev(logpsi, argnums=1)(x, params, state_indices)
        return jax.tree_map(lambda jac: jac[0] + 1j * jac[1], grad_params)

    return quantum_score_fn

####################################################################################
def make_quantum_force(logpsi):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def quantum_force(x, params, state_indices):
        logpsi2 = lambda x, params, state_indices: 2 * logpsi(x, params, state_indices)[0]
        return jax.grad(logpsi2)(x, params, state_indices)

    return quantum_force

