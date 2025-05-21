import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

class RADState(NamedTuple):
    step: jnp.ndarray
    exp_avg: optax.Params  # same structure as params
    exp_avg_sq: optax.Params
    mu_product: optax.Params  # if needed
    max_exp_avg_sq: optax.Params  # if amsgrad=True
    # add more buffers as needed (zeta, delta, etc.)

def rad(
    lr=1e-3,
    betas=(0.9, 0.999),
    delta=1.0,
    order=1,
    weight_decay=0.0,
    amsgrad=False,
    nesterov=False,
    # ... any other hyperparams ...
):
    """Builds a RAD optimizer in JAX/Optax."""

    def init_fn(params):
        zeros_like = jax.tree_map(jnp.zeros_like, params)
        state = RADState(
            step=jnp.zeros([], jnp.int32),
            exp_avg=zeros_like,
            exp_avg_sq=zeros_like,
            mu_product=zeros_like if nesterov else None,
            max_exp_avg_sq=zeros_like if amsgrad else None,
        )
        return state

    def update_fn(grads, state, params):
        """Core RAD update logic."""
        step = state.step + 1

        # Decay factors
        beta1, beta2 = betas
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        # Weight decay
        def apply_weight_decay(p, g):
            return g if weight_decay == 0 else g + weight_decay * p

        grads = jax.tree_map(apply_weight_decay, params, grads)

        # Exponential moving averages
        exp_avg = jax.tree_map(
            lambda ema, g: beta1 * ema + (1.0 - beta1) * g, state.exp_avg, grads
        )
        exp_avg_sq = jax.tree_map(
            lambda ema_sq, g: beta2 * ema_sq + (1.0 - beta2) * (g * g),
            state.exp_avg_sq,
            grads,
        )

        # AMSGrad
        if amsgrad:
            max_exp_avg_sq = jax.tree_map(
                lambda old, new: jnp.maximum(old, new), state.max_exp_avg_sq, exp_avg_sq
            )
            denom_sq = jax.tree_map(
                lambda ms: ms, max_exp_avg_sq
            )
        else:
            max_exp_avg_sq = state.max_exp_avg_sq  # unchanged
            denom_sq = exp_avg_sq

        # The core RAD's 'relativistic' denominator part:
        # denom = 1 / sqrt(denom_sq * (delta^2) * 4 + 4 * zeta) etc.
        # You can adapt your code or rewrite it here. For simplicity:
        def rad_denom(e_sq):
            base = jnp.sqrt(e_sq + 1e-8)
            return base  # adapt the RAD formula here

        denom = jax.tree_map(rad_denom, denom_sq)

        # Combine with bias corrections and LR
        def param_update(p, ema, d):
            step_size = (lr / bias_correction1) * (ema / (d * jnp.sqrt(bias_correction2)))
            return p - step_size

        updates = jax.tree_map(param_update, params, exp_avg, denom)

        # Nesterov, delta annealing, etc., can be inserted similarly.

        new_state = RADState(
            step=step,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            mu_product=state.mu_product,  # or updated if nesterov is used
            max_exp_avg_sq=max_exp_avg_sq,
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# Example usage:
# opt_init, opt_update = rad(lr=1e-3, betas=(0.9,0.999), delta=1, ...)
# state = opt_init(params)
# grads = ...
# updates, state = opt_update(grads, state, params)
# new_params = optax.apply_updates(params, updates)
