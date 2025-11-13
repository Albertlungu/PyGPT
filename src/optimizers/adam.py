import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Tuple, Union

class AdamNested:
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.state = {}

        # Create JIT-compiled Adam step function
        self._jit_adam_step = jax.jit(self._adam_step_fn)

    @staticmethod
    def _adam_step_fn(params, grads, m, v, t, beta1, beta2, lr, epsilon):
        """Pure JAX function for Adam update (can be JIT compiled)."""
        # Update biased first moment estimate
        m_new = beta1 * m + (1 - beta1) * grads

        # Update biased second moment estimate
        v_new = beta2 * v + (1 - beta2) * (grads ** 2)

        # Compute bias-corrected first moment
        m_hat = m_new / (1 - beta1 ** t)

        # Compute bias-corrected second moment
        v_hat = v_new / (1 - beta2 ** t)

        # Update parameters
        updated_params = params - lr * m_hat / (jnp.sqrt(v_hat) + epsilon)

        return updated_params, m_new, v_new

    def _get_state_key(self, path):
        return str(path)

    def _step_single(self, params, grads, path=()):
        key = self._get_state_key(path)

        if key not in self.state:
            self.state[key] = {
                'm': jnp.zeros_like(params),
                'v': jnp.zeros_like(params)
            }

        m = self.state[key]['m']
        v = self.state[key]['v']

        # Use JIT-compiled function
        updated_params, m_new, v_new = self._jit_adam_step(
            params, grads, m, v, self.t,
            self.beta1, self.beta2, self.lr, self.epsilon
        )

        self.state[key]['m'] = m_new
        self.state[key]['v'] = v_new

        return updated_params

    def step(self, params, grads, path=()):
        if len(path) == 0:
            self.t += 1

        if isinstance(params, dict):
            return {
                key: self.step(params[key], grads[key], path + (key,))
                for key in params.keys()
            }

        elif isinstance(params, (list, tuple)):
            updated = [
                self.step(p, g, path + (i,))
                for i, (p,g) in enumerate(zip(params, grads))
            ]
            return type(params)(updated)

        else:
            return self._step_single(params, grads, path)
        