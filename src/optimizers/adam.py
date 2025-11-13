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

    @staticmethod
    def _get_state_key(path):
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

        m = self.beta1 * m + (1-self.beta1) * grads
        v = self.beta2 * v + (1-self.beta2) * grads**2

        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)

        updated_params = params - self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

        self.state[key]['m'] = m
        self.state[key]['v'] = v

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
        