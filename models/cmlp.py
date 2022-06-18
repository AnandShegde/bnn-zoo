from flax import linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    features : list
        
    @nn.compact
    def __call__(self, X):
        
        for i, feature in enumerate(self.features):
            X = nn.Dense(feature, kernel_init=jax.nn.initializers.glorot_normal(), name=f"Dense_{i}")(X)
            X = nn.relu(X)            
        X = nn.Dense(1, name=f"Dense_{i+1}")(X)
       
        X = nn.sigmoid(X)
        return X

    def loss_fn(self, params, X, y):
        y_hat = self.apply(params, X)
        @jax.jit
        def binary_cross_entropy(params,y_hat, y):
            bce = y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat)
            return jnp.mean(-bce)
        return jnp.mean(jax.vmap(binary_cross_entropy, in_axes=(None, 0, 0))(params,y_hat, y))
    
