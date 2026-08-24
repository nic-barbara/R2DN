import jax
import jax.numpy as jnp
import timeit

from robustnn.utils import count_num_params


def initialise_model(model, batches, horizon, seed=0):
    """Initialise params, states, and define input sequence."""
    # Sort out RNG keys
    rng = jax.random.key(seed)
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    
    # Create dummy input data
    states = model.initialize_carry(key1, (batches, model.input_size))
    states = jax.random.normal(key1, states.shape)
    inputs = jax.random.normal(key2, (horizon, batches, model.input_size))
    
    # Initialise the model and check parameter count
    params = model.init(key3, states, inputs[0])
    return params, states, inputs


def time_forwards(model, params, states, inputs, n_repeats):
    """Time the forwards pass of a model."""
    # Define a simple forwards pass for timing
    @jax.jit
    def forward(params, x0, u):
        x1, y = model.simulate_sequence(params, x0, u)
        return x1, y
    
    # Time compilation
    start = timeit.default_timer()
    jax.block_until_ready(forward(params, states, inputs))
    compile_time = timeit.default_timer() - start
    
    # Time evaluation
    eval_time = timeit.timeit(
        lambda: jax.block_until_ready(forward(params, states, inputs)),
        number=n_repeats
    )
    return compile_time, eval_time / n_repeats


def time_backwards(model, params, states, inputs, n_repeats):
    """Time the backwards pass of a model (computing grads)."""
    # Dummy loss function to backpropagate through
    @jax.jit
    def loss(params, x0, u):
        x1, y = model.simulate_sequence(params, x0, u)
        return jnp.mean(x1**2) + jnp.mean(y**2)
    
    grad_func = jax.jit(jax.grad(loss))
    
    def grad_test(params, x0, u):
        grads = grad_func(params, x0, u)
        jax.tree.map(lambda x: x.block_until_ready(), grads)
        return grads
    
    # Time compilation
    start = timeit.default_timer()
    grad_test(params, states, inputs)
    compile_time = timeit.default_timer() - start
    
    # Time evaluation
    eval_time = timeit.timeit(
        lambda: grad_test(params, states, inputs),
        number=n_repeats
    )
    return compile_time, eval_time / n_repeats


def time_model(model, batches, horizon, n_repeats):
    """Time forwards and backwards passes, print and store results."""
    
    # Initialise the model params and count them
    params, states, inputs = initialise_model(model, batches, horizon)
    num_params = count_num_params(params)
    
    # Time the forwards pass
    cf_time, rf_time = time_forwards(model, params, states, inputs, n_repeats)
    print(f"Forwards compile time: {cf_time:.6f} seconds")
    print(f"Forwards eval time   : {rf_time:.6f} seconds")
    
    # Time the backwards pass
    cb_time, rb_time = time_backwards(model, params, states, inputs, n_repeats)
    print(f"Backwards compile time: {cb_time:.6f} seconds")
    print(f"Backwards eval time   : {rb_time:.6f} seconds")

    return {
        "nv": model.features,
        "batches": batches,
        "horizon": horizon,
        "num_params": num_params,
        "forwards_compile": cf_time,
        "forwards_eval": rf_time,
        "backwards_compile": cb_time,
        "backwards_eval": rb_time,
    }
