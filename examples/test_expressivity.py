import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

from robustnn import ren
from robustnn import r2dn
from robustnn.utils import count_num_params

from utils.plot_utils import startup_plotting
from utils import sysid
from utils import utils

import optax

startup_plotting()
dirpath = Path(__file__).resolve().parent
jax.config.update("jax_default_matmul_precision", "highest")

# Training hyperparameters
config = {
    "experiment": "expressivity",
    "network": "contracting_ren",
    "epochs": 3*500,
    "batches": 128,
    "batchsize": 512,
    "clip_grad": 10,
    "schedule": {
        "init_value": 1e-3,
        "decay_steps": 500,
        "decay_rate": 0.1,
        "end_value": 1e-5,
    },
    "nx": 1,
    "polar": True,
    "init_method": "random",
    "seed": 0,
} 


def generate_data(rng, nx=1, batches=128, batchsize=512, uval=None):
    
    def dynamics(x, u):
        bw = x + u
        return (
            0.2 * jnp.sin(x) + 
            0.05 * jnp.cos(2*bw) + 
            0.05 * jnp.sin(3*bw) + 
            0.075 * jnp.sin(4*bw) * jnp.atan(0.1*bw**2)
        ) + 0.05*x + u
    
    x0_list = []
    x1_list = []
    for _ in range(batches):
        rng1, rng2 = jax.random.split(rng)
        x0 = jax.random.uniform(rng1, (batchsize, nx), minval=-30, maxval=30)
        if uval is None:
            u0 = jax.random.uniform(rng2, (batchsize, nx), minval=-1, maxval=1)
        else:
            u0 = uval * jnp.ones(x0.shape)
        x1 = dynamics(x0, u0)
        x0_list.append((x0, u0))
        x1_list.append(x1)
        
    return x0_list, x1_list


def build_model(config):
    """Build neural models."""
    nu, nx, ny = config["nx"], config["nx"], config["nx"]
    if config["network"] == "contracting_ren":
        model = ren.ContractingREN(
            nu, nx, config["nv"], ny, 
            identity_output=True,
            activation=utils.get_activation(config["activation"]),
            init_method=config["init_method"],
            do_polar_param=config["polar"]
        )
    elif config["network"] == "contracting_r2dn":
        model = r2dn.ContractingR2DN(
            nu, nx, config["nv"], ny, config["nh"], 
            identity_output=True,
            activation=utils.get_activation(config["activation"]),
            init_method=config["init_method"],
            do_polar_param=config["polar"]
        )
    return model


def train_expressivity(config, verbose=True):
    
    # Random seeds
    test_rng = jax.random.key(42)
    rng = jax.random.key(config["seed"])
    rng, key1 = jax.random.split(rng)
    
    # Initialise model
    model = build_model(config)
    
    # Loss function and training step
    def loss_fn(params, x, u, xn):
        x_pred, _ = model.apply(params, x, u)
        return jnp.mean((xn - x_pred)**2)
    
    @jax.jit
    def train_step(params, opt_state, x, u, xn):
        grad_loss = jax.jit(jax.value_and_grad(loss_fn))
        loss_value, grads = grad_loss(params, x, u, xn)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # Initialise REN and optimizer
    optimizer = sysid.setup_optimizer(config, config["batches"])
    init_x = jnp.zeros((config["batchsize"], model.state_size))
    init_u = jnp.zeros(init_x.shape)
    params = model.init(key1, init_x, init_u)
    opt_state = optimizer.init(params)
    
    # Loop for training
    train_loss = []
    for epoch in range(config["epochs"]):
        
        # Training data
        rng, _  = jax.random.split(rng)
        train = generate_data(
            rng, config["nx"], batches=config["batches"], batchsize=config["batchsize"]
        )
        
        # Training update
        batch_loss = []
        for xu, xn in zip(*train):
            x, u = xu
            params, opt_state, loss_value = train_step(
                params, opt_state, x, u, xn
            )
            batch_loss.append(loss_value)
            
        # Logging
        train_loss.append(jnp.mean(jnp.array(batch_loss)))
        lr = opt_state[1].hyperparams['learning_rate']
        if verbose:
            print(f"Epoch: {epoch+1}/{config['epochs']}, " +
                  f"Loss: {train_loss[-1]:.3g}, " +
                  f"lr: {lr:.3g}")
    
    # Get validation loss
    val = generate_data(test_rng, config["nx"], uval=1, batches=1, batchsize=2048)
    x0_val, u_val = val[0][0]
    x1_val = val[1][0]
    xh_val, _ = model.apply(params, x0_val, u_val)
        
    val_mse = jnp.mean((x1_val - xh_val)**2)
    val_nrmse = jnp.sqrt(val_mse / jnp.mean(x1_val**2))
    
    # Store results
    results = {
        "train_loss": jnp.array(train_loss),
        "val_mse": val_mse,
        "val_nrmse": val_nrmse,
        "x0_val": x0_val,
        "u_val": u_val,
        "x1_val": x1_val,
        "xh_val": xh_val
    }
    
    # Save results for later evaluation
    results["num_params"] = count_num_params(params)
    utils.save_results(config, params, results)
    return params, results


def train_and_test(config, verbose=True):
    
    # Train the model
    train_expressivity(config, verbose)
    
    # Load and test it
    config, _, results = utils.load_results_from_config(config)
    _, fname = utils.generate_fname(config)
    
    print("Number of params: ", results["num_params"])    
    print("MSE:   ", results["val_mse"])
    print("NRMSE: ", results["val_nrmse"])
    
    # Plot the loss curve
    plt.figure()
    plt.plot(results["train_loss"])
    plt.xlabel("Training epochs")
    plt.ylabel("Training loss")
    plt.yscale('log')
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_loss.pdf")
    plt.close()
    
    # Plot in phase space for a batch
    indx = jnp.argsort(results["x0_val"], axis=0)
    x = results["x0_val"][indx][..., 0]
    y_true = results["x1_val"][indx][..., 0]
    y_pred = results["xh_val"][indx][..., 0]
    
    plt.figure()
    plt.plot(x, y_true, label="Model")
    plt.plot(x, y_pred, label="Prediction")
    plt.xlabel("x")
    plt.ylabel("xnext")
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_phasespace.pdf")
    plt.close()

# Train for many random seeds
for s in range(5):
    
    config["seed"] = s

    # Run for a bunch of S-RENs
    r2dn_config = deepcopy(config)
    r2dn_config["network"] = "contracting_r2dn"
    r2dn_config["activation"] = "relu"
    layers = 6      
    nv_r2dn = 16
    for nh in [8, 16, 24, 32, 48, 64, 80, 96]:
        r2dn_config["layers"] = layers
        r2dn_config["nv"] = nv_r2dn
        r2dn_config["nh"] = (nh,) * layers
        print(f"R2DN {nh=} {s=}")
        train_and_test(r2dn_config)

    # Run for a bunch of RENs
    ren_config = deepcopy(config)
    ren_config["activation"] = "tanh"
    for nv in [20, 30, 40, 60, 80, 100, 128, 200]:
        ren_config["nv"] = nv
        print(f"REN {nv=} {s=}")
        train_and_test(ren_config)
