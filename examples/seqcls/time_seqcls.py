"""Stage 0: check that the models train at all, and time them at full size.

The REN equilibrium layer is sequential in `nv` and sits inside a scan over the
sequence, so wall-clock (not memory or FLOPs) decides what is affordable here.
Measure it before launching anything long.
"""
import jax
import jax.numpy as jnp
import optax
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from robustnn.utils import count_num_params

from utils import seqcls
from utils.seqcls import TASKS, matched_r2dn_config

ren_config = seqcls.default_config()

jax.config.update("jax_default_matmul_precision", "highest")


def smoke_test(network, steps=300, batchsize=64):
    """Overfit a single batch of psMNIST. The loss must go to (almost) zero."""
    config = deepcopy(ren_config)
    config.update({
        "nx": 32, "nv": 32, "epochs": steps, "lr_cuts": (),
        "batchsize": batchsize, "eval_batchsize": batchsize,
    })
    if network == "r2dn":
        config = matched_r2dn_config(config, TASKS["psmnist"])

    u, y = seqcls.load_task("psmnist")[0]
    batch = (u[:batchsize], y[:batchsize])

    model = seqcls.build_model(config, TASKS["psmnist"])
    _, results = seqcls.train(model, config, batch, batch, batch, verbose=False)
    print(f"  {network}: loss {results['train_loss'][0]:.3f} -> "
          f"{results['train_loss'][-1]:.2e}, "
          f"accuracy {100*results['final_test_acc']:.1f}%")


def time_gradient_step(config, task, batchsize, seq_len, steps=5):
    """Time compilation and a single gradient step at full size."""
    nu = TASKS[task]
    model = seqcls.build_model(config, nu)
    key1, key2 = jax.random.split(jax.random.key(0))

    u = jnp.zeros((seq_len, batchsize, nu))
    y = jnp.zeros((batchsize,), dtype=jnp.int32)
    x0 = model.initialize_carry(key1, (batchsize, nu))
    params = model.init(key2, x0, u[0])

    def loss_fn(params, x0, u, y):
        _, logits = model.simulate_sequence(params, x0, u)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits[-1], y)
        )

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    t0 = time.time()
    _, grads = grad_fn(params, x0, u, y)
    jax.block_until_ready(grads)
    compile_time = time.time() - t0

    t0 = time.time()
    for _ in range(steps):
        _, grads = grad_fn(params, x0, u, y)
    jax.block_until_ready(grads)
    step_time = (time.time() - t0) / steps

    return count_num_params(params), compile_time, step_time


def time_models(task, batchsizes=(128, 512), n_train=54000, epochs=150):
    """Time both models on a task and extrapolate to a full training run."""
    seq_len = {"psmnist": 784, "smnist": 784, "scifar10": 1024, "rowmnist": 28}[task]
    configs = {
        "ren": ren_config,
        "r2dn": matched_r2dn_config(ren_config, TASKS[task]),
    }
    print(f"\n{task} (seq_len {seq_len}, nu {TASKS[task]}, "
          f"nx {ren_config['nx']}, nv {ren_config['nv']})")
    for name, config in configs.items():
        for batchsize in batchsizes:
            nparams, compile_time, step_time = time_gradient_step(
                config, task, batchsize, seq_len
            )
            epoch_time = step_time * (n_train // batchsize)
            print(f"  {name:5s} batch {batchsize:5d}: {nparams} params, "
                  f"compile {compile_time:6.1f}s, step {step_time:6.3f}s, "
                  f"epoch {epoch_time/60:5.1f}min, "
                  f"{epochs} epochs {epoch_time*epochs/3600:6.1f}h")


if __name__ == "__main__":
    print("Smoke test (overfit 64 psMNIST images, nx=32, nv=32):")
    smoke_test("ren")
    smoke_test("r2dn")

    time_models("psmnist")
    time_models("scifar10", n_train=45000)
