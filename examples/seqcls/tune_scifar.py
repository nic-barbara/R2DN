"""Re-tune the R2DN architecture for sCIFAR10 at the final training schedule.

Runs unattended: works through a probe, a decision gate, and a fallback sweep,
stopping when it runs out of useful work or hits a wall-clock deadline.

Background. The architecture currently used for sCIFAR10 (nx=128, nv=39, L=2)
was chosen by an allocation sweep run at 150 epochs, before we found that
extending the schedule to 600 epochs was worth +7 points - four times larger
than the entire allocation effect. Under-training plausibly favours state
dimension over nonlinear capacity, since a model behaves close to linearly
early in training, so that ranking may not survive at the real schedule. The
symptom is that the harder task ended up with the weaker nonlinearity: on
psMNIST the LBDN gets ~70k parameters, on sCIFAR10 only ~40k, because nx=128
spends (2nx)^2 = 66k on the linear part alone.

Plan.
  Stage 1 (probe, ~4h):  nx=96 at L=2 and L=4, seed 0.
  Gate:                  did either clear 64.06%, the top of the incumbent's
                         3-seed range? Anything below that is seed noise.
  Stage 2a (~4h):        if yes, two more seeds of the winner -> a 3-seed
                         number, and the REN results stand unchanged.
  Stage 2b (~10h):       if no, sweep nv against nh at nx=96. That split has
                         never been varied on either task: matched_r2dn_config
                         hard-codes nv_R2DN = nv_REN/2.
  Stage 3:               extra seeds of the best config, if time remains.

Everything is skipped if its results file already exists, so this can be
stopped and restarted freely.
"""
import jax
import sys
import time
from pathlib import Path

dirpath = Path(__file__).resolve().parent
sys.path.append(str(dirpath.parent))

from utils import seqcls
from utils import utils

jax.config.update("jax_default_matmul_precision", "highest")

# Stop starting new runs once this much wall-clock has passed
DEADLINE_HOURS = 15.0
HOURS_PER_RUN = 2.2

# Top of the incumbent's 3-seed range (nx=128, nv=39, L=2: 63.31% mean,
# 62.54-64.06%). A new config must beat this to be worth adopting.
GATE = 0.6406
INCUMBENT = "nx=128, nv=39, L=2 (63.31% mean, 62.54-64.06%)"

BUDGET = 131366        # parameter count of the incumbent
NX = 96
PROBE = [(NX, 76, 2), (NX, 76, 4)]           # (nx, nv, layers)
SWEEP = [(NX, nv, 2) for nv in (20, 40, 60, 100, 120)]

SCHEDULE = {
    "experiment": "scifar10",
    "network": "contracting_r2dn",
    "batchsize": 512,
    "epochs": 600,
    "lr_cuts": (120, 550),
    "lr": 1e-3,
    "activation": "relu",
    "init_method": "long_memory",
    "augment": True,
    "tag": "tune600",
}

start_time = time.time()


def elapsed():
    return (time.time() - start_time) / 3600


def time_remaining():
    return DEADLINE_HOURS - elapsed()


def r2dn_params(nu, nx, nv, ny, nh):
    """Parameter count for an R2DN, matching robustnn.r2dn.ContractingR2DN."""
    n = (2*nx)**2 + nx**2 + 2*nx*nv + nx*nu + nv*nu + ny*(nx+nv+nu) + nx+nv+ny + 1
    ins = (nv,) + tuple(nh)
    outs = tuple(nh) + (nv,)
    for k in range(len(nh)):
        n += (ins[k] + outs[k])*outs[k] + 1 + 2*outs[k]
    n += (ins[-1] + nv)*nv + 1 + nv
    return n


def fit_nh(nx, nv, layers, nu=3, ny=10):
    """Widest LBDN that fits the budget at this nx, nv and depth."""
    nh = 4
    while r2dn_params(nu, nx, nv, ny, (nh + 1,)*layers) <= BUDGET:
        nh += 1
    return nh


def build_config(nx, nv, layers, seed):
    config = seqcls.default_config()
    config.update(SCHEDULE)
    config.update({
        "nx": nx, "nv": nv, "nh": (fit_nh(nx, nv, layers),)*layers, "seed": seed,
    })
    return config


def run_one(nx, nv, layers, seed):
    """Train one configuration unless its results are already on disk."""
    config = build_config(nx, nv, layers, seed)
    filepath, fname = utils.generate_fname(config)
    if filepath.exists():
        print(f"[{elapsed():.1f}h] Skipping {fname}, already done.", flush=True)
        return
    if time_remaining() < HOURS_PER_RUN:
        print(f"[{elapsed():.1f}h] Out of time, not starting {fname}.", flush=True)
        return
    print(f"\n[{elapsed():.1f}h] === nx={nx} nv={nv} L={layers} "
          f"nh={config['nh'][0]} seed={seed} "
          f"({r2dn_params(3, nx, nv, 10, config['nh'])} params) ===", flush=True)
    seqcls.print_summary(config, seqcls.run(config))


def completed_runs():
    """Every finished run from this sweep, best first."""
    runs = []
    for f in sorted(Path("results/scifar10").glob("*.pickle")):
        config, _, results = utils.load_results(f)
        if config.get("tag") != SCHEDULE["tag"]:
            continue
        runs.append((results["test_at_best_val"], config))
    return sorted(runs, key=lambda r: -r[0])


def describe(config):
    return (f"nx={config['nx']}, nv={config['nv']}, L={len(config['nh'])}, "
            f"nh={config['nh'][0]}, seed={config['seed']}")


def summarize(note):
    """Write a summary for Nic to read in the morning."""
    runs = completed_runs()
    lines = [
        "# sCIFAR10 R2DN re-tuning\n",
        f"Incumbent: {INCUMBENT}\n",
        f"Gate to beat: {100*GATE:.2f}%\n",
        f"\n{note}\n",
        "\n| config | seed | test @ best val |",
        "|---|---|---|",
    ]
    for acc, config in runs:
        lines.append(f"| nx={config['nx']}, nv={config['nv']}, L={len(config['nh'])}, "
                     f"nh={config['nh'][0]} | {config['seed']} | {100*acc:.2f}% |")
    lines.append(f"\nFinished after {elapsed():.1f}h.\n")
    text = "\n".join(lines)
    Path("results/scifar_tuning_summary.md").write_text(text)
    print("\n" + text, flush=True)


def main():
    # Stage 1: probe nx=96 at two depths
    for nx, nv, layers in PROBE:
        run_one(nx, nv, layers, seed=0)

    runs = completed_runs()
    if not runs:
        summarize("No runs completed. Something went wrong.")
        return

    best_acc, best = runs[0]
    print(f"\n[{elapsed():.1f}h] Best probe: {describe(best)} at "
          f"{100*best_acc:.2f}% (gate {100*GATE:.2f}%)", flush=True)

    # Stage 2b: the probe did not clear the gate, so sweep the nv/nh split
    if best_acc < GATE:
        print(f"[{elapsed():.1f}h] Gate not cleared. Sweeping nv at nx={NX}.", flush=True)
        for nx, nv, layers in SWEEP:
            run_one(nx, nv, layers, seed=0)
        runs = completed_runs()
        best_acc, best = runs[0]
        print(f"\n[{elapsed():.1f}h] Best after sweep: {describe(best)} at "
              f"{100*best_acc:.2f}%", flush=True)

    # Stage 2a/3: repeat the best configuration over more seeds
    if best_acc >= GATE:
        print(f"[{elapsed():.1f}h] Gate cleared. Adding seeds for "
              f"{describe(best)}.", flush=True)
        for seed in (1, 2):
            run_one(best["nx"], best["nv"], len(best["nh"]), seed)
        note = (f"**{describe(best)} cleared the gate at {100*best_acc:.2f}%.** "
                "Seeds added below. The REN results stand unchanged, so this is "
                "the new R2DN number if the extra seeds hold up.")
    else:
        note = (f"**Nothing cleared the gate.** Best was {describe(best)} at "
                f"{100*best_acc:.2f}%, against {100*GATE:.2f}% needed. The "
                "incumbent architecture stands - ship the existing numbers.")

    summarize(note)


if __name__ == "__main__":
    main()
