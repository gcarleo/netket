"""Exercise explicit minSR with a standard RBM and the transverse-field Ising model.

Use --distributed under srun (one process per GPU). Run each control in a
fresh process; a previous solver call can change first-use behaviour.
"""

import argparse
import faulthandler
import importlib.metadata
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--length", type=int, default=16)
parser.add_argument("--alpha", type=int, default=8)
parser.add_argument("--samples", type=int, default=4096)
parser.add_argument("--chains", type=int, default=512)
parser.add_argument("--chunk", type=int, default=128)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--output", type=Path, default=Path("srt-rbm-results"))
args = parser.parse_args()
if args.distributed:
    jax.distributed.initialize()
jax.config.update("jax_enable_x64", True)

# NetKet must be imported after distributed JAX initialization.
import netket as nk  # noqa: E402

faulthandler.dump_traceback_later(90, repeat=True)
rank = jax.process_index()
args.output.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
record = {
    "rank": rank,
    "processes": jax.process_count(),
    "devices": jax.device_count(),
    "device_kind": jax.local_devices()[0].device_kind,
    "versions": {
        name: importlib.metadata.version(name) for name in ("netket", "jax", "jaxlib")
    },
    "length": args.length,
    "alpha": args.alpha,
    "samples": args.samples,
    "chains": args.chains,
    "chunk": args.chunk,
    "complete": False,
    "steps": [],
}


def report(stage):
    record.update(stage=stage, elapsed=time.perf_counter() - started)
    (args.output / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"rank={rank} {stage} elapsed={record['elapsed']:.3f}", flush=True)


graph = nk.graph.Hypercube(length=args.length, n_dim=2, pbc=True)
hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
hamiltonian = nk.operator.IsingJax(hilbert, graph, h=3.0)
model = nk.models.RBM(alpha=args.alpha, param_dtype=jnp.float64)
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=args.chains, sweep_size=1)
state = nk.vqs.MCState(
    sampler,
    model,
    n_samples=args.samples,
    n_discard_per_chain=0,
    chunk_size=args.chunk,
    seed=17,
    sampler_seed=23,
)
record["parameters"] = state.n_parameters
driver = nk.driver.VMC_SR(
    hamiltonian,
    nk.optimizer.Sgd(learning_rate=0.001),
    variational_state=state,
    diag_shift=1e-4,
    use_ntk=True,
    on_the_fly=False,
    mode="real",
    chunk_size_bwd=args.chunk,
    linear_solver=nk.optimizer.solver.cholesky_with_fallback,
)
jax.block_until_ready(state.parameters)
report("before_first_update")


def completed_step(step, log_data, current_driver):
    jax.block_until_ready((current_driver.state.parameters, log_data["Energy"]))
    assert all(
        bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(state.parameters)
    )
    record["steps"].append(
        {"step": int(step), "elapsed": time.perf_counter() - started}
    )
    report("iteration_complete")
    return True


driver.run(n_iter=args.steps, show_progress=False, callback=completed_step)
assert len(record["steps"]) == args.steps
record["complete"] = True
report("complete")
faulthandler.cancel_dump_traceback_later()
if args.distributed:
    jax.distributed.shutdown()
