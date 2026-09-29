"""Exercise explicit minSR with a standard RBM and the transverse-field Ising model.

Use --distributed under srun (one process per GPU). Run each control in a
fresh process; a previous solver call can change first-use behaviour.
"""

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--length", type=int, default=16)
parser.add_argument("--alpha", type=int, default=8)
parser.add_argument("--complex-parameters", action="store_true")
parser.add_argument("--initialize-sr", action="store_true")
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
from netket._src.ngd.sr_srt_common import _prepare_input  # noqa: E402

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
    "complex_parameters": args.complex_parameters,
    "initialize_sr": args.initialize_sr,
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
model = nk.models.RBM(
    alpha=args.alpha,
    param_dtype=jnp.complex128 if args.complex_parameters else jnp.float64,
)
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
    mode="complex" if args.complex_parameters else "real",
    chunk_size_bwd=args.chunk,
    linear_solver=nk.optimizer.solver.cholesky_with_fallback,
)
jax.block_until_ready(state.parameters)

if args.initialize_sr:
    # Diagnostic only: evaluate and discard each stage before the intact driver.
    # The Gram uses actual samples; the solver uses a same-sized identity.
    initial_parameters = jax.tree.map(np.asarray, state.parameters)
    initialization_start = time.perf_counter()
    report("initializing_sr")
    energy = jax.block_until_ready(state.local_estimators(hamiltonian))
    samples = state.samples.reshape(-1, hilbert.size)
    mode = "complex" if args.complex_parameters else "real"
    jacobian = jax.block_until_ready(
        nk.jax.jacobian(
            state._apply_fun,
            state.parameters,
            samples,
            state.model_state,
            mode=mode,
            dense=True,
            center=True,
            chunk_size=args.chunk,
        )
    )
    jacobian, _ = jax.block_until_ready(
        _prepare_input(
            jacobian,
            energy,
            mode=mode,
            scaling_factor=1 / samples.shape[0],
        )
    )
    column_sharding = NamedSharding(jax.sharding.get_mesh(), P(None, "S"))
    replicated = NamedSharding(jax.sharding.get_mesh(), P())

    def redistribute(x):
        x = nk.jax.sharding.pad_axis_for_sharding(x, axis=1, padding_value=0.0)
        return jax.lax.with_sharding_constraint(x, column_sharding)

    columns = jax.block_until_ready(
        jax.jit(
            redistribute,
            out_shardings=column_sharding,
        )(jacobian)
    )
    gram = jax.block_until_ready(
        jax.jit(
            lambda x: x @ x.T,
            out_shardings=replicated,
        )(columns)
    )
    side, dtype = gram.shape[0], gram.dtype
    record["initialization_gram_sha256"] = hashlib.sha256(
        np.ascontiguousarray(gram.addressable_shards[0].data).tobytes()
    ).hexdigest()
    record["initialization_gram_shape"] = list(gram.shape)
    del jacobian, columns, gram, energy, samples
    multihost_utils.sync_global_devices("rbm_gram_initialized")
    report("initializing_solver")

    @jax.jit
    def initialize_solver(scale):
        return nk.optimizer.solver.cholesky_with_fallback(
            scale * jnp.eye(side, dtype=dtype), jnp.ones(side, dtype=dtype)
        )

    # A runtime scalar prevents constant folding of the factorization.
    solution, info = jax.block_until_ready(
        initialize_solver(jax.device_put(np.asarray(1.0, dtype=dtype), replicated))
    )
    np.testing.assert_allclose(np.asarray(solution), 1.0)
    assert not bool(info["solver_fallback"])
    del solution, info
    multihost_utils.sync_global_devices("rbm_solver_initialized")
    for before, after in zip(
        jax.tree.leaves(initial_parameters), jax.tree.leaves(state.parameters)
    ):
        np.testing.assert_array_equal(before, np.asarray(after))
    del initial_parameters
    record["initialization_seconds"] = time.perf_counter() - initialization_start

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
