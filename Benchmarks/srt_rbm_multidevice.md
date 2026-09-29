# Explicit minSR with a distributed RBM

This standalone probe uses NetKet's built-in real-valued RBM, a periodic
two-dimensional transverse-field Ising Hamiltonian, `MetropolisLocal`, and
`VMC_SR(use_ntk=True, on_the_fly=False)` with the standard
`cholesky_with_fallback` solver. It generates its own parameters and samples.

The defaults are 256 spins, RBM density 8, 4,096 global samples, 512 chains,
and chunk size 128. Three optimizer iterations run in a fresh process.
This is a short execution diagnostic; it is not a converged energy estimate.

Under an allocated Slurm job, with the site's usual GPU environment loaded:

```sh
srun --input=none --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 \
  --gpu-bind=none python Benchmarks/srt_rbm_multidevice.py --distributed
```

Use a bounded allocation when testing a possible stall. Per-process JSON
records and progress messages distinguish startup from completed updates.
A Python stack is printed every 90 seconds while the program is running.
Repeat each configuration in a fresh allocation/process rather than after
other SR or solver calls.

A small CPU smoke check is:

```sh
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 \
  python Benchmarks/srt_rbm_multidevice.py \
  --length 3 --alpha 1 --samples 32 --chains 8 --chunk 8 --steps 3
```

Validation status: the four-device CPU smoke check passes with NetKet
3.22.3 and JAX/JAXlib 0.10.1. Multi-node GPU reproduction is pending.
There is no library modification in this branch.
