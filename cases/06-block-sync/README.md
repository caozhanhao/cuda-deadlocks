# Block synchronization

> This case was reduced from [kvcache-ai/Mooncake#4791](https://github.com/kvcache-ai/Mooncake/pull/4791/changes/48f65da1b5febee4f833e2e925cd54ea5371c2ab).

This case compares `cooperative_groups::thread_block::sync()` with
`__syncthreads()` at the same entry barrier. In recorded runs, the `block.sync()`
variant timed out, while the `__syncthreads()` variant completed.

`-DFIX_ENTRY` switches only the entry barrier to `__syncthreads()`.

Compile [repro.cu](repro.cu) from this directory:

```sh
nvcc -std=c++17 -O2 -Xcompiler=-pthread repro.cu -o repro
nvcc -std=c++17 -O2 -Xcompiler=-pthread -DFIX_ENTRY repro.cu -o repro-syncthreads
```

Compare the default entry barrier with `__syncthreads()`:

```sh
timeout -k 2s 8s ./repro
timeout -k 2s 8s ./repro-syncthreads
```

In [machine-code experiments](analysis/README.md), changing only the convergence
register assignments allowed the kernel to complete when two code regions used
different registers; sharing one restored the timeout. This suggests a problem
with convergence register reuse.

## Validation records

### 2026-11-01

| Component | Version / configuration |
| :--- | :--- |
| GPU | NVIDIA H20 (`sm_90`) |
| CUDA toolkit | 13.0.88 |
| Host compiler | GCC 11.4.0 |
| Linux kernel | 5.15.0-91-generic |
| NVIDIA driver (Linux kernel module) | 535.161.08 |
| CUDA driver (`libcuda.so`) | 580.126.20, from the CUDA compatibility package |
| Common build options | `-std=c++17 -O2 -Xcompiler=-pthread` |

| Architecture option | Configuration | Observed result | Runs |
| :--- | :--- | :--- | ---: |
| Omitted (nvcc default: `sm_75`) | `block.sync()` entry barrier | Timeout | 3/3 |
| Omitted (nvcc default: `sm_75`) | `-DFIX_ENTRY` (`__syncthreads()` entry barrier) | Completed | 3/3 |
| `-arch=sm_90` | `block.sync()` entry barrier | Timeout | 3/3 |
| `-arch=sm_90` | `-DFIX_ENTRY` (`__syncthreads()` entry barrier) | Completed | 3/3 |
