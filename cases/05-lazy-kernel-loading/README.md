# Lazy kernel loading

A kernel waits for a flag that a second kernel writes. The launches use
separate non-blocking streams with lazy loading enabled. In the recorded
runs, both launch calls returned, but the final stream synchronization
remained pending until timeout.

`CUDA_MODULE_LOADING=EAGER` loads the kernels during initialization.
`--preload-kernels` loads both kernels with `cudaFuncGetAttributes` before
either launch.

Compile [repro.cu](repro.cu) from this directory:

```sh
nvcc -std=c++17 -O2 repro.cu -o repro
```

Compare default loading, eager loading, and explicit kernel preloading:

```sh
timeout -k 2s 8s ./repro
CUDA_MODULE_LOADING=EAGER timeout -k 2s 8s ./repro
timeout -k 2s 8s ./repro --preload-kernels
```

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
| Build options | `-std=c++17 -O2 -arch=sm_90 -Xcompiler=-pthread` |

| Configuration | Observed result | Runs |
| :--- | :--- | ---: |
| `CUDA_MODULE_LOADING=LAZY` | Timeout | 3/3 |
| `CUDA_MODULE_LOADING=EAGER` | Completed | 3/3 |
| `CUDA_MODULE_LOADING=LAZY` with `--preload-kernels` | Completed | 3/3 |
