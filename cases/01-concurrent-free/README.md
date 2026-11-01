# Concurrent free

A kernel waits for a device flag. A worker calls `cudaFree` on an unrelated
allocation, while the main thread submits H2D on a separate non-blocking
stream to set the flag. In recorded runs, both host calls remained pending
until timeout.

`--defer-free` moves the unrelated free after the kernel finishes.

Compile [repro.cu](repro.cu) from this directory:

```sh
nvcc -std=c++17 -O2 -Xcompiler=-pthread repro.cu -o repro
```

Compare the concurrent free with the deferred free:

```sh
timeout -k 2s 8s ./repro
timeout -k 2s 8s ./repro --defer-free
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
| Default (`cudaFree` overlaps H2D) | Timeout | 3/3 |
| `--defer-free` | Completed | 3/3 |
