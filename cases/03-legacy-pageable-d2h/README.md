# D2H: stream type and host memory

A kernel waits for a flag on the legacy stream, followed by a four-byte
pageable D2H. A worker submits the flag's H2D on a separate non-blocking
stream. In recorded runs, both copy host calls remained pending until timeout.

The case varies **the work stream type** and **the D2H host memory type**,
one at a time.

| Variable | Option | Comparison |
| --- | --- | --- |
| Work stream type | `--nonblocking-work-stream` | Legacy vs. non-blocking, keeping the D2H destination pageable. |
| D2H host memory type | `--pinned-d2h` | Pageable vs. pinned, keeping the work stream legacy. |

Compile [repro.cu](repro.cu) from this directory:

```sh
nvcc -std=c++17 -O2 -Xcompiler=-pthread repro.cu -o repro
```

Compare work stream type and host memory, one at a time:

```sh
timeout -k 2s 8s ./repro
timeout -k 2s 8s ./repro --nonblocking-work-stream
timeout -k 2s 8s ./repro --pinned-d2h
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
| Default (legacy stream, pageable D2H) | Timeout | 3/3 |
| `--nonblocking-work-stream` | Completed | 3/3 |
| `--pinned-d2h` | Completed | 3/3 |
