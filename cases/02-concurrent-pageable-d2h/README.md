# Concurrent D2H: size and host memory

A kernel waits for a flag on stream A, followed by D2H A. The main thread
submits D2H B on an independent stream and sets the flag after B's host call
returns.

The case varies **A's copy length** and **A's host memory type**, one at
a time. B stays at 257 bytes with a pageable destination.

In recorded runs, the 257/257-byte pageable case timed out. The program
completed after either **reducing A's copy size from 257 to 256 bytes** or
**switching A's host destination from pageable to pinned**. Buffer capacities
remain fixed at 1 MiB.

| Variable | Option | Comparison |
| --- | --- | --- |
| A's copy length | `--size N` | 257 vs. 256 bytes, keeping A pageable. |
| A's host memory type | `--pinned-d2h` | Pageable vs. pinned, keeping A at 257 bytes. |

Compile [repro.cu](repro.cu) from this directory:

```sh
nvcc -std=c++17 -O2 -Xcompiler=-pthread repro.cu -o repro
```

Compare copy length and host memory, one at a time:

```sh
timeout -k 2s 8s ./repro
timeout -k 2s 8s ./repro --size 256
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
| Default (257-byte pageable D2H) | Timeout | 3/3 |
| `--size 256` | Completed | 3/3 |
| `--pinned-d2h` | Completed | 3/3 |
