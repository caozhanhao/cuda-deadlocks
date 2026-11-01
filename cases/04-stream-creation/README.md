# Stream creation during D2H

A kernel waits for a flag. A pageable D2H follows it on the same stream.
A worker creates an unused stream, then sets the flag. In recorded runs,
both the D2H and stream-creation host calls remained pending until timeout.

`--precreate-stream` creates the unused stream during setup.

Compile [repro.cu](repro.cu) from this directory:

```sh
nvcc -std=c++17 -O2 -Xcompiler=-pthread repro.cu -o repro
```

Compare stream creation during D2H with creation during setup:

```sh
timeout -k 2s 8s ./repro
timeout -k 2s 8s ./repro --precreate-stream
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
| Default (stream creation during D2H) | Timeout | 3/3 |
| `--precreate-stream` | Completed | 3/3 |
