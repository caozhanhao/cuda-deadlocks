# Block synchronization analysis

This experiment changes only the register operands of two BSSY/BSYNC pairs
in the kernel from [repro.cu](../repro.cu).

| Case | Pair at `0x3c0` / `0x470` | Pair at `0x5e0` / `0x680` | Recorded result |
| --- | --- | --- | --- |
| `B0-B0` | B0 | B0 | 3/3 TIMEOUT |
| `B0-B3` | B0 | B3 | 3/3 PASS |
| `B3-B0` | B3 | B0 | 3/3 PASS |
| `B3-B3` | B3 | B3 | 3/3 TIMEOUT |

Sharing a convergence register between these two pairs appears to cause the
deadlock.

The supplied [reference.cubin](reference.cubin) contains the original `B0-B0`
kernel compiled with CUDA 13.0.88. [experiment.py](experiment.py) checks its
SHA256, then changes the register operands to produce all four cases in `build/`.
The corresponding [PTX](reference.ptx) and [SASS](reference.sass) are included
for inspection.

Compile the [loader](loader.cpp) and run the comparison from this directory:

```sh
nvcc -std=c++17 -O2 --cudart=none -Xcompiler=-pthread loader.cpp -lcuda -o loader
python3 experiment.py --run
```

Each case runs in three separate processes. The loader reports `PASS` when the
kernel completes, or `TIMEOUT` if it remains pending after three seconds.
Running `python3 experiment.py` without `--run` only generates the cubins.

The offsets above are relative to the kernel, which starts at file offset
`0x780` in the supplied cubin. The register operand occupies bits 19:16 of each
instruction; every other bit stays unchanged. To inspect the instructions:

```sh
nvdisasm reference.cubin
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
| Cubin build options | `-O2 -arch=sm_90` |
