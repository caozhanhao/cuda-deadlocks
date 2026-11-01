# CUDA deadlocks

A collection of minimal, standalone reproducers for deadlocks encountered on NVIDIA GPUs. Most of them were collected while debugging [Mooncake](https://github.com/kvcache-ai/Mooncake).


| Case | Scenario |
| --- | --- |
| [01 · Concurrent free](cases/01-concurrent-free/) | `cudaFree` on an unrelated allocation overlaps an H2D write needed by a waiting kernel. |
| [02 · Concurrent D2H](cases/02-concurrent-pageable-d2h/) | Two pageable D2H copies use separate streams; one follows a waiting kernel. |
| [03 · Legacy-stream D2H](cases/03-legacy-pageable-d2h/) | A pageable D2H copy follows a waiting kernel on the legacy default stream. |
| [04 · Stream creation](cases/04-stream-creation/) | A worker creates a stream while a pageable D2H copy waits for a kernel to finish. |
| [05 · Lazy kernel loading](cases/05-lazy-kernel-loading/) | A kernel waits for another kernel to set a flag, with lazy loading enabled. |
| [06 · Block synchronization](cases/06-block-sync/) | The same kernel uses `block.sync()` or `__syncthreads()` at its entry barrier. |

## Build and run

```sh
cd cases/01-concurrent-free
nvcc -std=c++17 -O2 -Xcompiler=-pthread repro.cu -o repro
timeout -k 2s 8s ./repro
timeout -k 2s 8s ./repro --defer-free
```

See each case for its build commands, comparisons, and validation records.
