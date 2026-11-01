#include "../common.h"

#include <atomic>
#include <cuda/atomic>
#include <thread>

// The flag carries only 0 or 1; it does not publish other data.
using FlagRef = cuda::atomic_ref<int, cuda::thread_scope_system>;

// Main writes 1 after D2H B returns.
__global__ void wait_for_release(int* flag) {
    const FlagRef release(*flag);
    while (release.load(cuda::memory_order_relaxed) == 0) {
    }
}

int main(int argc, char** argv) {
    constexpr size_t capacity = 1024 * 1024;
    constexpr size_t bytes_b = 257;
    size_t bytes_a = 257;
    bool pinned_d2h = false;
    ArgumentParser args;
    args.add_argument("--size", bytes_a, "D2H A copy size; B stays at 257 bytes", 1, capacity);
    args.add_argument("--pinned-d2h", pinned_d2h, "Pin D2H A's host destination; B stays pageable");
    setup(argc, argv, args);
    // --size changes only A's submitted byte count. All payload buffers keep
    // the same 1 MiB capacity, and B always copies 257 bytes.
    std::printf("COPY bytes_a=%zu bytes_b=%zu capacity=%zu\n", bytes_a, bytes_b, capacity);

    // B has no stream/event dependency on A or the waiting kernel.
    cudaStream_t stream_a, stream_b;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking));
    unsigned char *device_a = nullptr, *device_b = nullptr;
    CUDA_CHECK(cudaMalloc(&device_a, capacity));
    CUDA_CHECK(cudaMalloc(&device_b, capacity));
    unsigned char* host_a = nullptr;
    // --pinned-d2h uses pinned memory for A. B always uses pageable memory.
    if (pinned_d2h)
        CUDA_CHECK(cudaMallocHost(&host_a, capacity));
    else
        host_a = static_cast<unsigned char*>(std::malloc(capacity));
    auto* host_b = static_cast<unsigned char*>(std::malloc(capacity));
    if (!host_a || !host_b) return 1;
    // Initialize and touch the full buffers before the kernel starts, keeping
    // their preparation identical for every submitted copy size.
    std::memset(host_a, 0, capacity);
    std::memset(host_b, 0, capacity);
    CUDA_CHECK(cudaMemsetAsync(device_a, 0x2b, capacity, stream_a));
    CUDA_CHECK(cudaMemsetAsync(device_b, 0x2b, capacity, stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));

    // The flag is separate from all four payload buffers. CPU and GPU share
    // one allocation: main writes host_flag and the kernel polls device_flag.
    int *host_flag = nullptr, *device_flag = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host_flag, sizeof(int), cudaHostAllocMapped));
    *host_flag = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&device_flag, host_flag, 0));
    std::atomic<bool> worker_ready{false}, start_a{false}, a_requested{false};
    std::thread worker([&] {
        CUDA_CHECK(cudaSetDevice(0));
        worker_ready.store(true);
        while (!start_a.load()) std::this_thread::yield();
        log_step("ENTER D2H A");
        a_requested.store(true);
        // D2H A cannot execute until the kernel on stream_a exits.
        // Main writes the release flag only after D2H B's host call returns.
        CUDA_CHECK(cudaMemcpyAsync(host_a, device_a, bytes_a, cudaMemcpyDeviceToHost, stream_a));
        log_step("RETURN D2H A");
    });
    while (!worker_ready.load()) std::this_thread::yield();
    log_step("SETUP complete");

    // Intended circular wait:
    //   D2H A needs the preceding kernel on stream_a to finish.
    //   The kernel needs main to write 1 to host_flag.
    //   Main writes only after D2H B's host call returns.
    //   In the observed hang, B's host call stalls while A's host call is pending,
    //   so main cannot release the kernel and let D2H A complete.
    log_step("ENTER waiting kernel");
    wait_for_release<<<1, 1, 0, stream_a>>>(device_flag);
    CUDA_CHECK(cudaGetLastError());
    log_step("RETURN waiting kernel");
    start_a.store(true);
    while (!a_requested.load()) std::this_thread::yield();
    // Give thread A time to enter D2H before main submits copy B.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // B uses an independent stream. In the observed hang, B's host call blocks
    // while A is pending, so main cannot reach the store that releases the kernel.
    log_step("ENTER D2H B");
    CUDA_CHECK(cudaMemcpyAsync(host_b, device_b, bytes_b, cudaMemcpyDeviceToHost, stream_b));
    log_step("RETURN D2H B");
    // Release the kernel before synchronizing stream_a, which is waiting for it.
    log_step("STORE CPU release flag");
    FlagRef(*host_flag).store(1, cuda::memory_order_relaxed);
    log_step("ENTER final synchronization");
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_b));
    worker.join();
    log_step("RETURN final synchronization");

    // Both copies and the waiting kernel have finished; their buffers can be freed.
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    // Match the allocator used for A's host destination.
    if (pinned_d2h)
        CUDA_CHECK(cudaFreeHost(host_a));
    else
        std::free(host_a);
    std::free(host_b);
    CUDA_CHECK(cudaFreeHost(host_flag));
    CUDA_CHECK(cudaStreamDestroy(stream_a));
    CUDA_CHECK(cudaStreamDestroy(stream_b));
    std::puts("RESULT completed=1");
    return 0;
}
