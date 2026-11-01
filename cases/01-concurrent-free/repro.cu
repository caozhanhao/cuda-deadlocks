#include "../common.h"

#include <atomic>
#include <cuda/atomic>
#include <thread>

// The flag carries only 0 or 1; it does not publish other data.
using FlagRef = cuda::atomic_ref<int, cuda::thread_scope_system>;

// Main's H2D writes 1 to let this kernel exit.
__global__ void wait_for_release(int* flag) {
    const FlagRef release(*flag);
    while (release.load(cuda::memory_order_relaxed) == 0) {
    }
}

int main(int argc, char** argv) {
    bool defer_free = false;
    ArgumentParser args;
    args.add_argument("--defer-free", defer_free, "Defer the unrelated free until cleanup");
    setup(argc, argv, args);

    // The kernel runs on work_stream; H2D writes its flag on release_stream.
    // There are no events or stream waits ordering one stream after the other.
    cudaStream_t work_stream, release_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&release_stream, cudaStreamNonBlocking));
    int *device_flag = nullptr, *host_one = nullptr;
    void* unrelated = nullptr;
    CUDA_CHECK(cudaMalloc(&device_flag, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&host_one, sizeof(int)));
    // Neither the kernel nor H2D uses this buffer; it is only a cudaFree target.
    CUDA_CHECK(cudaMalloc(&unrelated, 4 * 1024 * 1024));
    *host_one = 1;
    // Set the flag to 0 before starting the kernel that polls it.
    CUDA_CHECK(cudaMemsetAsync(device_flag, 0, sizeof(int), work_stream));
    CUDA_CHECK(cudaStreamSynchronize(work_stream));

    std::atomic<bool> worker_ready{false}, start_free{false}, free_requested{false};
    std::thread worker([&] {
        // Select this thread's device before the main thread launches the kernel.
        CUDA_CHECK(cudaSetDevice(0));
        worker_ready.store(true);
        while (!start_free.load()) std::this_thread::yield();
        if (!defer_free) {
            log_step("ENTER cudaFree");
            free_requested.store(true);
            // cudaFree can wait for the running kernel to finish, even though
            // the kernel does not use the allocation being freed.
            CUDA_CHECK(cudaFree(unrelated));
            log_step("RETURN cudaFree");
        } else {
            // Leave unrelated allocated. Main will free it after the kernel exits.
            log_step("CONTROL defer cudaFree until cleanup");
            free_requested.store(true);
        }
    });
    while (!worker_ready.load()) std::this_thread::yield();
    log_step("SETUP complete");

    // Observed wait cycle without --defer-free:
    //   cudaFree waits for the kernel to finish.
    //   The kernel waits for H2D to write 1 to device_flag.
    //   The H2D host call blocks while cudaFree is pending.
    // --defer-free moves cudaFree after the kernel and breaks this cycle.
    log_step("ENTER waiting kernel");
    wait_for_release<<<1, 1, 0, work_stream>>>(device_flag);
    CUDA_CHECK(cudaGetLastError());
    log_step("RETURN waiting kernel");
    start_free.store(true);
    while (!free_requested.load()) std::this_thread::yield();
    // Give the worker time to enter cudaFree before main submits H2D.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Write 1 so the kernel can exit. The observed hang stops inside this host
    // call, with the kernel still polling 0 and the worker still in cudaFree.
    log_step("ENTER release H2D");
    CUDA_CHECK(cudaMemcpyAsync(device_flag, host_one, sizeof(int), cudaMemcpyHostToDevice,
                               release_stream));
    log_step("RETURN release H2D");
    log_step("ENTER final synchronization");
    // Wait for the H2D write to device_flag to finish.
    CUDA_CHECK(cudaStreamSynchronize(release_stream));
    // Wait for the kernel to see 1 and exit its polling loop.
    CUDA_CHECK(cudaStreamSynchronize(work_stream));
    worker.join();
    log_step("RETURN final synchronization");

    // With --defer-free, the worker leaves unrelated for main to free here.
    // Otherwise, the worker has already freed it.
    if (defer_free) CUDA_CHECK(cudaFree(unrelated));
    CUDA_CHECK(cudaFree(device_flag));
    CUDA_CHECK(cudaFreeHost(host_one));
    CUDA_CHECK(cudaStreamDestroy(work_stream));
    CUDA_CHECK(cudaStreamDestroy(release_stream));
    std::puts("RESULT completed=1");
    return 0;
}
