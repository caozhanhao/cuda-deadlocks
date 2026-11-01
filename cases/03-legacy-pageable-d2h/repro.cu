#include "../common.h"

#include <atomic>
#include <cuda/atomic>
#include <thread>

// The flag carries only 0 or 1; it does not publish other data.
using FlagRef = cuda::atomic_ref<int, cuda::thread_scope_system>;

// The worker's H2D writes 1 to let this kernel exit.
__global__ void wait_for_release(int* flag) {
    const FlagRef release(*flag);
    while (release.load(cuda::memory_order_relaxed) == 0) {
    }
}

int main(int argc, char** argv) {
    bool nonblocking_work_stream = false, pinned_d2h = false;
    ArgumentParser args;
    args.add_argument("--nonblocking-work-stream", nonblocking_work_stream,
                      "Use a non-blocking work stream");
    args.add_argument("--pinned-d2h", pinned_d2h, "Pin the D2H host destination");
    setup(argc, argv, args);

    // --nonblocking-work-stream moves both the kernel and its D2H together.
    // The copy remains after the kernel in either stream configuration.
    cudaStream_t work_stream = cudaStreamLegacy;
    if (nonblocking_work_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking));
    }
    // Submit the flag update on a separate non-blocking stream.
    cudaStream_t release_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&release_stream, cudaStreamNonBlocking));
    int *device_flag = nullptr, *device_data = nullptr, *host_one = nullptr;
    CUDA_CHECK(cudaMalloc(&device_flag, sizeof(int)));
    // D2H payload and release flag are separate allocations; neither copy
    // overlaps the other's data. Their connection is the progress dependency.
    CUDA_CHECK(cudaMalloc(&device_data, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&host_one, sizeof(int)));
    *host_one = 1;
    // Allocate the same four-byte D2H destination in both variants. Only its
    // host memory type changes; the worker's H2D source stays pinned.
    int* host_data = nullptr;
    if (pinned_d2h)
        CUDA_CHECK(cudaMallocHost(&host_data, sizeof(int)));
    else
        host_data = static_cast<int*>(std::malloc(sizeof(int)));
    if (!host_data) return 1;
    *host_data = -1;
    CUDA_CHECK(cudaMemsetAsync(device_flag, 0, sizeof(int), release_stream));
    CUDA_CHECK(cudaMemsetAsync(device_data, 0, sizeof(int), release_stream));
    CUDA_CHECK(cudaStreamSynchronize(release_stream));

    std::atomic<bool> worker_ready{false}, start_release{false};
    std::thread worker([&] {
        CUDA_CHECK(cudaSetDevice(0));
        worker_ready.store(true);
        while (!start_release.load()) std::this_thread::yield();
        // Give main time to enter D2H before the worker submits H2D.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // This write would end the kernel. In the pageable variant, the H2D
        // host API can block while the main thread's D2H is pending.
        log_step("ENTER release H2D");
        CUDA_CHECK(cudaMemcpyAsync(device_flag, host_one, sizeof(int), cudaMemcpyHostToDevice,
                                   release_stream));
        log_step("RETURN release H2D");
    });
    while (!worker_ready.load()) std::this_thread::yield();
    log_step("SETUP complete");

    // Intended circular wait:
    //   D2H needs the preceding kernel on work_stream to finish.
    //   The kernel needs H2D on release_stream to write 1 to device_flag.
    //   In the observed hang, the H2D host call stalls during the pending D2H
    //   host call, leaving the flag update and hence D2H unable to complete.
    log_step("ENTER waiting kernel");
    wait_for_release<<<1, 1, 0, work_stream>>>(device_flag);
    CUDA_CHECK(cudaGetLastError());
    log_step("RETURN waiting kernel");
    start_release.store(true);
    // D2H follows the waiting kernel on work_stream.
    // The worker submits the flag update on the separate non-blocking
    // release_stream. In the observed hang, both copy API calls stay blocked.
    //
    // --pinned-d2h changes only the D2H destination's memory type.
    log_step("ENTER D2H");
    CUDA_CHECK(
        cudaMemcpyAsync(host_data, device_data, sizeof(int), cudaMemcpyDeviceToHost, work_stream));
    log_step("RETURN D2H");
    // In the observed hang, both copy calls remain blocked above this point.
    log_step("ENTER final synchronization");
    CUDA_CHECK(cudaStreamSynchronize(work_stream));
    worker.join();
    CUDA_CHECK(cudaStreamSynchronize(release_stream));
    log_step("RETURN final synchronization");

    // Both copies and the kernel have finished; their buffers can be freed.
    CUDA_CHECK(cudaFree(device_flag));
    CUDA_CHECK(cudaFree(device_data));
    CUDA_CHECK(cudaFreeHost(host_one));
    // Match the allocator chosen for the D2H destination.
    if (pinned_d2h)
        CUDA_CHECK(cudaFreeHost(host_data));
    else
        std::free(host_data);
    if (nonblocking_work_stream) CUDA_CHECK(cudaStreamDestroy(work_stream));
    CUDA_CHECK(cudaStreamDestroy(release_stream));
    std::puts("RESULT completed=1");
    return 0;
}
