#include "../common.h"

#include <cuda/atomic>

// The flag carries only 0 or 1; it does not publish other data.
using FlagRef = cuda::atomic_ref<int, cuda::thread_scope_system>;

// Only release_flag writes 1, so this kernel cannot finish without it.
__global__ void wait_for_release(int* flag) {
    const FlagRef release(*flag);
    while (release.load(cuda::memory_order_relaxed) == 0) {
    }
}

__global__ void release_flag(int* flag) { FlagRef(*flag).store(1, cuda::memory_order_relaxed); }

int main(int argc, char** argv) {
    bool preload_kernels = false;
    ArgumentParser args;
    args.add_argument("--preload-kernels", preload_kernels,
                      "Load both kernels before launching either");
    const auto device = setup(argc, argv, args);

    // release_flag must execute before wait_for_release can finish.
    if (!device.concurrentKernels) {
        std::fputs("This case needs concurrent kernel support\n", stderr);
        return 1;
    }

    // Create both non-blocking streams before launching either kernel.
    cudaStream_t work_stream, release_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&release_stream, cudaStreamNonBlocking));
    int* device_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&device_flag, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(device_flag, 0, sizeof(int), work_stream));
    CUDA_CHECK(cudaStreamSynchronize(work_stream));

    if (preload_kernels) {
        // Load both kernels before launching wait_for_release.
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, wait_for_release));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, release_flag));
        log_step("CONTROL kernels preloaded");
    }
    log_step("SETUP complete");

    // Intended circular wait:
    //   wait_for_release needs release_flag to write 1 to device_flag.
    //   release_flag needs its first-use loading to finish.
    //   Loading may wait for wait_for_release to finish.
    log_step("ENTER waiting kernel");
    wait_for_release<<<1, 1, 0, work_stream>>>(device_flag);
    CUDA_CHECK(cudaGetLastError());
    log_step("RETURN waiting kernel");
    // wait_for_release needs this kernel to write 1 to device_flag.
    // Without --preload-kernels, this is the first use of release_flag.
    log_step("ENTER release kernel");
    release_flag<<<1, 1, 0, release_stream>>>(device_flag);
    CUDA_CHECK(cudaGetLastError());
    log_step("RETURN release kernel");
    // Wait for release_flag to finish writing, then for wait_for_release to exit.
    log_step("ENTER final synchronization");
    CUDA_CHECK(cudaStreamSynchronize(release_stream));
    CUDA_CHECK(cudaStreamSynchronize(work_stream));
    log_step("RETURN final synchronization");

    // Both kernels have finished; release their shared flag and streams.
    CUDA_CHECK(cudaFree(device_flag));
    CUDA_CHECK(cudaStreamDestroy(work_stream));
    CUDA_CHECK(cudaStreamDestroy(release_stream));
    std::puts("RESULT completed=1");
    return 0;
}
