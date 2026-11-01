#include "../common.h"

#include <atomic>
#include <cuda/atomic>
#include <thread>

// The flag carries only 0 or 1; it does not publish other data.
using FlagRef = cuda::atomic_ref<int, cuda::thread_scope_system>;

// The worker writes 1 after new_stream has been created.
__global__ void wait_for_release(int* flag) {
    const FlagRef release(*flag);
    while (release.load(cuda::memory_order_relaxed) == 0) {
    }
}

int main(int argc, char** argv) {
    bool precreate_stream = false;
    ArgumentParser args;
    args.add_argument("--precreate-stream", precreate_stream,
                      "Create the unused stream before the wait");
    setup(argc, argv, args);

    cudaStream_t work_stream, new_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking));
    // The only A/B change is WHEN this stream is created. It never carries
    // work: the main experiment needs its host API to return, not GPU use of it.
    if (precreate_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&new_stream, cudaStreamNonBlocking));
    }
    int* device_data = nullptr;
    int readback = 0;
    CUDA_CHECK(cudaMalloc(&device_data, sizeof(int)));
    // This initialization is complete before the first kernel launch.
    CUDA_CHECK(cudaMemsetAsync(device_data, 0x2b, sizeof(int), work_stream));
    CUDA_CHECK(cudaStreamSynchronize(work_stream));
    // CPU and GPU share one allocation: the worker writes through host_flag,
    // and the kernel polls the same integer through device_flag.
    int *host_flag = nullptr, *device_flag = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host_flag, sizeof(int), cudaHostAllocMapped));
    *host_flag = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&device_flag, host_flag, 0));

    std::atomic<bool> worker_ready{false}, start_worker{false};
    std::thread worker([&] {
        CUDA_CHECK(cudaSetDevice(0));
        worker_ready.store(true);
        while (!start_worker.load()) std::this_thread::yield();
        // Give main time to enter D2H before the worker creates new_stream.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!precreate_stream) {
            // Observed host wait: pending pageable D2H can prevent this API
            // from returning, even though the requested stream is non-blocking.
            log_step("ENTER stream creation");
            CUDA_CHECK(cudaStreamCreateWithFlags(&new_stream, cudaStreamNonBlocking));
            log_step("RETURN stream creation");
        }
        // Write 1 so the kernel can exit. If stream creation blocks above,
        // the worker never reaches this store and the kernel keeps waiting.
        log_step("STORE CPU release flag");
        FlagRef(*host_flag).store(1, cuda::memory_order_relaxed);
    });
    while (!worker_ready.load()) std::this_thread::yield();
    log_step("SETUP complete");

    // Intended circular wait:
    //   D2H -> earlier kernel -> CPU store after stream creation
    //       -> stream-creation host wait during the pending D2H.
    log_step("ENTER waiting kernel");
    wait_for_release<<<1, 1, 0, work_stream>>>(device_flag);
    CUDA_CHECK(cudaGetLastError());
    log_step("RETURN waiting kernel");
    start_worker.store(true);
    // This copy uses the kernel's stream, so it cannot finish until the worker
    // writes the release flag and the kernel exits.
    log_step("ENTER D2H");
    CUDA_CHECK(
        cudaMemcpyAsync(&readback, device_data, sizeof(int), cudaMemcpyDeviceToHost, work_stream));
    log_step("RETURN D2H");
    log_step("ENTER final synchronization");
    CUDA_CHECK(cudaStreamSynchronize(work_stream));
    worker.join();
    log_step("RETURN final synchronization");

    // The worker has returned and the kernel has exited; release their resources.
    CUDA_CHECK(cudaStreamDestroy(new_stream));
    CUDA_CHECK(cudaStreamDestroy(work_stream));
    CUDA_CHECK(cudaFree(device_data));
    CUDA_CHECK(cudaFreeHost(host_flag));
    std::puts("RESULT completed=1");
    return 0;
}
