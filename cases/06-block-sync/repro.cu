#include "../common.h"

#include <cooperative_groups.h>
#include <cstdint>
#include <thread>

struct Work {
    unsigned rounds;
    std::uint64_t* destination;
    std::uint64_t payload;
};

__device__ __forceinline__ bool receive(unsigned* state) {
    const unsigned value = atomicCAS_system(state, 0, 0);
    if (value == 0) return false;
    return atomicCAS_system(state, value, 4) == value;
}

__global__ void reproduce(Work* work, unsigned* idle) {
    auto block = cooperative_groups::this_thread_block();
    const unsigned rank = block.thread_rank();

    if (rank == 0) {
        while (receive(idle)) {
            while (atomicCAS_system(idle, 0, 0)) {
            }
        }
    }

    __shared__ Work shared;
    if (rank == 0) shared = *work;

#ifdef FIX_ENTRY
    __syncthreads();
#else
    block.sync();
#endif

    for (unsigned round = 0; round < shared.rounds; ++round) {
        if (rank < shared.rounds) {
            auto* destination = reinterpret_cast<volatile std::uint64_t*>(shared.destination);
            destination[rank] = *reinterpret_cast<const volatile std::uint64_t*>(&shared.payload);
        }
        block.sync();
    }
    __syncthreads();
}

int main() {
    Work* work = nullptr;
    unsigned* idle = nullptr;
    std::uint64_t* scratch = nullptr;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&work, sizeof(Work)));
    CUDA_CHECK(cudaMalloc(&idle, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(idle, 0, sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&scratch, 2 * sizeof(std::uint64_t)));

    Work input{};
    input.rounds = 2;
    input.destination = scratch;
    CUDA_CHECK(cudaMemcpy(work, &input, sizeof(input), cudaMemcpyHostToDevice));

    cudaEvent_t done;
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    reproduce<<<1, 128>>>(work, idle);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(done));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    for (;;) {
        const auto status = cudaEventQuery(done);
        if (status == cudaSuccess) break;
        if (status != cudaErrorNotReady) CUDA_CHECK(status);
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "TIMEOUT iteration=0\n");
            std::exit(2);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::printf("PASS\n");
}
