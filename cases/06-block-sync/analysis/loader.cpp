// Driver-API loader for case 06 convergence-register controls.
// Usage: loader <cubin>
// Exit:  0 = completed, 2 = 3-second deadline, 3 = CUDA API error.
#include "../../common.h"

#include <cstdint>
#include <thread>

struct Work {
    unsigned rounds;
    std::uint64_t* destination;
    std::uint64_t payload;
};

int main(int argc, char** argv) {
    if (argc != 2) return 8;
    CU_CHECK(cuInit(0));
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));
    CUcontext context;
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CU_CHECK(cuCtxSetCurrent(context));
    CUmodule module;
    CU_CHECK(cuModuleLoad(&module, argv[1]));
    CUfunction function;
    CU_CHECK(cuModuleGetFunction(&function, module, "_Z9reproduceP4WorkPj"));

    CUdeviceptr work, idle, scratch;
    CU_CHECK(cuMemAlloc(&work, sizeof(Work)));
    CU_CHECK(cuMemAlloc(&idle, sizeof(unsigned)));
    CU_CHECK(cuMemAlloc(&scratch, 2 * sizeof(std::uint64_t)));
    CU_CHECK(cuMemsetD8(idle, 0, sizeof(unsigned)));

    Work input{};
    input.rounds = 2;
    input.destination = reinterpret_cast<std::uint64_t*>(scratch);
    CU_CHECK(cuMemcpyHtoD(work, &input, sizeof(input)));

    CUevent event;
    CU_CHECK(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    void* arguments[] = {&work, &idle};
    CU_CHECK(cuLaunchKernel(function, 1, 1, 1, 128, 1, 1, 0, nullptr, arguments, nullptr));
    CU_CHECK(cuEventRecord(event, nullptr));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    for (;;) {
        const CUresult status = cuEventQuery(event);
        if (status == CUDA_SUCCESS) break;
        if (status != CUDA_ERROR_NOT_READY) CU_CHECK(status);
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(stderr, "TIMEOUT iteration=0\n");
            std::_Exit(2);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::printf("PASS\n");
    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuDevicePrimaryCtxRelease(device));
}
