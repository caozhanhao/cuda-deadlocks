#pragma once

#include "args.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Host utilities only: keep every experimental operation in its case.
inline void check_cuda(cudaError_t status, const char* call, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: %s: %s\n", file, line, call, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

inline void check_cu(CUresult status, const char* call, const char* file, int line) {
    if (status == CUDA_SUCCESS) return;
    const char* name = "unknown";
    cuGetErrorName(status, &name);
    std::fprintf(stderr, "%s:%d: %s: %d %s\n", file, line, call, int(status), name);
    std::_Exit(3);
}

#define CU_CHECK(call) check_cu((call), #call, __FILE__, __LINE__)

inline void log_step(const char* message) {
    static const auto start = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::printf("%9.3f ms  %s\n", ms, message);
}

// Common program startup, called once at the beginning of main. Keep the
// experimental allocations, streams, copies, and launches in each case.
// Each caller declares its own options; no case-specific flag lives here.
inline cudaDeviceProp setup(int argc, char** argv, ArgumentParser& args) {
    // Help, type/range errors, and unknown options exit before touching CUDA.
    args.parse_args(argc, argv);

    // stdout is normally buffered when redirected to a file or pipe. Disable
    // buffering before the first print so a killed/hung process keeps its last
    // API-entry marker in the captured log. This does not synchronize the GPU.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    args.print_values();
    log_step("ENTER setup");

    // Select the first device exposed through CUDA_VISIBLE_DEVICES.
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("DEVICE=%s compute_capability=%d.%d driver_api=%d runtime=%d\n", device.name,
                device.major, device.minor, driver, runtime);
    const char* settings[] = {"CUDA_VISIBLE_DEVICES",        "CUDA_MODULE_LOADING",
                              "CUDA_MODULE_DATA_LOADING",    "CUDA_LAUNCH_BLOCKING",
                              "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_MAX_COPY_CONNECTIONS"};
    for (const char* key : settings) {
        std::printf("ENV %s=%s\n", key, std::getenv(key) ? std::getenv(key) : "<unset>");
    }
    return device;
}
