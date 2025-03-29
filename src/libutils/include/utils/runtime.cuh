#pragma once

#include <cstdlib>

#include <utils/errors.cuh>
#include <utils/location.cuh>

namespace utils {

template<Location L>
auto malloc(std::size_t bytes) -> void* {
    if constexpr (L == Location::Host) {
        return std::malloc(bytes);
    } else {
        void* ptr{};
        handle_cuda_error(cudaMalloc(&ptr, bytes));
        return ptr;
    }
}

template<Location L = Location::Device>
auto malloc_async(std::size_t bytes, cudaStream_t stream = cudaStreamDefault) {
    static_assert(L == Location::Device,
                  "Only device memory can be allocated asynchronously");
    void* ptr{};
    handle_cuda_error(cudaMallocAsync(&ptr, bytes, stream));
    return ptr;
}

inline auto memset(void* ptr, int value, std::size_t bytes) -> void {
    handle_cuda_error(cudaMemset(ptr, value, bytes));
}

inline auto memset_async(void* ptr,
                         int value,
                         std::size_t bytes,
                         cudaStream_t stream = cudaStreamDefault) -> void {
    handle_cuda_error(cudaMemsetAsync(ptr, value, bytes, stream));
}

inline auto memcpy(void* dst, void* src, std::size_t bytes) -> void {
    handle_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
}

inline auto memcpy_async(void* dst,
                         void* src,
                         std::size_t bytes,
                         cudaStream_t stream = cudaStreamDefault) -> void {
    handle_cuda_error(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
}

template<Location L>
auto free(void* ptr) -> void {
    if constexpr (L == Location::Host) {
        std::free(ptr);
    } else {
        handle_cuda_error(cudaFree(ptr));
    }
}

template<Location L = Location::Device>
auto free_async(void* ptr, cudaStream_t stream = cudaStreamDefault) -> void {
    static_assert(L == Location::Device,
                  "Only device memory can be freed asynchronously");
    handle_cuda_error(cudaFreeAsync(ptr, stream));
}

inline auto event_record(cudaEvent_t event, cudaStream_t stream = cudaStreamDefault)
    -> void {
    handle_cuda_error(cudaEventRecord(event, stream));
}

inline auto stream_wait_event(cudaEvent_t event, cudaStream_t stream = cudaStreamDefault)
    -> void {
    handle_cuda_error(cudaStreamWaitEvent(stream, event));
}

inline auto stream_sync(cudaStream_t stream = cudaStreamDefault) -> void {
    handle_cuda_error(cudaStreamSynchronize(stream));
}

inline auto event_sync(cudaEvent_t event) -> void {
    handle_cuda_error(cudaEventSynchronize(event));
}

inline auto device_sync() -> void {
    handle_cuda_error(cudaDeviceSynchronize());
}

template<typename Kernel, typename... Args>
void launch_kernel(Kernel kernel,
                   std::int32_t grid,
                   std::int32_t block,
                   std::int32_t smem,
                   cudaStream_t stream,
                   Args&&... args) {
    kernel<<<grid, block, smem, stream>>>(std::forward<Args>(args)...);
}

} // namespace utils
