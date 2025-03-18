#pragma once

#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>

#include <gsl/gsl-lite.hpp>

#include <utils/errors.cuh>

namespace utils {

enum class Location : std::uint8_t {
    Host,
    Device,
};

template<std::floating_point T, Location L>
struct CSR {
    std::int32_t nnz{};
    std::int32_t m{};
    std::int32_t n{};
    std::int32_t* rows_ptr{};
    std::int32_t* cols{};
    T* values{};

    CSR() = default;
    CSR(std::int32_t nnz_, std::int32_t m_, std::int32_t n_);
    CSR(const CSR& other);
    CSR(CSR&& other) noexcept;
    auto operator=(const CSR& other) -> CSR&;
    auto operator=(CSR&& other) noexcept -> CSR&;
    ~CSR();

    static auto load_from_filename(const std::string& filename) -> CSR<T, L>
    requires(L == Location::Host);

    auto save_to_filename(const std::string& filename) const -> void
    requires(L == Location::Host);

    template<Location To>
    auto to() const -> CSR<T, To>;

    template<std::floating_point T_, Location L_>
    friend auto swap(CSR<T_, L_>&, CSR<T_, L_>) noexcept -> void;

    auto release() -> void;

private:

    static auto allocate(std::int32_t count, std::size_t size) -> void*;

    static auto free(void* ptr) -> void;

    static auto copy(void* dst,
                     const void* src,
                     std::int32_t count,
                     std::size_t size,
                     cudaMemcpyKind kind) -> void;
};

template<std::floating_point T, Location L>
CSR<T, L>::CSR(const std::int32_t nnz_, const std::int32_t m_, const std::int32_t n_)
    : nnz{nnz_},
      m{m_},
      n{n_},
      rows_ptr{static_cast<std::int32_t*>(allocate(m + 1, sizeof(std::int32_t)))},
      cols{static_cast<std::int32_t*>(allocate(nnz, sizeof(std::int32_t)))},
      values{static_cast<T*>(allocate(nnz, sizeof(T)))} {}

template<std::floating_point T, Location L>
CSR<T, L>::CSR(const CSR& other)
    : nnz{other.nnz},
      m{other.m},
      n{other.n},
      rows_ptr{static_cast<std::int32_t*>(allocate(m + 1, sizeof(std::int32_t)))},
      cols{static_cast<std::int32_t*>(allocate(nnz, sizeof(std::int32_t)))},
      values{static_cast<T*>(allocate(nnz, sizeof(T)))} {
    const auto direction = (L == Location::Host) ? cudaMemcpyHostToHost
                                                 : cudaMemcpyDeviceToDevice;

    copy(rows_ptr, other.rows_ptr, m + 1, sizeof(std::int32_t), direction);
    copy(cols, other.cols, nnz, sizeof(std::int32_t), direction);
    copy(values, other.values, nnz, sizeof(T), direction);
}

template<std::floating_point T, Location L>
CSR<T, L>::CSR(CSR&& other) noexcept
    : nnz{std::exchange(other.nnz, 0)},
      m{std::exchange(other.m, 0)},
      n{std::exchange(other.n, 0)},
      rows_ptr{std::exchange(other.rows_ptr, nullptr)},
      cols{std::exchange(other.cols, nullptr)},
      values{std::exchange(other.values, nullptr)} {}

template<std::floating_point T, Location L>
auto CSR<T, L>::operator=(const CSR& other) -> CSR& {
    if (this == &other)
        return *this;
    auto tmp{other};
    swap(*this, tmp);
    return *this;
}

template<std::floating_point T, Location L>
auto CSR<T, L>::operator=(CSR&& other) noexcept -> CSR& {
    if (this == &other)
        return *this;
    auto tmp{std::move(other)};
    swap(*this, tmp);
    return *this;
}

template<std::floating_point T, Location L>
CSR<T, L>::~CSR() {
    release();
}

template<std::floating_point T, Location L>
auto CSR<T, L>::load_from_filename(const std::string& filename) -> CSR<T, L>
requires(L == Location::Host)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Failed to open file " + filename);
    const auto filesize = std::filesystem::file_size(filename);
    if (filesize < 3 * sizeof(std::int32_t))
        throw std::runtime_error("File " + filename + " is too small");

    auto read = [&ifs, &filename](auto* data, std::int32_t size = 1) {
        const auto bytes = gsl::narrow_cast<std::int32_t>(sizeof(*data)) * size;
        ifs.read(reinterpret_cast<char*>(data), bytes);
        if (ifs.fail())
            throw std::runtime_error("Failed to read file " + filename);
    };

    std::int32_t nnz{};
    std::int32_t m{};
    std::int32_t n{};
    read(&nnz);
    read(&m);
    read(&n);
    CSR<T, L> csr(nnz, m, n);
    read(csr.rows_ptr, csr.m + 1);
    read(csr.cols, csr.nnz);
    read(csr.values, csr.nnz);

    if (gsl::narrow_cast<unsigned long>(ifs.tellg()) < filesize)
        throw std::runtime_error("File " + filename + " is too large");

    return csr;
}

template<std::floating_point T, Location L>
auto CSR<T, L>::save_to_filename(const std::string& filename) const -> void
requires(L == Location::Host)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Failed to open file " + filename);

    auto write = [&ofs](const auto* data, std::int32_t size = 1) {
        const auto bytes = gsl::narrow_cast<std::int32_t>(sizeof(*data)) * size;
        ofs.write(reinterpret_cast<const char*>(data), bytes);
        if (ofs.fail())
            throw std::runtime_error("Failed to write to file");
    };

    write(&nnz);
    write(&m);
    write(&n);
    write(rows_ptr, m + 1);
    write(cols, nnz);
    write(values, nnz);
}

template<std::floating_point T, Location L>
template<Location To>
auto CSR<T, L>::to() const -> CSR<T, To> {
    static_assert(To != L, "Cannot convert to the same location");

    CSR<T, To> to(nnz, m, n);

    const auto direction = (To == Location::Device) ? cudaMemcpyHostToDevice
                                                    : cudaMemcpyDeviceToHost;

    copy(to.rows_ptr, rows_ptr, m + 1, sizeof(std::int32_t), direction);
    copy(to.cols, cols, nnz, sizeof(std::int32_t), direction);
    copy(to.values, values, nnz, sizeof(T), direction);

    return to;
}

template<std::floating_point T, Location L>
auto swap(CSR<T, L>& lhs, CSR<T, L>& rhs) noexcept -> void {
    using std::swap;
    swap(lhs.nnz, rhs.nnz);
    swap(lhs.m, rhs.m);
    swap(lhs.n, rhs.n);
    swap(lhs.rows_ptr, rhs.rows_ptr);
    swap(lhs.cols, rhs.cols);
    swap(lhs.values, rhs.values);
}

template<std::floating_point T, Location L>
auto CSR<T, L>::release() -> void {
    free(rows_ptr);
    free(cols);
    free(values);
    nnz = 0;
    m = 0;
    n = 0;
    rows_ptr = nullptr;
    cols = nullptr;
    values = nullptr;
}

template<std::floating_point T, Location L>
auto CSR<T, L>::allocate(std::int32_t count, std::size_t size) -> void* {
    void* ptr{};
    if constexpr (L == Location::Host) {
        handle_cuda_error(cudaMallocHost(&ptr, count * size));
    } else {
        handle_cuda_error(cudaMalloc(&ptr, count * size));
    }
    return ptr;
}

template<std::floating_point T, Location L>
auto CSR<T, L>::free(void* ptr) -> void {
    if constexpr (L == Location::Host) {
        handle_cuda_error(cudaFreeHost(ptr));
    } else {
        handle_cuda_error(cudaFree(ptr));
    }
}

template<std::floating_point T, Location L>
auto CSR<T, L>::copy(void* dst,
                     const void* src,
                     std::int32_t count,
                     std::size_t size,
                     cudaMemcpyKind kind) -> void {
    handle_cuda_error(cudaMemcpy(dst, src, count * size, kind));
}

} // namespace utils
