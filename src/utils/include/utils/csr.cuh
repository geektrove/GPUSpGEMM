#pragma once

#include <concepts>
#include <cstddef>
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

template<std::floating_point T>
auto get_csr_byte_size(const std::int32_t nnz, const std::int32_t m) -> std::size_t {
    return (gsl::narrow_cast<std::size_t>(m + 1) * sizeof(std::int32_t))
           + (gsl::narrow_cast<std::size_t>(nnz) * (sizeof(std::int32_t) + sizeof(T)));
}

template<std::floating_point T, Location L>
struct CSR {
    std::int32_t nnz{};
    std::int32_t m{};
    std::int32_t n{};
    std::int32_t* rows_ptr{};
    std::int32_t* cols{};
    T* values{};

    CSR(std::int32_t nnz_, std::int32_t m_, std::int32_t n_);
    CSR(const CSR& /*other*/);
    CSR(CSR&& /*other*/) noexcept;
    auto operator=(const CSR& /*other*/) -> CSR&;
    auto operator=(CSR&& /*other*/) noexcept -> CSR&;
    ~CSR();

    static auto load_from_filename(const std::string& filename) -> CSR<T, L>
    requires(L == Location::Host);

    auto save_to_filename(const std::string& filename) const -> void
    requires(L == Location::Host);

    template<Location To>
    auto to() const -> CSR<T, To>;

    template<std::floating_point T_, Location L_>
    friend auto swap(CSR<T_, L_>&, CSR<T_, L_>) noexcept -> void;

    auto free() -> void;
};

template<std::floating_point T, Location L>
CSR<T, L>::CSR(const std::int32_t nnz_, const std::int32_t m_, const std::int32_t n_)
    : nnz{nnz_}, m{m_}, n{n_} {
    const auto bytes = get_csr_byte_size<T>(nnz_, m_);
    void* ptr{};
    if constexpr (L == Location::Host) {
        handle_cuda_error(cudaMallocHost(&ptr, bytes));
    } else {
        handle_cuda_error(cudaMalloc(&ptr, bytes));
    }
    rows_ptr = static_cast<std::int32_t*>(ptr);
    cols = rows_ptr + m_ + 1;
    values = reinterpret_cast<T*>(cols + nnz_);
}

template<std::floating_point T, Location L>
CSR<T, L>::CSR(const CSR& other) : nnz{other.nnz}, m{other.m}, n{other.n} {
    const auto bytes = get_csr_byte_size<T>(nnz, m);
    void* ptr{};
    if constexpr (L == Location::Host) {
        handle_cuda_error(cudaMallocHost(&ptr, bytes));
    } else {
        handle_cuda_error(cudaMalloc(&ptr, bytes));
    }
    rows_ptr = static_cast<std::int32_t*>(ptr);
    cols = rows_ptr + m + 1;
    values = reinterpret_cast<T*>(cols + nnz);
    if constexpr (L == Location::Host)
        handle_cuda_error(
            cudaMemcpy(rows_ptr, other.rows_ptr, bytes, cudaMemcpyHostToHost));
    else
        handle_cuda_error(
            cudaMemcpy(rows_ptr, other.rows_ptr, bytes, cudaMemcpyDeviceToDevice));
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
    nnz = std::exchange(other.nnz, 0);
    m = std::exchange(other.m, 0);
    n = std::exchange(other.n, 0);
    rows_ptr = std::exchange(other.rows_ptr, nullptr);
    cols = std::exchange(other.cols, nullptr);
    values = std::exchange(other.values, nullptr);
    return *this;
}

template<std::floating_point T, Location L>
CSR<T, L>::~CSR() {
    free();
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
    CSR<T, To> csr(nnz, m, n);
    const auto bytes = get_csr_byte_size<T>(nnz, m);
    if constexpr (To == Location::Device) {
        handle_cuda_error(
            cudaMemcpy(csr.rows_ptr, rows_ptr, bytes, cudaMemcpyHostToDevice));
    } else {
        handle_cuda_error(
            cudaMemcpy(csr.rows_ptr, rows_ptr, bytes, cudaMemcpyDeviceToHost));
    }
    return csr;
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
auto CSR<T, L>::free() -> void {
    if constexpr (L == Location::Host) {
        handle_cuda_error(cudaFreeHost(rows_ptr));
    } else {
        handle_cuda_error(cudaFree(rows_ptr));
    }
    nnz = 0;
    m = 0;
    n = 0;
    rows_ptr = nullptr;
    cols = nullptr;
    values = nullptr;
}

} // namespace utils
