#pragma once

#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>

#include <gsl/gsl-lite.hpp>

#include <utils/errors.cuh>
#include <utils/location.cuh>

namespace utils {

template<std::floating_point T, Location L>
struct CSR {
    std::int32_t nnz{};
    std::int32_t m{};
    std::int32_t n{};
    std::int32_t* rpt{};
    std::int32_t* col{};
    T* val{};

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
    friend auto swap(CSR<T_, L_>&, CSR<T_, L_>&) noexcept -> void;

    auto release() -> void;

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
      rpt{static_cast<std::int32_t*>(allocate(m + 1, sizeof(std::int32_t)))},
      col{static_cast<std::int32_t*>(allocate(nnz, sizeof(std::int32_t)))},
      val{static_cast<T*>(allocate(nnz, sizeof(T)))} {}

template<std::floating_point T, Location L>
CSR<T, L>::CSR(const CSR& other)
    : nnz{other.nnz},
      m{other.m},
      n{other.n},
      rpt{static_cast<std::int32_t*>(allocate(m + 1, sizeof(std::int32_t)))},
      col{static_cast<std::int32_t*>(allocate(nnz, sizeof(std::int32_t)))},
      val{static_cast<T*>(allocate(nnz, sizeof(T)))} {
    const auto direction = (L == Location::Host) ? cudaMemcpyHostToHost
                                                 : cudaMemcpyDeviceToDevice;

    copy(rpt, other.rpt, m + 1, sizeof(std::int32_t), direction);
    copy(col, other.col, nnz, sizeof(std::int32_t), direction);
    copy(val, other.val, nnz, sizeof(T), direction);
}

template<std::floating_point T, Location L>
CSR<T, L>::CSR(CSR&& other) noexcept
    : nnz{std::exchange(other.nnz, 0)},
      m{std::exchange(other.m, 0)},
      n{std::exchange(other.n, 0)},
      rpt{std::exchange(other.rpt, nullptr)},
      col{std::exchange(other.col, nullptr)},
      val{std::exchange(other.val, nullptr)} {}

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
    read(csr.rpt, csr.m + 1);
    read(csr.col, csr.nnz);
    read(csr.val, csr.nnz);

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
    write(rpt, m + 1);
    write(col, nnz);
    write(val, nnz);
}

template<std::floating_point T, Location L>
template<Location To>
auto CSR<T, L>::to() const -> CSR<T, To> {
    static_assert(To != L, "Cannot convert to the same location");

    CSR<T, To> to(nnz, m, n);

    const auto direction = (To == Location::Device) ? cudaMemcpyHostToDevice
                                                    : cudaMemcpyDeviceToHost;

    copy(to.rpt, rpt, m + 1, sizeof(std::int32_t), direction);
    copy(to.col, col, nnz, sizeof(std::int32_t), direction);
    copy(to.val, val, nnz, sizeof(T), direction);

    return to;
}

template<std::floating_point T, Location L>
inline auto swap(CSR<T, L>& lhs, CSR<T, L>& rhs) noexcept -> void {
    if (&lhs == &rhs)
        return;
    using std::swap;
    swap(lhs.nnz, rhs.nnz);
    swap(lhs.m, rhs.m);
    swap(lhs.n, rhs.n);
    swap(lhs.rpt, rhs.rpt);
    swap(lhs.col, rhs.col);
    swap(lhs.val, rhs.val);
}

template<std::floating_point T, Location L>
auto CSR<T, L>::release() -> void {
    free(rpt);
    free(col);
    free(val);
    nnz = 0;
    m = 0;
    n = 0;
    rpt = nullptr;
    col = nullptr;
    val = nullptr;
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

template<std::floating_point T>
using HostCSR = utils::CSR<T, utils::Location::Host>;
template<std::floating_point T>
using DeviceCSR = utils::CSR<T, utils::Location::Device>;

} // namespace utils
