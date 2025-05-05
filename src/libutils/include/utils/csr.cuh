#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <type_traits>

#include <gsl/gsl-lite.hpp>

#include <utils/errors.cuh>
#include <utils/location.cuh>
#include <utils/runtime.cuh>

namespace utils {

// Compressed Sparse Row (CSR) matrix representation
// Template parameters:
// - T: The floating-point data type (float or double)
// - L: Location specifier (Host or Device) indicating where the matrix data is stored
template<std::floating_point T, Location L>
struct CSR {
    std::int32_t nnz{};  // Number of non-zero elements
    std::int32_t m{};    // Number of rows
    std::int32_t n{};    // Number of columns
    std::int32_t* rpt{}; // Row pointers (array of size m+1)
    std::int32_t* col{}; // Column indices (array of size nnz)
    T* val{};            // Values (array of size nnz)

    CSR() = default;
    CSR(std::int32_t nnz_, std::int32_t m_, std::int32_t n_);
    CSR(const CSR& other);
    CSR(CSR&& other) noexcept;
    auto operator=(const CSR& other) -> CSR&;
    auto operator=(CSR&& other) noexcept -> CSR&;
    ~CSR();

    // Load a CSR matrix from a binary file
    static auto load_from_filename(const std::string& filename) -> CSR<T, L>
    requires(L == Location::Host);

    // Save a CSR matrix to a binary file
    auto save_to_filename(const std::string& filename) const -> void
    requires(L == Location::Host);

    // Convert matrix between Host and Device
    template<Location To>
    auto to() const -> CSR<T, To>;

    // Swap the contents of two CSR matrices
    template<std::floating_point T_, Location L_>
    friend auto swap(CSR<T_, L_>&, CSR<T_, L_>&) noexcept -> void;

    // Compare two CSR matrices for equality
    template<std::floating_point T_, Location L_>
    friend auto operator==(const CSR<T_, L_>&, const CSR<T_, L_>&) noexcept -> bool
    requires(L_ == Location::Host);

    // Release all allocated resources
    auto release() -> void;
};

template<std::floating_point T, Location L>
CSR<T, L>::CSR(const std::int32_t nnz_, const std::int32_t m_, const std::int32_t n_)
    : nnz{nnz_},
      m{m_},
      n{n_},
      rpt{static_cast<std::int32_t*>(malloc<L>((m + 1) * sizeof(std::int32_t)))},
      col{static_cast<std::int32_t*>(malloc<L>(nnz * sizeof(std::int32_t)))},
      val{static_cast<T*>(malloc<L>(nnz * sizeof(T)))} {}

template<std::floating_point T, Location L>
CSR<T, L>::CSR(const CSR& other)
    : nnz{other.nnz},
      m{other.m},
      n{other.n},
      rpt{static_cast<std::int32_t*>(malloc<L>((m + 1) * sizeof(std::int32_t)))},
      col{static_cast<std::int32_t*>(malloc<L>(nnz * sizeof(std::int32_t)))},
      val{static_cast<T*>(malloc<L>(nnz * sizeof(T)))} {
    memcpy(rpt, other.rpt, (m + 1) * sizeof(std::int32_t));
    memcpy(col, other.col, nnz * sizeof(std::int32_t));
    memcpy(val, other.val, nnz * sizeof(T));
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
    if constexpr (std::is_same_v<T, double>) {
        read(csr.val, csr.nnz);
    } else {
        // Data are always read in double precision
        auto* tmp = static_cast<double*>(malloc<L>(nnz * sizeof(double)));
        read(tmp, nnz);
        std::copy(tmp, tmp + nnz, csr.val);
        free<L>(tmp);
    }

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
        const auto bytes = sizeof(*data) * size;
        ofs.write(reinterpret_cast<const char*>(data), bytes);
        if (ofs.fail())
            throw std::runtime_error("Failed to write to file");
    };

    write(&nnz);
    write(&m);
    write(&n);
    write(rpt, m + 1);
    write(col, nnz);
    if constexpr (std::is_same_v<T, double>) {
        write(val, nnz);
    } else {
        // Data are always written in double precision
        auto* tmp = static_cast<double*>(malloc<L>(nnz * sizeof(double)));
        std::copy(this->val, this->val + nnz, tmp);
        write(tmp, nnz);
        free<L>(tmp);
    }
}

template<std::floating_point T, Location L>
template<Location To>
auto CSR<T, L>::to() const -> CSR<T, To> {
    static_assert(To != L, "Cannot convert to the same location");

    CSR<T, To> to(nnz, m, n);
    memcpy(to.rpt, rpt, (m + 1) * sizeof(std::int32_t));
    memcpy(to.col, col, nnz * sizeof(std::int32_t));
    memcpy(to.val, val, nnz * sizeof(T));
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
inline auto operator==(const CSR<T, L>& lhs, const CSR<T, L>& rhs) noexcept -> bool
requires(L == Location::Host)
{
    if (&lhs == &rhs)
        return true;
    if (lhs.nnz != rhs.nnz || lhs.m != rhs.m || lhs.n != rhs.n)
        return false;
    if (!std::equal(lhs.rpt, lhs.rpt + lhs.m + 1, rhs.rpt))
        return false;
    if (!std::equal(lhs.col, lhs.col + lhs.nnz, rhs.col))
        return false;

    // Check values match (with tolerance for floating-point comparison)
    return std::equal(lhs.val, lhs.val + lhs.nnz, rhs.val, [](const T& l, const T& r) {
        static constexpr T atol = 1e-8; // Absolute tolerance
        static constexpr T rtol = 1e-5; // Relative tolerance
        if (l == r)
            return true;
        return std::fabs(l - r) <= (atol + rtol * std::max(std::fabs(l), std::fabs(r)));
    });
}

template<std::floating_point T, Location L>
auto CSR<T, L>::release() -> void {
    free<L>(rpt);
    free<L>(col);
    free<L>(val);
    nnz = 0;
    m = 0;
    n = 0;
    rpt = nullptr;
    col = nullptr;
    val = nullptr;
}

// Type aliases for common use cases
template<std::floating_point T>
using HostCSR = utils::CSR<T, utils::Location::Host>;
template<std::floating_point T>
using DeviceCSR = utils::CSR<T, utils::Location::Device>;

} // namespace utils
