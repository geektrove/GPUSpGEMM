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

    CSR(const std::int32_t nnz_, const std::int32_t m_, const std::int32_t n_)
        : nnz{nnz_}, m{m_}, n{n_} {
        const auto bytes = get_csr_byte_size<T>(nnz_, m_);
        void* ptr{};
        if constexpr (L == Location::Host) {
            CHECK_CUDA(cudaMallocHost(&ptr, bytes));
        } else {
            CHECK_CUDA(cudaMalloc(&ptr, bytes));
        }
        rows_ptr = static_cast<std::int32_t*>(ptr);
        cols = reinterpret_cast<std::int32_t*>(rows_ptr + m_ + 1);
        values = reinterpret_cast<T*>(cols + nnz_);
    }

    static auto load_from_filename(const std::string& filename) -> CSR<T, L>
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

    auto save_to_filename(const std::string& filename) const -> void
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

    template<Location To>
    auto to() const -> CSR<T, To> {
        static_assert(To != L, "Cannot convert to the same location");
        CSR<T, To> csr(nnz, m, n);
        const auto bytes = get_csr_byte_size<T>(nnz, m);
        if constexpr (To == Location::Device) {
            CHECK_CUDA(cudaMemcpy(csr.rows_ptr, rows_ptr, bytes, cudaMemcpyHostToDevice));
        } else {
            CHECK_CUDA(cudaMemcpy(csr.rows_ptr, rows_ptr, bytes, cudaMemcpyDeviceToHost));
        }
        return csr;
    }

    auto free() -> void {
        if constexpr (L == Location::Host) {
            CHECK_CUDA(cudaFreeHost(rows_ptr));
        } else {
            CHECK_CUDA(cudaFree(rows_ptr));
        }
        nnz = 0;
        m = 0;
        n = 0;
        rows_ptr = nullptr;
        cols = nullptr;
        values = nullptr;
    }
};
} // namespace utils
