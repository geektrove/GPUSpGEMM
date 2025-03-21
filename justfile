#!/usr/bin/env -S just --justfile

preset := "dev-clang"
build_path := "build/" + preset
benchmark_options := "\
    --benchmark_min_warmup_time=3 \
    --benchmark_min_time=10x \
    --benchmark_repetitions=1 \
    --benchmark_time_unit=ms"

default:
    just --list

configure:
    cmake --preset={{ preset }}

compdb:
    compdb -p {{ build_path }} list >compile_commands.json

refresh: configure compdb

build:
    cmake --build --preset={{ preset }}

format:
    just --fmt --unstable
    git ls-files -- '*.hpp' '*.cpp' | xargs clang-format -i
    git ls-files -- '*.cuh' '*.cu' | xargs clang-format -i
    git ls-files -- '**/CMakeLists.txt' '*.cmake' | xargs cmake-format -i
    git ls-files -- '*.yaml' '.clang-format' '.clang-tidy' | xargs yamlfmt
    git ls-files -- '*.json' | xargs jsonfmt -w
    git ls-files -- '*.py' | xargs ruff format --cache-dir=.cache/ruff >/dev/null
    git ls-files -- '*.md' | xargs mdformat

benchmark inputA="data/csr/webbase-1M.csr" inputB="data/csr/webbase-1M.csr": build
    "{{ build_path }}/src/apps/cusparse_benchmark" "{{ inputA }}" "{{ inputB }}" {{ benchmark_options }}
    "{{ build_path }}/src/apps/opsparse_benchmark" "{{ inputA }}" "{{ inputB }}" {{ benchmark_options }}
