#!/usr/bin/env just --justfile

default:
    just --list

clean:
    rm -rf build/
    rm -f CMakeConanPresets.json

conan profile="clang" type="release":
    conan install conanfile.py \
    --profile:all={{ profile }} \
    --settings=build_type={{ capitalize(type) }} \
    --build=missing

configure preset="dev-clang-release":
    cmake \
    --preset={{ preset }} \
    -DCMAKE_CUDA_ARCHITECTURES="86;80"

build preset="dev-clang-release":
    cmake --build --preset={{ preset }}

compdb build_dir="build/clang-release":
    compdb -p {{ build_dir }} list >compile_commands.json

format:
    just --fmt --unstable
    git ls-files -- '*.hpp' '*.cpp' | xargs clang-format -i
    git ls-files -- '*.cuh' '*.cu' | xargs clang-format -i
    git ls-files -- '**/CMakeLists.txt' '*.cmake' | xargs cmake-format -i
    git ls-files -- '*.yaml' '.clang-format' '.clang-tidy' | xargs yamlfmt
    git ls-files -- '*.json' | xargs jsonfmt -w
    git ls-files -- '*.py' | xargs ruff format --cache-dir=.cache/ruff >/dev/null
    git ls-files -- '*.md' | xargs mdformat
