#!/usr/bin/env just --justfile

preset := "dev-clang"

default:
    just --list

clean:
    rm -rf build/

configure:
    cmake \
    --preset={{ preset }} \

build:
    cmake --build --preset={{ preset }}

compdb:
    compdb -p build/{{ preset }} list >compile_commands.json

format:
    just --fmt --unstable
    git ls-files -- '*.hpp' '*.cpp' | xargs clang-format -i
    git ls-files -- '*.cuh' '*.cu' | xargs clang-format -i
    git ls-files -- '**/CMakeLists.txt' '*.cmake' | xargs cmake-format -i
    git ls-files -- '*.yaml' '.clang-format' '.clang-tidy' | xargs yamlfmt
    git ls-files -- '*.json' | xargs jsonfmt -w
    git ls-files -- '*.py' | xargs ruff format --cache-dir=.cache/ruff >/dev/null
    git ls-files -- '*.md' | xargs mdformat
