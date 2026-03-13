#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-examples}"

JOBS="${JOBS:-2}"
CUDA_ARCHS="${CUDA_ARCHS:-86}"
EXAMPLE_TARGET="${EXAMPLE_TARGET:-00_basic_gemm}"

while getopts ":t:a:j:" opt; do
  case "${opt}" in
    t)
      EXAMPLE_TARGET="${OPTARG}"
      ;;
    a)
      CUDA_ARCHS="${OPTARG}"
      ;;
    j)
      JOBS="${OPTARG}"
      ;;
    :)
      echo "option -${OPTARG} requires an argument" >&2
      exit 1
      ;;
    \?)
      echo "unknown option: -${OPTARG}" >&2
      exit 1
      ;;
  esac
done

EXAMPLE_BINARY="${BUILD_DIR}/examples/${EXAMPLE_TARGET}/${EXAMPLE_TARGET}"

echo "jobs=${JOBS} arch=${CUDA_ARCHS} target=${EXAMPLE_TARGET}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_ENABLE_LIBRARY=OFF \
  -DCUTLASS_NVCC_ARCHS="${CUDA_ARCHS}" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CUDA_HOST_COMPILER=clang++ \
  -G Ninja

  # -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  # -DCMAKE_INSTALL_PREFIX="${ROOT_DIR}" \

# using bear generates compile_commands.json
bear -- ninja -C "${BUILD_DIR}" "${EXAMPLE_TARGET}" -j"${JOBS}" -v
cp "${EXAMPLE_BINARY}" "${ROOT_DIR}"
