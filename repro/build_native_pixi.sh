#!/usr/bin/env bash
# Pixi port of build_native.sh: same targets and flags, conda-forge toolchain from $CONDA_PREFIX.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
: "${CONDA_PREFIX:?run through pixi}"
# Match the Docker build: only the project's own flags (drop conda's -march=nocona -O2 defaults).
unset CXXFLAGS CFLAGS CPPFLAGS DEBUG_CXXFLAGS
cmake -S slam_dunk/engine/native -B slam_dunk/engine/native/build -G Ninja \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
  -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS:-} -pthread" -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS:-} -pthread"
cmake --build slam_dunk/engine/native/build --target aria_vio track_union -j"${BUILD_JOBS:-8}"
"${CXX:-c++}" -std=c++17 -O3 -DNDEBUG -march=x86-64-v3 -Wno-deprecated-declarations \
  -DEIGEN_DONT_PARALLELIZE -DBASALT_INSTANTIATIONS_DOUBLE -DGLOG_USE_GLOG_EXPORT \
  -I"$CONDA_PREFIX/include/eigen3" -I"$CONDA_PREFIX/include" -Islam_dunk/engine/native \
  slam_dunk/engine/native/batch_joint.cpp \
  -o slam_dunk/engine/native/build/batch_joint \
  ${LDFLAGS:-} -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" -lceres -lglog -ltbb -pthread
