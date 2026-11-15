#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Preserve the Full frontend's original build. Compile the controlled BA backend
# with the exact portable ISA/flags used for the paper's training factorial.
cmake -S slam_dunk/engine/native -B slam_dunk/engine/native/build -G Ninja
cmake --build slam_dunk/engine/native/build --target aria_vio track_union -j1
c++ -std=c++17 -O3 -DNDEBUG -march=x86-64-v3 -Wno-deprecated-declarations \
  -DEIGEN_DONT_PARALLELIZE -DBASALT_INSTANTIATIONS_DOUBLE \
  -I/usr/include/eigen3 -Islam_dunk/engine/native \
  slam_dunk/engine/native/batch_joint.cpp \
  -o slam_dunk/engine/native/build/batch_joint -lceres -lglog -ltbb -lpthread
