cmake_minimum_required(VERSION 3.10)

project(FIR_SIMD)

add_executable(FIR_SIMD FIRFilter.cpp)

# target_compile_options(FIR_SIMD PRIVATE /arch:AVX)
