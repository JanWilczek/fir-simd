cmake_minimum_required(VERSION 3.10)

project(FIR_SIMD)

add_executable(FIR_SIMD src/FIRFilter.cpp src/main.cpp src/test.cpp)

target_include_directories(FIR_SIMD 
PUBLIC "${PROJECT_SOURCE_DIR}/dependencies"
PUBLIC "${PROJECT_SOURCE_DIR}/include")

# target_compile_options(FIR_SIMD PRIVATE /arch:AVX)
