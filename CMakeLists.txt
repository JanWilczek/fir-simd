cmake_minimum_required(VERSION 3.10)

project(FIR_SIMD)

add_executable(FIR_SIMD src/FIRFilter.cpp src/main.cpp src/test.cpp src/benchmark.cpp)

target_include_directories(FIR_SIMD 
PUBLIC "${PROJECT_SOURCE_DIR}/dependencies"
PUBLIC "${PROJECT_SOURCE_DIR}/include")

if (MSVC)
    target_compile_options(FIR_SIMD 
                PRIVATE /EHsc
                PRIVATE /arch:AVX)
endif()

set_property(TARGET FIR_SIMD PROPERTY CXX_STANDARD 20)
