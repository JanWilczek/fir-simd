cmake_minimum_required(VERSION 3.10)

project(FIR_SIMD)

set(SOURCES
    src/FIRFilter.cpp
    src/main.cpp
    src/test.cpp
)

add_executable(FIR_SIMD ${SOURCES})

target_include_directories(FIR_SIMD 
    PUBLIC dependencies
    PUBLIC include)

if (MSVC)
    target_compile_options(FIR_SIMD 
                PRIVATE /EHsc
                PRIVATE /arch:AVX)
endif()

set_property(TARGET FIR_SIMD PROPERTY CXX_STANDARD 20)

configure_file(include/data/saw.wav ${CMAKE_CURRENT_BINARY_DIR}/saw.wav COPYONLY)
configure_file(include/data/classroomImpulseResponse.wav ${CMAKE_CURRENT_BINARY_DIR}/classroomImpulseResponse.wav COPYONLY)
