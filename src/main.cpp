#include <iostream>

#include "FIRFilter.h"
#include "benchmark.h"
#include "test.h"

using namespace fir;

void test() {
  std::cout << "#------------- FIR filter single --------------------#"
            << std::endl;
  testFirFilterBigRandomVectors<alignof(float)>(applyFirFilterInnerLoopVectorization);

  std::cout
      << "#------------- FIR filter AVX --------------------#" << std::endl
      << "#------------- Inner Loop Vectorization --------------------#"
      << std::endl;
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterInnerLoopVectorization);
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterAVX_innerLoopVectorization);
  testFirFilterImpulseResponses<alignof(float)>(
      applyFirFilterAVX_innerLoopVectorization);

  std::cout
      << "#------------- Outer Loop Vectorization --------------------#"
      << std::endl;
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterOuterLoopVectorization);
  testFirFilterBigRandomVectors<AVX_FLOAT_COUNT * alignof(float)>(
      applyFirFilterAVX_outerLoopVectorization);
  testFirFilterImpulseResponses<AVX_FLOAT_COUNT * alignof(float)>(
      applyFirFilterAVX_outerLoopVectorization);

  std::cout << "#------------- Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterOuterInnerLoopVectorization);
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterAVX_outerInnerLoopVectorization);
  testFirFilterImpulseResponses<alignof(float)>(
      applyFirFilterAVX_outerInnerLoopVectorization);

  std::cout << "#------------- Aligned Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  testFirFilterImpulseResponses<AVX_FLOAT_COUNT * alignof(float)>(
      applyFirFilterAVX_outerInnerLoopVectorizationAligned);
}

void benchmark() {
  std::cout << "#------------- FIR filter single --------------------#"
            << std::endl;
  benchmarkFirFilterBigRandomVectors<alignof(float)>(applyFirFilterSingle);

  std::cout
      << "#------------- FIR filter AVX --------------------#" << std::endl
      << "#------------- Inner Loop Vectorization --------------------#"
      << std::endl;
  benchmarkFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterAVX_innerLoopVectorization);

  std::cout
      << "#------------- Outer Loop Vectorization --------------------#"
      << std::endl;
  benchmarkFirFilterBigRandomVectors<fir::AVX_FLOAT_COUNT * alignof(float)>(
      applyFirFilterAVX_outerLoopVectorization);

  std::cout << "#------------- Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  benchmarkFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterAVX_outerInnerLoopVectorization);

  std::cout << "#------------- Aligned Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  benchmarkFirFilterBigRandomVectors<fir::AVX_FLOAT_COUNT * alignof(float)>(
      applyFirFilterAVX_outerInnerLoopVectorizationAligned);
}

int main() {
  test();
  benchmark();
  std::cout << "Success!" << std::endl;
}
