#include <iostream>

#include "FIRFilter.h"
//#include "benchmark.h"
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

  std::cout
      << "#------------- Outer Loop Vectorization --------------------#"
      << std::endl;
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterOuterLoopVectorization);
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterAVX_outerLoopVectorization);

  std::cout << "#------------- Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterOuterInnerLoopVectorization);
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterAVX_outerInnerLoopVectorization);

  std::cout << "#------------- Aligned Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  testFirFilterImpulseResponses<AVX_FLOAT_COUNT * alignof(float)>(
      applyFirFilterAVX_outerInnerLoopVectorizationAligned);
}

void benchmark() {
  //std::cout << "#------------- FIR filter single --------------------#"
  //          << std::endl;
  //benchmarkFirFilterBigRandomVectors(applyFirFilterSingle);

  //std::cout
  //    << "#------------- FIR filter AVX --------------------#" << std::endl
  //    << "#------------- Inner Loop Vectorization --------------------#"
  //    << std::endl;
  //benchmarkFirFilterBigRandomVectors(
  //    applyFirFilterAVX_innerLoopVectorization);

  //std::cout
  //    << "#------------- Outer Loop Vectorization --------------------#"
  //    << std::endl;
  //benchmarkFirFilterBigRandomVectors(
  //    applyFirFilterAVX_outerLoopVectorization);

  //std::cout << "#------------- Outer-Inner Loop Vectorization "
  //             "--------------------#"
  //          << std::endl;
  //benchmarkFirFilterBigRandomVectors(
  //    applyFirFilterAVX_outerInnerLoopVectorization);

  //std::cout << "#------------- Aligned Outer-Inner Loop Vectorization "
  //             "--------------------#"
  //          << std::endl;
  //benchmarkFirFilterBigRandomVectors(
  //    applyFirFilterAVX_outerInnerLoopVectorizationAligned, 8u);
}

int main() {
  test();
  //benchmark();
  std::cout << "Success!" << std::endl;
}
