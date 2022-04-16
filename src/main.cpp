#include <iostream>

#include "FIRFilter.h"
#include "benchmark.h"
#include "test.h"

using namespace fir;

void test() {
  //std::cout << "#------------- FIR filter single --------------------#"
  //          << std::endl;
  //testFirFilterBigRandomVectors(applyFirFilterInnerLoopVectorization, 1u);

  //std::cout
  //    << "#------------- FIR filter AVX --------------------#" << std::endl
  //    << "#------------- Inner Loop Vectorization --------------------#"
  //    << std::endl;
  //testFirFilterBigRandomVectors(applyFirFilterInnerLoopVectorization, 1u);
  //testFirFilterBigRandomVectors(applyFirFilterAVX_innerLoopVectorization,
  //                              1u);

  //std::cout
  //    << "#------------- Outer Loop Vectorization --------------------#"
  //    << std::endl;
  //testFirFilterBigRandomVectors(applyFirFilterOuterLoopVectorization, 1u);
  //testFirFilterBigRandomVectors(applyFirFilterAVX_outerLoopVectorization,
  //                              1u);
  //benchmarkFirFilterBigRandomVectors(
  //    applyFirFilterAVX_outerLoopVectorization, 1u);

  std::cout << "#------------- Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  testFirFilterBigRandomVectors(applyFirFilterOuterInnerLoopVectorization,
                                1u);
  testFirFilterBigRandomVectors(
      applyFirFilterAVX_outerInnerLoopVectorization, 1u);

  std::cout << "#------------- Aligned Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  testFirFilterImpulseResponses(
      applyFirFilterAVX_outerInnerLoopVectorizationAligned, 1u);
}

void benchmark() {
  //std::cout << "#------------- FIR filter single --------------------#"
  //          << std::endl;
  //benchmarkFirFilterBigRandomVectors(applyFirFilterSingle, 1u);

  //std::cout
  //    << "#------------- FIR filter AVX --------------------#" << std::endl
  //    << "#------------- Inner Loop Vectorization --------------------#"
  //    << std::endl;
  //benchmarkFirFilterBigRandomVectors(
  //    applyFirFilterAVX_innerLoopVectorization, 1u);

  //std::cout
  //    << "#------------- Outer Loop Vectorization --------------------#"
  //    << std::endl;
  //benchmarkFirFilterBigRandomVectors(
  //    applyFirFilterAVX_outerLoopVectorization, 1u);

  std::cout << "#------------- Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  benchmarkFirFilterBigRandomVectors(
      applyFirFilterAVX_outerInnerLoopVectorization, 1u);

  std::cout << "#------------- Aligned Outer-Inner Loop Vectorization "
               "--------------------#"
            << std::endl;
  benchmarkFirFilterBigRandomVectors(
      applyFirFilterAVX_outerInnerLoopVectorizationAligned, 8u);
}

int main() {
  test();
  benchmark();
  std::cout << "Success!" << std::endl;
}
