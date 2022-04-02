#include <iostream>

#include "FIRFilter.h"
#include "benchmark.h"
#include "test.h"

using namespace fir;

int main() {
  std::cout << "#------------- FIR filter single --------------------#"
            << std::endl;
  testFirFilterBigRandomVectors(applyFirFilterInnerLoopVectorization, 1u);
  benchmarkFirFilterBigRandomVectors(applyFirFilterSingle, 1u);

  std::cout << "#------------- FIR filter AVX --------------------#"
            << std::endl
            << "#------------- Inner Loop Vectorization --------------------#"
            << std::endl;
  testFirFilterBigRandomVectors(applyFirFilterInnerLoopVectorization, 1u);
  testFirFilterBigRandomVectors(applyFirFilterAVX_innerLoopVectorization,
                                1u);
  benchmarkFirFilterBigRandomVectors(applyFirFilterAVX_innerLoopVectorization,
                                     1u);

  std::cout << "#------------- Outer Loop Vectorization --------------------#"
            << std::endl;
  testFirFilterBigRandomVectors(applyFirFilterOuterLoopVectorization, 1u);
  testFirFilterBigRandomVectors(applyFirFilterAVX_outerLoopVectorization, 1u);
  benchmarkFirFilterBigRandomVectors(
      applyFirFilterAVX_outerLoopVectorization, 1u);

  std::cout
      << "#------------- Outer-Inner Loop Vectorization --------------------#"
      << std::endl;
  testFirFilterBigRandomVectors(applyFirFilterOuterInnerLoopVectorization, 1u);
  testFirFilterBigRandomVectors(
      applyFirFilterAVX_outerInnerLoopVectorization, 1u);
  benchmarkFirFilterBigRandomVectors(
      applyFirFilterAVX_outerInnerLoopVectorization, 1u);

  //std::cout << "#------------- Aligned Outer-Inner Loop Vectorization "
               //"--------------------#"
            //<< std::endl;
  //testFirFilterBigRandomVectors(
      //applyFirFilterAVX_outerInnerLoopVectorizationAligned, 1u);
  //benchmarkFirFilterBigRandomVectors(
      //applyFirFilterAVX_outerInnerLoopVectorizationAligned, 1u);

  std::cout << "Success!" << std::endl;
}
