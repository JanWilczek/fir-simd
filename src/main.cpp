#include <iostream>

#include "FIRFilter.h"
#include "benchmark.h"
#include "test.h"

int main() {
  // testFirFilter(applyFirFilter);
  // testFirFilterBigRandomVectors(applyFirFilter);

  std::cout << "#------------- FIR filter single --------------------#"
            << std::endl;
  // benchmarkFirFilterImpulseResponses(applyFirFilterSingle);
  benchmarkFirFilterBigRandomVectors(applyFirFilterSingle, 1u);
  std::cout << "#------------- FIR filter AVX --------------------#"
            << std::endl
            << "#------------- Inner Loop Vectorization --------------------#"
            << std::endl;
  // benchmarkFirFilterImpulseResponses(applyFirFilterAVX);
  benchmarkFirFilterBigRandomVectors(applyFirFilterAVX_innerLoopVectorization,
                                     AVX_FLOAT_COUNT);
  std::cout
      << "#------------- Outer-Inner Loop Vectorization --------------------#"
      << std::endl;
  benchmarkFirFilterBigRandomVectors(
      applyFirFilterAVX_outerInnerLoopVectorization,
                                     AVX_FLOAT_COUNT);
  testFirFilterImpulseResponses(applyFirFilterAVX_outerInnerLoopVectorization,
                                AVX_FLOAT_COUNT);

  std::cout << "Success!" << std::endl;
}
