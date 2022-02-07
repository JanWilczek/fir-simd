#include <iostream>

#include "FIRFilter.h"
#include "benchmark.h"
#include "test.h"

int main() {
  std::cout << "#------------- FIR filter single --------------------#"
            << std::endl;
  benchmarkFirFilterBigRandomVectors(applyFirFilterSingle, 1u);
  std::cout << "#------------- FIR filter AVX --------------------#"
            << std::endl
            << "#------------- Inner Loop Vectorization --------------------#"
            << std::endl;
  benchmarkFirFilterBigRandomVectors(applyFirFilterAVX_innerLoopVectorization,
                                     AVX_FLOAT_COUNT);
  benchmarkFirFilterImpulseResponses(applyFirFilterAVX_innerLoopVectorization,
                                     AVX_FLOAT_COUNT);
  std::cout
      << "#------------- Outer-Inner Loop Vectorization --------------------#"
      << std::endl;
  benchmarkFirFilterBigRandomVectors(
      applyFirFilterAVX_outerInnerLoopVectorization,
                                     AVX_FLOAT_COUNT);
  benchmarkFirFilterImpulseResponses(
      applyFirFilterAVX_outerInnerLoopVectorization, AVX_FLOAT_COUNT);

  std::cout << "Success!" << std::endl;
}
