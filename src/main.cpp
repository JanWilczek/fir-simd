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
            << "#------------- Inner Loop Vectorization --------------------#"
            << std::endl;
  // benchmarkFirFilterImpulseResponses(applyFirFilterAVX);
  benchmarkFirFilterBigRandomVectors(applyFirFilterAVX_innerLoopVectorization,
                                     AVX_FLOAT_COUNT);
  // testFirFilterImpulseResponses(applyFirFilterAVX, AVX_FLOAT_COUNT);

  std::cout << "Success!" << std::endl;
}
