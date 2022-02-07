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
  benchmarkFirFilterBigRandomVectors(applyFirFilterSingle);
  std::cout << "#------------- FIR filter AVX --------------------#"
            << std::endl;
  // benchmarkFirFilterImpulseResponses(applyFirFilterAVX);
  benchmarkFirFilterBigRandomVectors(applyFirFilterAVX);
  testFirFilterImpulseResponses(applyFirFilterAVX);

  std::cout << "Success!" << std::endl;
}
