#include <iostream>
#include "test.h"
#include "FIRFilter.h"
#include "benchmark.h"

int main() {
    //testFirFilter(applyFirFilter);
    // testFirFilterBigRandomVectors(applyFirFilter);

    //testFirFilterImpulseResponses(applyFirFilter);

    std::cout << "#------------- FIR filter single --------------------#" << std::endl;
    benchmarkFirFilterImpulseResponses(applyFirFilterSingle);
    std::cout << "#------------- FIR filter AVX --------------------#" << std::endl;
    benchmarkFirFilterImpulseResponses(applyFirFilterAVX);

    std::cout << "Success!" << std::endl;
}
