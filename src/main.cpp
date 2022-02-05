#include <iostream>
#include "test.h"
#include "FIRFilter.h"

int main() {
    testFirFilter(applyFirFilter);
    // testFirFilterBigRandomVectors(applyFirFilter);
    testFirFilterImpulseResponses(applyFirFilter);

    std::cout << "Success!" << std::endl;
}
