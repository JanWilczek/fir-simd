#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>
#include <cmath>
#include "assert.h"
#include "data/BigRandomVectors.h"

#ifdef __AVX__
#include <immintrin.h>
#endif


std::vector<float> applyFirFilterSingle(const std::vector<float>& signal, const std::vector<float>& impulseResponse);

template<typename T>
T highestPowerOf2NotGreaterThan(T x) {
    using namespace std;
    return static_cast<T>(pow(2., floor(log2(static_cast<double>(x)))));
}

template<typename T, T N>
T highestMultipleOfNIn(T x) {
    return static_cast<long long>(x / N);
}

void testFirFilter(std::function<std::vector<float>(const std::vector<float>&,const std::vector<float>&)> filteringFunction) {
    std::vector<float> signal{1.f, 2.f, 3.f, 4.f,};
    std::vector<float> ir{1.f};

    const auto filtered = filteringFunction(signal, ir);

    assert(filtered == signal);

    const auto filtered2 = filteringFunction(signal, {0.f, 1.f});

    assert(filtered2[0] == 0.f);
    assert(filtered2[1] == 1.f);
    assert(filtered2[2] == 2.f);
    assert(filtered2[3] == 3.f);
    assert(filtered2[4] == 4.f);
    assert(filtered2.size() == 5u);

    std::vector<float> ir3{ 0.4, 0.2, 0.4, -0.1, -0.4, -0.3, -0.5, -0.11, -0.3};
    const auto expected = std::vector<float>{0.4f,  1.f,  2.f,  2.9f,  1.4f,  0.2f, -2.7f, -3.61f, -3.22f,
       -2.93f, -1.34f, -1.2f};
    
    const auto filtered3 = filteringFunction(signal, ir3);
    for (auto i = 0u; i < filtered3.size(); ++i) {
        assert(std::abs(filtered3[i] - expected[i]) < 1e-6f);
    }
}

void testFirFilterBigRandomVectors(std::function<std::vector<float>(const std::vector<float>&,const std::vector<float>&)> filteringFunction) {
    const auto expected = applyFirFilterSingle(random1, random2);
    const auto given = filteringFunction(random1, random2);

    assert(expected.size() == given.size());
    for (auto i = 0u; i < expected.size(); ++i) {
        assert(std::abs(expected[i] - given[i]) < 1e-6f);
    }
}

std::vector<float> applyFirFilterSingle(const std::vector<float>& signal, const std::vector<float>& impulseResponse) {
    std::vector<float> output(signal.size() + impulseResponse.size() - 1u);

    for (int i = 0; i < static_cast<int>(output.size()); ++i) {
        output[i] = 0.f;
        const auto startId = std::max(0, i - static_cast<int>(impulseResponse.size()) + 1);
        const auto endId = std::min(i, static_cast<int>(signal.size()) - 1);
        for (auto j = startId; j <= endId; ++j) {
            output[i] += signal[j] * impulseResponse[i - j];
        }
    }
    return output;
}
#define __AVX__
#ifdef __AVX__
std::vector<float> applyFirFilterAVX(const std::vector<float>& signal, const std::vector<float>& impulseResponse) {
    std::vector<float> output(signal.size() + impulseResponse.size() - 1u);

    
    
    const auto& longer = signal.size() >= impulseResponse.size() ? signal : impulseResponse;
    const auto& shorter = signal.size() < impulseResponse.size() ? signal : impulseResponse;

    constexpr auto AVX_FLOAT_COUNT = 256 / 32;
    const auto vectorizableLength = highestMultipleOfNIn<AVX_FLOAT_COUNT>(longer.size()) * AVX_FLOAT_COUNT;

    std::vector<float> irChunk(AVX_FLOAT_COUNT, 0.f);

    for (auto i = 0; i < vectorizableLength; i += AVX_FLOAT_COUNT) {
        const auto startId = std::max(0, i - static_cast<int>(shorter.size()) + 1);
        const auto endId = std::min(i, static_cast<int>(longer.size()) - 1);

        auto temp = _mm256_setzero_ps();


        // std::copy
        
        auto a = _mm256_loadu_ps(longer);
    }

}
#endif

std::vector<float> applyFirFilter(const std::vector<float>& signal, const std::vector<float>& impulseResponse) {
#if __AVX__
    return applyFirFilterAVX(signal, impulseResponse);
#else
    return applyFirFilterSingle(signal, impulseResponse);
#endif
}

int main() {
    testFirFilter(applyFirFilter);
    // testFirFilterBigRandomVectors(applyFirFilter);

    std::cout << "Success!" << std::endl;
}