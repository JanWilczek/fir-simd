#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
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

template<typename T>
T highestMultipleOfNIn(T x, T N) {
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

    std::vector<float> ir3{ 0.4f, 0.2f, 0.4f, -0.1f, -0.4f, -0.3f, -0.5f, -0.11f, -0.3f};
    const auto expected = std::vector<float>{0.4f,  1.f,  2.f,  2.9f,  1.4f,  0.2f, -2.7f, -3.61f, -3.22f,
       -2.93f, -1.34f, -1.2f};
    
    const auto filtered3 = filteringFunction(signal, ir3);
    for (auto i = 0u; i < filtered3.size(); ++i) {
        assert(std::abs(filtered3[i] - expected[i]) < 1e-6f);
    }
}

void testFirFilterBigRandomVectors(std::function<std::vector<float>(const std::vector<float>&,const std::vector<float>&)> filteringFunction) {
    std::cout << "Starting long vectors test." << std::endl;

    const auto expected = applyFirFilterSingle(random1, random2);
    const auto given = filteringFunction(random1, random2);

    assert(expected.size() == given.size());
    for (auto i = 0u; i < expected.size(); ++i) {
        if (std::abs(expected[i] - given[i]) > 1e-4f) {
            assert(false);
        }
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

#ifdef __AVX__
std::vector<float> applyFirFilterAVX(const std::vector<float>& signal, const std::vector<float>& impulseResponse) {
    constexpr size_t AVX_FLOAT_COUNT = 256u / 32u;
    
    const auto minimalPaddedSize = signal.size() + 2 * impulseResponse.size() - 2;
    const auto avxAlignedPaddedSize = AVX_FLOAT_COUNT * (highestMultipleOfNIn(minimalPaddedSize - 1u, AVX_FLOAT_COUNT) + 1u);

    std::vector<float> paddedSignal(avxAlignedPaddedSize, 0.f);
    std::copy(signal.begin(), signal.end(), paddedSignal.begin() + impulseResponse.size() - 1u);

    std::vector<float> output(signal.size() + impulseResponse.size() - 1u);
    
    const auto avxAlignedImpulseResponseSize = AVX_FLOAT_COUNT * (highestMultipleOfNIn(impulseResponse.size() - 1u, AVX_FLOAT_COUNT) + 1);
    std::vector<float> reversedImpulseResponse(avxAlignedImpulseResponseSize, 0.f);
    
    std::reverse_copy(impulseResponse.begin(), impulseResponse.end(), reversedImpulseResponse.begin());
    for (auto i = impulseResponse.size(); i < reversedImpulseResponse.size(); ++i) 
        reversedImpulseResponse[i] = 0.f;
    
    const auto* x = paddedSignal.data();
    const auto* c = reversedImpulseResponse.data();

    std::array<float, AVX_FLOAT_COUNT> outStore;

    for (auto i = 0; i < output.size(); ++i) {
        auto outChunk = _mm256_setzero_ps();

        for (auto j = 0; j < avxAlignedImpulseResponseSize; j += AVX_FLOAT_COUNT) {
            auto xChunk = _mm256_loadu_ps(x + i + j);
            auto cChunk = _mm256_loadu_ps(c + j);

            auto temp = _mm256_mul_ps(xChunk, cChunk);

            outChunk = _mm256_add_ps(outChunk, temp);
        }
        
        _mm256_storeu_ps(outStore.data(), outChunk);

        output[i] = std::accumulate(outStore.begin(), outStore.end(), 0.f);
    }

    return output;
}
#endif

std::vector<float> applyFirFilter(const std::vector<float>& signal, const std::vector<float>& impulseResponse) {
#ifdef __AVX__
    std::cout << "Using AVX instructions." << std::endl;
    return applyFirFilterAVX(signal, impulseResponse);
#else
    std::cout << "Using single instructions." << std::endl;
    return applyFirFilterSingle(signal, impulseResponse);
#endif
}

int main() {
    testFirFilter(applyFirFilter);
    testFirFilterBigRandomVectors(applyFirFilter);

    std::cout << "Success!" << std::endl;
}
