#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef __AVX__
#include <immintrin.h>
#endif

#include "FIRFilter.h"

std::vector<float> applyFirFilterSingle(
    const std::vector<float>& signal,
    const std::vector<float>& impulseResponse) {
  std::vector<float> output(signal.size() + impulseResponse.size() - 1u);

  for (int i = 0; i < static_cast<int>(output.size()); ++i) {
    output[i] = 0.f;
    const auto startId =
        std::max(0, i - static_cast<int>(impulseResponse.size()) + 1);
    const auto endId = std::min(i, static_cast<int>(signal.size()) - 1);
    for (auto j = startId; j <= endId; ++j) {
      output[i] += signal[j] * impulseResponse[i - j];
    }
  }
  return output;
}

#ifdef __AVX__
std::vector<float> applyFirFilterAVX(
    const std::vector<float>& signal,
    const std::vector<float>& impulseResponse) {
  constexpr size_t AVX_FLOAT_COUNT = 256u / 32u;

  const auto minimalPaddedSize = signal.size() + 2 * impulseResponse.size() - 2;
  const auto avxAlignedPaddedSize =
      AVX_FLOAT_COUNT *
      (highestMultipleOfNIn(minimalPaddedSize - 1u, AVX_FLOAT_COUNT) + 1u);

  std::vector<float> paddedSignal(avxAlignedPaddedSize, 0.f);
  std::copy(signal.begin(), signal.end(),
            paddedSignal.begin() + impulseResponse.size() - 1u);

  std::vector<float> output(signal.size() + impulseResponse.size() - 1u);

  const auto avxAlignedImpulseResponseSize =
      AVX_FLOAT_COUNT *
      (highestMultipleOfNIn(impulseResponse.size() - 1u, AVX_FLOAT_COUNT) + 1);
  std::vector<float> reversedImpulseResponse(avxAlignedImpulseResponseSize,
                                             0.f);

  std::reverse_copy(impulseResponse.begin(), impulseResponse.end(),
                    reversedImpulseResponse.begin());
  for (auto i = impulseResponse.size(); i < reversedImpulseResponse.size(); ++i)
    reversedImpulseResponse[i] = 0.f;

  const auto* x = paddedSignal.data();
  const auto* c = reversedImpulseResponse.data();

  std::array<float, AVX_FLOAT_COUNT> outStore;

  // Inner loop vectorization
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

std::vector<float> applyFirFilter(const std::vector<float>& signal,
                                  const std::vector<float>& impulseResponse) {
#ifdef __AVX__
  std::cout << "Using AVX instructions." << std::endl;
  return applyFirFilterAVX(signal, impulseResponse);
#else
  std::cout << "Using single instructions." << std::endl;
  return applyFirFilterSingle(signal, impulseResponse);
#endif
}
