#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef __AVX__
#include <immintrin.h>
#endif

#include "FIRFilter.h"

std::vector<float> applyFirFilterSingle(FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;
  auto* output = input.y;

  for (auto i = 0u; i < input.outputLength; ++i) {
    output[i] = x[0] * c[0];
    for (auto j = 1u; j < input.filterLength; ++j) {
      output[i] += x[i + j] * c[j];
    }
  }
  return input.output();
}

#ifdef __AVX__
std::vector<float> applyFirFilterAVX_innerLoopVectorization(
    FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;

  std::array<float, AVX_FLOAT_COUNT> outStore;

  // Inner loop vectorization
  for (auto i = 0u; i < input.outputLength; ++i) {
    auto outChunk = _mm256_setzero_ps();

    for (auto j = 0u; j < input.filterLength; j += AVX_FLOAT_COUNT) {
      auto xChunk = _mm256_loadu_ps(x + i + j);
      auto cChunk = _mm256_loadu_ps(c + j);

      auto temp = _mm256_mul_ps(xChunk, cChunk);

      outChunk = _mm256_add_ps(outChunk, temp);
    }

    _mm256_storeu_ps(outStore.data(), outChunk);

    input.y[i] = std::accumulate(outStore.begin(), outStore.end(), 0.f);
  }

  return input.output();
}

std::vector<float> applyFirFilterAVX_outerInnerLoopVectorization(
    FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;

  std::array<float, AVX_FLOAT_COUNT> outStore;

  alignas(AVX_FLOAT_COUNT * alignof(float)) std::array<__m256, AVX_FLOAT_COUNT>
      outChunk;

  for (auto i = 0u; i < input.outputLength; i += AVX_FLOAT_COUNT) {
    for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
      outChunk[k] = _mm256_setzero_ps();
    }

    for (auto j = 0ul; j < input.filterLength; j += AVX_FLOAT_COUNT) {
      auto cChunk = _mm256_loadu_ps(c + j);

      for (auto k = 0ul; k < AVX_FLOAT_COUNT; ++k) {
          auto xChunk = _mm256_loadu_ps(x + i + j + k);

        auto temp = _mm256_mul_ps(xChunk, cChunk);

        outChunk[k] = _mm256_add_ps(outChunk[k], temp);
      }
    }

    for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
      _mm256_storeu_ps(outStore.data(), outChunk[k]);

      input.y[i + k] = std::accumulate(outStore.begin(), outStore.end(), 0.f);
    }
  }

  return input.output();
}
#endif

std::vector<float> applyFirFilter(FilterInput<float>& input) {
#ifdef __AVX__
  std::cout << "Using AVX instructions." << std::endl;
  return applyFirFilterAVX_innerLoopVectorization(input);
#else
  std::cout << "Using single instructions." << std::endl;
  return applyFirFilterSingle(input);
#endif
}
