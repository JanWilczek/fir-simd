#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

#ifdef __AVX__
#include <immintrin.h>
#endif

#include "FIRFilter.h"

namespace {
bool is_aligned(const void* p, std::size_t n) {
  std::cout << reinterpret_cast<std::uintptr_t>(p) % n << std::endl;
}
}  // namespace

namespace fir {
std::vector<float> applyFirFilterSingle(FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;
  auto* y = input.y;

  for (auto i = 0u; i < input.outputLength; ++i) {
    y[i] = x[i] * c[0];
    for (auto j = 1u; j < input.filterLength; ++j) {
      y[i] += x[i + j] * c[j];
    }
  }
  return input.output();
}

std::vector<float> applyFirFilterInnerLoopVectorization(
    FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;
  auto* y = input.y;

  for (auto i = 0u; i < input.outputLength; ++i) {
    y[i] = 0.f;
    for (auto j = 0u; j < input.filterLength; j += 4) {
      y[i] += x[i + j] * c[j] + x[i + j + 1] * c[j + 1] +
              x[i + j + 2] * c[j + 2] + x[i + j + 3] * c[j + 3];
    }
  }
  return input.output();
}

std::vector<float> applyFirFilterOuterLoopVectorization(
    FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;
  auto* y = input.y;

  for (auto i = 0u; i < input.outputLength; i += 4) {
    y[i] = 0.f;
    y[i + 1] = 0.f;
    y[i + 2] = 0.f;
    y[i + 3] = 0.f;
    for (auto j = 0u; j < input.filterLength; ++j) {
      y[i] += x[i + j] * c[j];
      y[i + 1] += x[i + j + 1] * c[j];
      y[i + 2] += x[i + j + 2] * c[j];
      y[i + 3] += x[i + j + 3] * c[j];
    }
  }
  return input.output();
}

std::vector<float> applyFirFilterOuterInnerLoopVectorization(
    FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;
  auto* y = input.y;

  for (auto i = 0u; i < input.outputLength; i += 4) {
    y[i] = 0.f;
    y[i + 1] = 0.f;
    y[i + 2] = 0.f;
    y[i + 3] = 0.f;
    for (auto j = 0u; j < input.filterLength; j += 4) {
      y[i] += x[i + j] * c[j] + x[i + j + 1] * c[j + 1] +
              x[i + j + 2] * c[j + 2] + x[i + j + 3] * c[j + 3];

      y[i + 1] += x[i + j + 1] * c[j] + x[i + j + 2] * c[j + 1] +
                  x[i + j + 3] * c[j + 2] + x[i + j + 4] * c[j + 3];

      y[i + 2] += x[i + j + 2] * c[j] + x[i + j + 3] * c[j + 1] +
                  x[i + j + 4] * c[j + 2] + x[i + j + 5] * c[j + 3];

      y[i + 3] += x[i + j + 3] * c[j] + x[i + j + 4] * c[j + 1] +
                  x[i + j + 5] * c[j + 2] + x[i + j + 6] * c[j + 3];
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

std::vector<float> applyFirFilterAVX_outerLoopVectorization(
    FilterInput<float, AVX_FLOAT_COUNT * alignof(float)>& input) {
  const auto* x = input.x;
  const auto* c = input.c;
  auto* y = input.y;

  for (auto i = 0u; i < input.outputLength; i += AVX_FLOAT_COUNT) {
    auto yChunk = _mm256_setzero_ps();

    for (auto j = 0u; j < input.filterLength; ++j) {
      auto xChunk = _mm256_loadu_ps(x + i + j);
      auto cChunk = _mm256_set1_ps(c[j]);

      auto temp = _mm256_mul_ps(xChunk, cChunk);

      yChunk = _mm256_add_ps(yChunk, temp);
    }

    _mm256_storeu_ps(y + i, yChunk);
  }

  return input.output();
}

std::vector<float> applyFirFilterAVX_outerInnerLoopVectorization(
    FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;

  std::array<float, AVX_FLOAT_COUNT> outStore;

  std::array<__m256, AVX_FLOAT_COUNT> outChunk;

  for (auto i = 0u; i < input.outputLength; i += AVX_FLOAT_COUNT) {
    for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
      outChunk[k] = _mm256_setzero_ps();
    }

    for (auto j = 0u; j < input.filterLength; j += AVX_FLOAT_COUNT) {
      auto cChunk = _mm256_loadu_ps(c + j);

      for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
        auto xChunk = _mm256_loadu_ps(x + i + j + k);

        auto temp = _mm256_mul_ps(xChunk, cChunk);

        outChunk[k] = _mm256_add_ps(outChunk[k], temp);
      }
    }

    for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
      _mm256_storeu_ps(outStore.data(), outChunk[k]);

      if (i + k < input.outputLength)
        input.y[i + k] =
            std::accumulate(outStore.begin(), outStore.end(), 0.f);
    }
  }

  return input.output();
}

std::vector<float> applyFirFilterAVX_outerInnerLoopVectorizationAligned(
    FilterInput<float, AVX_FLOAT_COUNT * alignof(float)>& input) {
  const auto* x = input.x;
  const auto* cAligned = input.cAligned;

  alignas(__m256) std::array<float, AVX_FLOAT_COUNT> outStore;

  std::array<__m256, AVX_FLOAT_COUNT> outChunk;

  for (auto i = 0u; i < input.outputLength; i += AVX_FLOAT_COUNT) {
    for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
      outChunk[k] = _mm256_setzero_ps();
    }

    for (auto j = 0u; j < input.filterLength; j += AVX_FLOAT_COUNT) {
      auto xChunk = _mm256_load_ps(x + i + j);

      for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
        auto cChunk = _mm256_load_ps(cAligned[k] + j);

        auto temp = _mm256_mul_ps(xChunk, cChunk);

        outChunk[k] = _mm256_add_ps(outChunk[k], temp);
      }
    }

    for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
      _mm256_store_ps(outStore.data(), outChunk[k]);
      if (i + k < input.outputLength)
        input.y[i + k] =
            std::accumulate(outStore.begin(), outStore.end(), 0.f);
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
}  // namespace fir
