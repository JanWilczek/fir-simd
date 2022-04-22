#pragma once
#include <algorithm>
#include <array>
#include <vector>

namespace fir {
template <typename T>
T highestPowerOf2NotGreaterThan(T x) {
  using namespace std;
  return static_cast<T>(pow(2., floor(log2(static_cast<double>(x)))));
}

template <typename T>
T highestMultipleOfNIn(T x, T N) {
  return static_cast<long long>(x / N);
}

#ifdef __AVX__
constexpr auto AVX_FLOAT_COUNT = 8u;

struct alignas(AVX_FLOAT_COUNT * alignof(float)) avx_alignment_t {};
#endif

template <typename SampleType, size_t alignment = alignof(SampleType)>
struct FilterInput {
  static constexpr auto VECTOR_SIZE = alignment / alignof(SampleType);

  FilterInput(const std::vector<SampleType>& inputSignal,
              const std::vector<SampleType>& filter) {
    const auto minimalPaddedSize = inputSignal.size() + 2 * filter.size() - 2u;
    const auto alignedPaddedSize =
        VECTOR_SIZE *
        (highestMultipleOfNIn(minimalPaddedSize - 1u, VECTOR_SIZE) + 2u);
    inputLength = alignedPaddedSize;

    inputStorage.resize(inputLength, 0.f);
    std::copy(inputSignal.begin(), inputSignal.end(),
              inputStorage.begin() + filter.size() - 1u);

    outputLength = inputSignal.size() + filter.size() - 1u;
    outputStorage.resize(outputLength);

    filterLength =
        VECTOR_SIZE *
        (highestMultipleOfNIn(filter.size() - 1u, VECTOR_SIZE) + 1);
    reversedFilterCoefficientsStorage.resize(filterLength);

    std::reverse_copy(filter.begin(), filter.end(),
                      reversedFilterCoefficientsStorage.begin());
    for (auto i = filter.size(); i < reversedFilterCoefficientsStorage.size();
         ++i)
      reversedFilterCoefficientsStorage[i] = 0.f;

    x = inputStorage.data();
    c = reversedFilterCoefficientsStorage.data();
    filterLength = reversedFilterCoefficientsStorage.size();
    y = outputStorage.data();

#ifdef __AVX__
    alignedStorageSize =
          reversedFilterCoefficientsStorage.size();
    for (auto k = 0u; k < VECTOR_SIZE; ++k) {
      alignedReversedFilterCoefficientsStorage[k].reset(
          reinterpret_cast<SampleType*>(new avx_alignment_t[alignedStorageSize]));

      for (auto i = 0u; i < k; ++i) {
        alignedReversedFilterCoefficientsStorage[k][i] = 0.f;
      }
      std::copy(reversedFilterCoefficientsStorage.begin(),
                reversedFilterCoefficientsStorage.end(),
                alignedReversedFilterCoefficientsStorage[k].get() + k);
      for (auto i = reversedFilterCoefficientsStorage.size() + k;
           i < alignedStorageSize; ++i) {
        alignedReversedFilterCoefficientsStorage[k][i] = 0.f;
      }
      cAligned[k] = alignedReversedFilterCoefficientsStorage[k].get();
    }
#endif
  }

  std::vector<SampleType> output() {
    auto result = outputStorage;
    result.resize(outputLength);
    return result;
  }

  const SampleType* x;  // input signal
  size_t inputLength;
  const SampleType* c;  // reversed filter coefficients
  size_t filterLength;
  SampleType* y;  // output (filtered) signal
  size_t outputLength;
#ifdef __AVX__
  size_t alignedStorageSize;
  SampleType* cAligned[VECTOR_SIZE];
#endif

 private:
  std::vector<float> inputStorage;
  std::vector<float> reversedFilterCoefficientsStorage;
  std::vector<float> outputStorage;
#ifdef __AVX__
  std::array<std::unique_ptr<SampleType[]>,
                 VECTOR_SIZE> alignedReversedFilterCoefficientsStorage;
#endif
};

std::vector<float> applyFirFilterSingle(FilterInput<float>& input);

std::vector<float> applyFirFilterInnerLoopVectorization(
    FilterInput<float>& input);

std::vector<float> applyFirFilterOuterLoopVectorization(
    FilterInput<float>& input);

std::vector<float> applyFirFilterOuterInnerLoopVectorization(
    FilterInput<float>& input);

#ifdef __AVX__
std::vector<float> applyFirFilterAVX_innerLoopVectorization(
    FilterInput<float>& input);

std::vector<float> applyFirFilterAVX_outerLoopVectorization(
    FilterInput<float, AVX_FLOAT_COUNT * alignof(float)>& input);

std::vector<float> applyFirFilterAVX_outerInnerLoopVectorization(
    FilterInput<float>& input);

std::vector<float> applyFirFilterAVX_outerInnerLoopVectorizationAligned(
    FilterInput<float, AVX_FLOAT_COUNT * alignof(float)>& input);
#endif

std::vector<float> applyFirFilter(FilterInput<float>& input);
}  // namespace fir
