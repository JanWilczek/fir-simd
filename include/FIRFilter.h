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
#endif

template <typename SampleType>
struct FilterInput {
  FilterInput(const std::vector<SampleType>& inputSignal,
              const std::vector<SampleType>& filter,
              size_t alignment = 1u)
      : alignment(alignment) {
    const auto minimalPaddedSize = inputSignal.size() + 2 * filter.size() - 2u;
    const auto alignedPaddedSize =
        alignment *
        (highestMultipleOfNIn(minimalPaddedSize - 1u, alignment) + 1u);
    inputLength = alignedPaddedSize;

    inputStorage.resize(inputLength, 0.f);
    std::copy(inputSignal.begin(), inputSignal.end(),
              inputStorage.begin() + filter.size() - 1u);

    outputLength = inputSignal.size() + filter.size() - 1u;
    outputStorage.resize(outputLength);

    filterLength =
        alignment * (highestMultipleOfNIn(filter.size() - 1u, alignment) + 1);
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

    for (auto k = 0u; k < AVX_FLOAT_COUNT; ++k) {
      const auto alignedStorageSize =
          reversedFilterCoefficientsStorage.size() + AVX_FLOAT_COUNT - 1u;
      alignedReversedFilterCoefficientsStorage[k].resize(alignedStorageSize);

      for (auto i = 0u; i < k; ++i) {
        alignedReversedFilterCoefficientsStorage[k][i] = 0.f;
      }
      std::copy(reversedFilterCoefficientsStorage.begin(),
                reversedFilterCoefficientsStorage.end(),
                alignedReversedFilterCoefficientsStorage[k].begin() + k);
      for (auto i = reversedFilterCoefficientsStorage.size() + k;
           i < alignedStorageSize; ++i) {
        alignedReversedFilterCoefficientsStorage[k][i] = 0.f;
      }
    }
    cAligned = alignedReversedFilterCoefficientsStorage.data();
  }

  std::vector<SampleType> output() {
    auto result = outputStorage;
    result.resize(outputLength);
    return result;
  }

  size_t alignment;
  const SampleType* x;  // input signal
  size_t inputLength;
  const SampleType* c;  // reversed filter coefficients
  size_t filterLength;
  SampleType* y;  // output (filtered) signal
  size_t outputLength;
  std::vector<SampleType>* cAligned;

 private:
  alignas(AVX_FLOAT_COUNT) std::vector<float> inputStorage;
  std::vector<float> reversedFilterCoefficientsStorage;
  alignas(AVX_FLOAT_COUNT) std::vector<float> outputStorage;
  alignas(AVX_FLOAT_COUNT)
      std::array<std::vector<float>,
                 AVX_FLOAT_COUNT> alignedReversedFilterCoefficientsStorage;
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
    FilterInput<float>& input);

std::vector<float> applyFirFilterAVX_outerInnerLoopVectorization(
    FilterInput<float>& input);

std::vector<float> applyFirFilterAVX_outerInnerLoopVectorizationAligned(
    FilterInput<float>& input);
#endif

std::vector<float> applyFirFilter(FilterInput<float>& input);
}  // namespace fir
