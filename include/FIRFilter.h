#pragma once
#include <vector>

template <typename T>
T highestPowerOf2NotGreaterThan(T x) {
  using namespace std;
  return static_cast<T>(pow(2., floor(log2(static_cast<double>(x)))));
}

template <typename T>
T highestMultipleOfNIn(T x, T N) {
  return static_cast<long long>(x / N);
}

template <typename SampleType>
struct FilterInput {
  FilterInput(const std::vector<SampleType>& inputSignal,
              const std::vector<SampleType>& filter, size_t alignment = 1u)
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
  }

  std::vector<SampleType> output() { return outputStorage; }

  size_t alignment;
  const SampleType* x;  // input signal
  size_t inputLength;
  const SampleType* c;  // reversed filter coefficients
  size_t filterLength;
  SampleType* y;  // output (filtered) signal
  size_t outputLength;

 private:
  std::vector<SampleType> inputStorage;
  std::vector<SampleType> reversedFilterCoefficientsStorage;
  std::vector<SampleType> outputStorage;
};

std::vector<float> applyFirFilterSingle(FilterInput<float>& input);

#ifdef __AVX__
constexpr size_t AVX_FLOAT_COUNT = 256u / 32u;

std::vector<float> applyFirFilterAVX_innerLoopVectorization(
    FilterInput<float>& input);

std::vector<float> applyFirFilterAVX_outerInnerLoopVectorization(
    FilterInput<float>& input);
#endif

std::vector<float> applyFirFilter(FilterInput<float>& input);