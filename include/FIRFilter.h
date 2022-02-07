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

std::vector<float> applyFirFilterSingle(
    const std::vector<float>& signal,
    const std::vector<float>& impulseResponse);

#ifdef __AVX__
std::vector<float> applyFirFilterAVX(const std::vector<float>& signal,
                                     const std::vector<float>& impulseResponse);
#endif

std::vector<float> applyFirFilter(const std::vector<float>& signal,
                                  const std::vector<float>& impulseResponse);