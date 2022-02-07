#pragma once
#include <cassert>
#include <functional>
#include <vector>

void testFirFilter(std::function<std::vector<float>(const std::vector<float>&,
                                                    const std::vector<float>&)>
                       filteringFunction);

template <typename T>
void assertEqualVectors(const std::vector<T>& a, const std::vector<T>& b,
                        T relativeError) {
  assert(a.size() == b.size());
  for (auto i = 0u; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > relativeError) {
      assert(false);
    }
  }
}

void testFirFilterBigRandomVectors(
    std::function<std::vector<float>(const std::vector<float>&,
                                     const std::vector<float>&)>
        filteringFunction);

void testFirFilterImpulseResponses(
    std::function<std::vector<float>(const std::vector<float>&,
                                     const std::vector<float>&)>
        filteringFunction);