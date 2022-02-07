#pragma once
#include <chrono>
#include <functional>
#include <vector>

template <typename SampleType>
struct FilterInput;

template <typename ResultType>
struct Result {
  ResultType returnValue;
  std::chrono::milliseconds averageTime;
};

template <typename ResultType>
Result<ResultType> benchmark(std::function<ResultType()> function,
                             int callCount = 1) {
  using namespace std::chrono;

  Result<ResultType> benchmarkResult;

  std::chrono::milliseconds totalTime{0};
  for (auto i = 0; i < callCount; ++i) {
    const auto start = high_resolution_clock::now();
    benchmarkResult.returnValue = function();
    const auto end = high_resolution_clock::now();
    totalTime += duration_cast<std::chrono::milliseconds>(end - start);
  }

  benchmarkResult.averageTime = totalTime / callCount;

  return benchmarkResult;
}

void benchmarkFirFilterImpulseResponses(
    std::function<std::vector<float>(FilterInput<float>&)>
        filteringFunction, size_t alignment = 1u);

void benchmarkFirFilterBigRandomVectors(
    std::function<std::vector<float>(FilterInput<float>&)>
        filteringFunction,
    size_t alignment = 1u);
