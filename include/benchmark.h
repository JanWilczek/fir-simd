#pragma once
#include <chrono>
#include <functional>
#include <vector>
#include <iostream>

#include "AudioFile/AudioFile.h"
#include "FIRFilter.h"
#include "data/BigRandomVectors.h"

template <typename ResultType>
struct Result {
  ResultType returnValue;
  std::chrono::microseconds averageTime;
};

template <typename ResultType>
Result<ResultType> benchmark(std::function<ResultType()> function,
                             int callCount = 1) {
  using namespace std::chrono;

  Result<ResultType> benchmarkResult;

  using duration_type = decltype(Result<ResultType>::averageTime);
  duration_type totalTime{0};
  for (auto i = 0; i < callCount; ++i) {
    const auto start = high_resolution_clock::now();
    benchmarkResult.returnValue = function();
    const auto end = high_resolution_clock::now();
    totalTime += duration_cast<duration_type>(end - start);
  }

  benchmarkResult.averageTime = totalTime / callCount;

  return benchmarkResult;
}

template<size_t alignment>
void benchmarkFirFilterImpulseResponses(
    std::function<std::vector<float>(fir::FilterInput<float, alignment>&)>
        filteringFunction) {
  std::cout << "Starting impulse responses benchmark." << std::endl;

  AudioFile<float> signal;
  signal.load("./../include/data/saw.wav");
  AudioFile<float> impulseResponse;
  impulseResponse.load("./../include/data/classroomImpulseResponse.wav");

  fir::FilterInput<float, alignment> input(signal.samples[0],
                                      impulseResponse.samples[0]);

  const auto benchmarkResult = benchmark<std::vector<float>>(
      [&] { return filteringFunction(input); }, 1);

  std::cout << "Average execution time: "
            << benchmarkResult.averageTime.count() << " us." << std::endl;
}

template<size_t alignment>
void benchmarkFirFilterBigRandomVectors(
    std::function<std::vector<float>(fir::FilterInput<float, alignment>&)>
        filteringFunction) {
  std::cout << "Starting big random vectors benchmark." << std::endl;

  fir::FilterInput<float, alignment> input(random1, random2);

  const auto benchmarkResult = benchmark<std::vector<float>>(
      [&] { return filteringFunction(input); }, 10000);

  std::cout << "Average execution time: "
            << benchmarkResult.averageTime.count() << " us." << std::endl;
}
