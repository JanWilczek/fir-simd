#include "benchmark.h"

#include <iostream>

#include "AudioFile/AudioFile.h"
#include "FIRFilter.h"
#include "data/BigRandomVectors.h"

void benchmarkFirFilterImpulseResponses(
    std::function<std::vector<float>(FilterInput<float>&)>
        filteringFunction, size_t alignment) {
  std::cout << "Starting impulse responses benchmark." << std::endl;

  AudioFile<float> signal;
  signal.load("./../include/data/saw.wav");
  AudioFile<float> impulseResponse;
  impulseResponse.load("./../include/data/classroomImpulseResponse.wav");

  FilterInput<float> input(signal.samples[0], impulseResponse.samples[0],
                          alignment);

  const auto benchmarkResult = benchmark<std::vector<float>>(
      [&] {
        return filteringFunction(input);
      },
      1);

  std::cout << "Average execution time: " << benchmarkResult.averageTime.count()
            << " us." << std::endl;
}

void benchmarkFirFilterBigRandomVectors(
    std::function<std::vector<float>(FilterInput<float>&)>
        filteringFunction,
    size_t alignment) {
  std::cout << "Starting big random vectors benchmark." << std::endl;

  FilterInput<float> input(random1, random2, alignment);

  const auto benchmarkResult = benchmark<std::vector<float>>(
      [&] { return filteringFunction(input); }, 10000);

  std::cout << "Average execution time: " << benchmarkResult.averageTime.count()
            << " us." << std::endl;
}
