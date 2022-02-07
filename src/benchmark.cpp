#include "benchmark.h"

#include <iostream>

#include "AudioFile/AudioFile.h"
#include "FIRFilter.h"
#include "data/BigRandomVectors.h"

void benchmarkFirFilterImpulseResponses(
    std::function<std::vector<float>(const std::vector<float>&,
                                     const std::vector<float>&)>
        filteringFunction) {
  std::cout << "Starting impulse responses benchmark." << std::endl;

  AudioFile<float> signal;
  signal.load("./../include/data/saw.wav");
  AudioFile<float> impulseResponse;
  impulseResponse.load("./../include/data/classroomImpulseResponse.wav");

  const auto benchmarkResult = benchmark<std::vector<float>>(
      [&] {
        return filteringFunction(signal.samples[0], impulseResponse.samples[0]);
      },
      1);

  std::cout << "Average execution time: " << benchmarkResult.averageTime.count()
            << " ms." << std::endl;
}

void benchmarkFirFilterBigRandomVectors(
    std::function<std::vector<float>(const std::vector<float>&,
                                     const std::vector<float>&)>
        filteringFunction) {
  std::cout << "Starting big random vectors benchmark." << std::endl;

  const auto benchmarkResult = benchmark<std::vector<float>>(
      [&] { return filteringFunction(random1, random2); }, 20);

  std::cout << "Average execution time: " << benchmarkResult.averageTime.count()
            << " ms." << std::endl;
}
