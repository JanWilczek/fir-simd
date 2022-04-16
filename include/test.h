#pragma once
#include <cassert>
#include <functional>
#include <vector>
#include <iostream>
#include "FIRFilter.h"
#include "AudioFile/AudioFile.h"
#include "data/BigRandomVectors.h"

template <typename T>
void assertEqualVectors(const std::vector<T>& a,
                        const std::vector<T>& b,
                        T relativeError) {
  assert(a.size() == b.size());
  for (auto i = 0u; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > relativeError) {
      assert(false);
    }
  }
}

template <size_t alignment>
void testFirFilter(
    std::function<std::vector<float>(fir::FilterInput<float, alignment>&)>
                       filteringFunction) {
  std::vector<float> signal{
      1.f,
      2.f,
      3.f,
      4.f,
  };
  std::vector<float> ir{1.f};

  fir::FilterInput<float, alignment> input1(signal, ir);
  const auto filtered = filteringFunction(input1);

  assert(filtered == signal);

  fir::FilterInput<float, alignment> input2(signal, {0.f, 1.f});
  const auto filtered2 = filteringFunction(input2);

  assert(filtered2[0] == 0.f);
  assert(filtered2[1] == 1.f);
  assert(filtered2[2] == 2.f);
  assert(filtered2[3] == 3.f);
  assert(filtered2[4] == 4.f);
  assert(filtered2.size() == 5u);

  std::vector<float> ir3{0.4f,  0.2f,  0.4f,   -0.1f, -0.4f,
                         -0.3f, -0.5f, -0.11f, -0.3f};
  const auto expected =
      std::vector<float>{0.4f,  1.f,    2.f,    2.9f,   1.4f,   0.2f,
                         -2.7f, -3.61f, -3.22f, -2.93f, -1.34f, -1.2f};

  fir::FilterInput<float, alignment> input3(signal, ir3);
  const auto filtered3 = filteringFunction(input3);
  for (auto i = 0u; i < filtered3.size(); ++i) {
    assert(std::abs(filtered3[i] - expected[i]) < 1e-6f);
  }
}

template <size_t alignment>
void testFirFilterTwoVectors(
    const std::vector<float>& signal,
    const std::vector<float>& impulseResponse,
    std::function<std::vector<float>(fir::FilterInput<float, alignment>&)>
        filteringFunction,
    float relativeError = 1e-6f) {
  fir::FilterInput<float> input(
      signal, impulseResponse);
  const auto expected =
      fir::applyFirFilterAVX_innerLoopVectorization(input);
  fir::FilterInput<float, alignment> inputAligned(signal, impulseResponse);
  const auto given = filteringFunction(inputAligned);

  assertEqualVectors(expected, given, relativeError);
}

template <size_t alignment>
void testFirFilterBigRandomVectors(
    std::function<std::vector<float>(fir::FilterInput<float, alignment>&)>
        filteringFunction) {
  std::cout << "Starting long vectors test." << std::endl;
  testFirFilterTwoVectors(random1, random2, filteringFunction, 1e-2f);
}

template <size_t alignment>
void testFirFilterImpulseResponses(
    std::function<std::vector<float>(fir::FilterInput<float, alignment>&)>
        filteringFunction) {
  std::cout << "Starting impulse responses test." << std::endl;

  AudioFile<float> signal;
  signal.load("saw.wav");
  AudioFile<float> impulseResponse;
  impulseResponse.load("classroomImpulseResponse.wav");

  fir::FilterInput<float> input(
      signal.samples[0], impulseResponse.samples[0]);
  fir::FilterInput<float, alignment>
      inputAligned(
      signal.samples[0], impulseResponse.samples[0]);

  const auto expected =
      fir::applyFirFilterAVX_outerInnerLoopVectorization(input);

  const auto filteredSignal = filteringFunction(inputAligned);

  /*assertEqualVectors(expected, filteredSignal, 1e-5f);*/
  auto maximumAbsoluteError = 0.f;
  for (auto i = 0u; i < input.outputLength; ++i) {
    const auto absoluteError = std::abs(expected[i] - filteredSignal[i]);
    if (absoluteError > maximumAbsoluteError) {
      maximumAbsoluteError = absoluteError;
    }
  }

  std::cout << "Maximum absolute error: " << maximumAbsoluteError
            << std::endl;
}



