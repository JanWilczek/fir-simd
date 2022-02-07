#include "test.h"

#include <iostream>

#include "AudioFile/AudioFile.h"
#include "FIRFilter.h"
#include "data/BigRandomVectors.h"

void testFirFilter(std::function<std::vector<float>(const std::vector<float>&,
                                                    const std::vector<float>&)>
                       filteringFunction) {
  std::vector<float> signal{
      1.f,
      2.f,
      3.f,
      4.f,
  };
  std::vector<float> ir{1.f};

  const auto filtered = filteringFunction(signal, ir);

  assert(filtered == signal);

  const auto filtered2 = filteringFunction(signal, {0.f, 1.f});

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

  const auto filtered3 = filteringFunction(signal, ir3);
  for (auto i = 0u; i < filtered3.size(); ++i) {
    assert(std::abs(filtered3[i] - expected[i]) < 1e-6f);
  }
}

void testFirFilterTwoVectors(
    const std::vector<float>& signal, const std::vector<float>& impulseResponse,
    std::function<std::vector<float>(const std::vector<float>&,
                                     const std::vector<float>&)>
        filteringFunction) {
  const auto expected = applyFirFilterSingle(signal, impulseResponse);
  const auto given = filteringFunction(signal, impulseResponse);

  assertEqualVectors(expected, given, 1e-6f);
}

void testFirFilterBigRandomVectors(
    std::function<std::vector<float>(const std::vector<float>&,
                                     const std::vector<float>&)>
        filteringFunction) {
  std::cout << "Starting long vectors test." << std::endl;
  testFirFilterTwoVectors(random1, random2, filteringFunction);
}

void testFirFilterImpulseResponses(
    std::function<std::vector<float>(const std::vector<float>&,
                                     const std::vector<float>&)>
        filteringFunction) {
  std::cout << "Starting impulse responses test." << std::endl;

  AudioFile<float> signal;
  signal.load("./../include/data/saw.wav");
  AudioFile<float> impulseResponse;
  impulseResponse.load("./../include/data/classroomImpulseResponse.wav");

  const auto expected =
      applyFirFilterSingle(signal.samples[0], impulseResponse.samples[0]);

  const auto avxFilteredSignal =
      filteringFunction(signal.samples[0], impulseResponse.samples[0]);

  /*assertEqualVectors(expected, avxFilteredSignal, 1e-5f);*/
  auto maximumAbsoluteError = 0.f;
  for (auto i = 0u; i < expected.size(); ++i) {
    const auto absoluteError = std::abs(expected[i] - avxFilteredSignal[i]);
    if (absoluteError > maximumAbsoluteError) {
      maximumAbsoluteError = absoluteError;
    }
  }

  std::cout << "Maximum absolute error: " << maximumAbsoluteError << std::endl;
}
