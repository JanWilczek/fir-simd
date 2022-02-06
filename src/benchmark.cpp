#include "benchmark.h"

#include <iostream>

#include "FIRFilter.h"
#include "AudioFile/AudioFile.h"

void benchmarkFirFilterImpulseResponses(std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> filteringFunction)
{
    std::cout << "Starting impulse responses benchmark." << std::endl;

    AudioFile<float> signal;
    signal.load("./../include/data/saw.wav");
    AudioFile<float> impulseResponse;
    impulseResponse.load("./../include/data/classroomImpulseResponse.wav");

    const auto benchmarkResult = benchmark<std::vector<float>>([&] { return applyFirFilterSingle(signal.samples[0], impulseResponse.samples[0]); }, 1);

    std::cout << "Average execution time: " << benchmarkResult.averageTime.count() << " ms." << std::endl;
}
