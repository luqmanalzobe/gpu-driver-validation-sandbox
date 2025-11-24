#pragma once
#include <chrono>

class FrameProfiler {
public:
    using Clock = std::chrono::high_resolution_clock;

    void beginFrame() {
        start = Clock::now();
    }

    void endFrame() {
        auto end = Clock::now();
        lastFrameMs = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double getLastFrameMs() const { return lastFrameMs; }

private:
    Clock::time_point start{};
    double lastFrameMs = 0.0;
};
