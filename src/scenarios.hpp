#pragma once
#include <vulkan/vulkan.h>

enum class Scenario {
    None,
    BadBarriers,
    FixedBarriers,
    InvalidDescriptorBinding
};

inline Scenario getActiveScenario() {
    // We’ll make this configurable later
    return Scenario::None;
}

void applyScenarioBarriers(VkCommandBuffer cmd);
void applyScenarioDescriptorMisbind(VkCommandBuffer cmd);
