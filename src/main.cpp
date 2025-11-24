#include <GLFW/glfw3.h>
#include <stdexcept>
#include <iostream>

#include "vulkan_context.hpp"

int main() {
    // Init GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    // Tell GLFW we don't want an OpenGL context, just a window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(800, 600, "GPU Driver Validation Sandbox", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    try {
        VulkanContext context(window);

        // Enable profiling
        context.setProfilingEnabled(true);

        // Choose scenario:
        // Normal:
        //context.setScenario(ValidationScenario::Normal);

        // Misconfigured / over-conservative barrier:
        context.setScenario(ValidationScenario::BadBarrier);

        // Brutal oversync (vkDeviceWaitIdle every frame):
        //context.setScenario(ValidationScenario::OverSync);

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            context.drawFrame();
        }

        context.waitIdle();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
