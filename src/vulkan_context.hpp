#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <optional>

struct GLFWwindow;

// Different sync / validation scenarios for the sandbox
enum class ValidationScenario {
    Normal,         // Correct, minimal sync
    BadBarrier,     // Broken sync: present does not wait on render-finished
    OverSync,       // Brutally over-synchronized CPU–GPU
    CacheBug_Bad,   // NEW: compute->fragment without proper image barrier
    CacheBug_Fixed  // NEW: compute->fragment with correct image barrier
};

// Queue family indices (graphics + present)
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Swapchain support info
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Simple vertex for the triangle
struct Vertex {
    float pos[2];   // x, y
    float color[3]; // r, g, b
};

class VulkanContext {
public:
    explicit VulkanContext(GLFWwindow* window);
    ~VulkanContext();

    // Called every frame
    void drawFrame();
    void waitIdle();

    VkDevice getDevice() const { return device; }

    // Enable/disable CPU + GPU profiling
    void setProfilingEnabled(bool enabled) { profilingEnabled = enabled; }

    // Choose which validation / sync scenario to run
    void setScenario(ValidationScenario s) { scenario = s; }

private:
    // Setup steps
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createPipelineLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createVertexBuffer();
    void createCommandBuffers(); // allocate only; record per-frame
    void createSyncObjects();
    void createTimestampQueryPool();

    // NEW: cache bug scenario resources
    void createCacheBugResources();
    void destroyCacheBugResources();

    // Per-frame command buffer recording (scenario-aware)
    void recordCommandBuffer(uint32_t imageIndex);

    // Device + swapchain helpers
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) const;
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) const;
    bool isDeviceSuitable(VkPhysicalDevice device) const;

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) const;
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) const;
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) const;
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) const;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;

private:
    GLFWwindow* windowHandle = nullptr;

    // Core Vulkan handles
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    // Queues
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    // Swapchain & images
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    VkFormat swapchainImageFormat{};
    VkExtent2D swapchainExtent{};
    std::vector<VkImageView> swapchainImageViews;

    // Render pipeline objects
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    // Framebuffers
    std::vector<VkFramebuffer> swapchainFramebuffers;

    // Command pool + buffers
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    // Vertex buffer (for triangle / normal scenarios)
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;

    // Sync
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    // Per-frame sync: acquire + CPU/GPU fencing
    std::vector<VkSemaphore> imageAvailableSemaphores; // size = MAX_FRAMES_IN_FLIGHT
    std::vector<VkFence>     inFlightFences;           // size = MAX_FRAMES_IN_FLIGHT

    // Per-image sync: render-finished semaphores (what present waits on)
    std::vector<VkSemaphore> renderFinishedSemaphores; // size = swapchainImages.size()

    size_t currentFrame = 0;

    // Validation / sync scenario
    ValidationScenario scenario = ValidationScenario::Normal;

    // CPU profiling
    bool   profilingEnabled        = true;
    double accumulatedFrameTimeMs  = 0.0;
    uint32_t frameCounter          = 0;

    // GPU timestamp profiling
    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
    double timestampPeriodNs       = 0.0;   // from physical device properties
    double accumulatedGpuTimeMs    = 0.0;
    uint32_t gpuFrameCounter       = 0;

    // ===== NEW: Cache bug scenario resources =====
    bool cacheResourcesCreated     = false;

    // Shared image (compute writes, fragment samples)
    VkImage        cacheImage      = VK_NULL_HANDLE;
    VkDeviceMemory cacheImageMemory= VK_NULL_HANDLE;
    VkImageView    cacheImageView  = VK_NULL_HANDLE;
    VkSampler      cacheSampler    = VK_NULL_HANDLE;

    // Descriptor pool + layouts + sets
    VkDescriptorPool        descriptorPool          = VK_NULL_HANDLE;
    VkDescriptorSetLayout   cacheComputeSetLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout   cacheGraphicsSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSet         cacheComputeSet         = VK_NULL_HANDLE;
    VkDescriptorSet         cacheGraphicsSet        = VK_NULL_HANDLE;

    // Pipelines for cache scenario
    VkPipelineLayout cacheComputePipelineLayout     = VK_NULL_HANDLE;
    VkPipelineLayout cacheGraphicsPipelineLayout    = VK_NULL_HANDLE;
    VkPipeline       cacheComputePipeline           = VK_NULL_HANDLE;
    VkPipeline       cacheGraphicsPipeline          = VK_NULL_HANDLE;

    // Simple time value for animation in compute shader
    float cacheTime = 0.0f;
};
