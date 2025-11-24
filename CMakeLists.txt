cmake_minimum_required(VERSION 3.16)
project(GPUDriverValidationSandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Vulkan (from LunarG SDK) ---
if (WIN32 AND DEFINED ENV{VULKAN_SDK})
    set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/Include")
    set(Vulkan_LIBRARY "$ENV{VULKAN_SDK}/Lib/vulkan-1.lib")
    set(Vulkan_INCLUDE_DIRS "${Vulkan_INCLUDE_DIR}")
    set(Vulkan_LIBRARIES "${Vulkan_LIBRARY}")
endif()

find_package(Vulkan REQUIRED)

# --- GLFW (from vcpkg) ---
# Tell CMake exactly where glfw3Config.cmake lives
set(glfw3_DIR "C:/Users/dayaa/vcpkg/installed/x64-windows/share/glfw3")

# Use CONFIG mode so it picks up glfw3Config.cmake
find_package(glfw3 CONFIG REQUIRED)

add_executable(gpu_driver_validation_sandbox
    src/main.cpp
    src/vulkan_context.cpp
    src/frame_profiler.cpp
    src/scenarios.cpp
)

target_include_directories(gpu_driver_validation_sandbox PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)

# vcpkg's GLFW target is called 'glfw'
target_link_libraries(gpu_driver_validation_sandbox PRIVATE
    ${Vulkan_LIBRARIES}
    glfw
)

if(MSVC)
    target_compile_options(gpu_driver_validation_sandbox PRIVATE /W4 /EHsc)
else()
    target_compile_options(gpu_driver_validation_sandbox PRIVATE -Wall -Wextra -Wpedantic)
endif()
