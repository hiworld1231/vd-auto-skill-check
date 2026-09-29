#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <cstdint>

#define VD_LAYER_NAME "VK_LAYER_VD_capture"
#define VD_LAYER_DESCRIPTION "VD Auto Skill Check Vulkan Capture Layer"
#define VD_LAYER_SPEC_VERSION VK_API_VERSION_1_3
#define VD_LAYER_IMPL_VERSION 1
#define VD_LAYER_SHM_MAGIC 0x56444C59 // 'VDLY'

#pragma pack(push, 8)
struct VdLayerShmHeader {
    uint32_t magic;                 // VD_LAYER_SHM_MAGIC
    uint32_t version;               // 1
    uint32_t struct_size;           // sizeof(VdLayerShmHeader)
    uint32_t status_flags;          // Bit 0: active, Bit 1: swapchain_ready

    uint32_t pid;                   // Process PID
    uint32_t sequence;              // Sequence counter (odd during write, even when stable)

    uint64_t init_timestamp_ns;     // CLOCK_MONOTONIC start time
    uint64_t last_present_ns;       // CLOCK_MONOTONIC last present time

    uint64_t frame_count;           // Total frames presented
    float current_fps;              // Moving average FPS
    uint32_t dropped_frames;

    // Swapchain information
    uint32_t swapchain_width;
    uint32_t swapchain_height;
    uint32_t swapchain_format;      // VkFormat (e.g., 44 = VK_FORMAT_B8G8R8A8_UNORM)
    uint32_t swapchain_present_mode;// VkPresentModeKHR (e.g., 0 = IMMEDIATE)
    uint32_t swapchain_image_count;
    uint32_t current_image_index;

    // Process info
    char process_name[64];

    // Framebuffer region capture ROI parameters (for subsequent stage)
    uint32_t roi_x;
    uint32_t roi_y;
    uint32_t roi_width;
    uint32_t roi_height;
    uint32_t roi_stride;
    uint32_t roi_data_offset;
    uint32_t roi_data_size;

    uint8_t padding[128];
};
#pragma pack(pop)

#define VD_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

// Layer negotiation and entry points
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct);
VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName);
VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName);
VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char *pName);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkInstance *pInstance);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDevice *pDevice);

// Layer properties
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount, VkLayerProperties *pProperties);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount, VkLayerProperties *pProperties);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties);

#ifdef __cplusplus
}
#endif
