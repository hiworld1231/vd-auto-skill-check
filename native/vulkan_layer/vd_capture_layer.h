#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <cstdint>

#define VD_LAYER_NAME "VK_LAYER_VD_capture"
#define VD_LAYER_DESCRIPTION "VD Auto Skill Check Vulkan Capture Layer"
#define VD_LAYER_SPEC_VERSION VK_API_VERSION_1_3
#define VD_LAYER_IMPL_VERSION 2
#define VD_LAYER_SHM_MAGIC 0x56444C59u // 'VDLY'
#define VD_LAYER_SHM_VERSION 2u
#define VD_CAPTURE_MAX_WIDTH 256u
#define VD_CAPTURE_MAX_HEIGHT 256u
#define VD_CAPTURE_BYTES_PER_PIXEL 4u
#define VD_CAPTURE_MAX_BYTES (VD_CAPTURE_MAX_WIDTH * VD_CAPTURE_MAX_HEIGHT * VD_CAPTURE_BYTES_PER_PIXEL)

// status_flags
#define VD_STATUS_ACTIVE          (1u << 0)
#define VD_STATUS_SWAPCHAIN_READY (1u << 1)
#define VD_STATUS_CAPTURE_ENABLED (1u << 2)
#define VD_STATUS_PIXELS_READY    (1u << 3)

#pragma pack(push, 8)
struct VdLayerShmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t struct_size;
    uint32_t status_flags;

    uint32_t pid;
    uint32_t sequence;              // odd while writer mutates mmap, even when stable

    uint64_t init_timestamp_ns;
    uint64_t last_present_ns;

    uint64_t frame_count;
    float current_fps;
    uint32_t dropped_frames;

    uint32_t swapchain_width;
    uint32_t swapchain_height;
    uint32_t swapchain_format;
    uint32_t swapchain_present_mode;
    uint32_t swapchain_image_count;
    uint32_t current_image_index;

    char process_name[64];

    uint32_t roi_x;
    uint32_t roi_y;
    uint32_t roi_width;
    uint32_t roi_height;
    uint32_t roi_stride;
    uint32_t roi_data_offset;
    uint32_t roi_data_size;

    uint64_t roi_capture_count;
    uint64_t roi_capture_timestamp_ns;

    uint8_t padding[112];
};
#pragma pack(pop)

#define VD_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct);
VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName);
VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount, VkLayerProperties *pProperties);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount, VkLayerProperties *pProperties);
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties);

#ifdef __cplusplus
}
#endif
