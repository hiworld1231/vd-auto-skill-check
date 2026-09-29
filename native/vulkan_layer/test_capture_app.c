#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(call, label) do { VkResult _r = (call); if (_r != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", (label), (int)_r); return 2; } } while (0)

static uint32_t clamp_u32(uint32_t value, uint32_t lo, uint32_t hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static VkCompositeAlphaFlagBitsKHR choose_alpha(VkCompositeAlphaFlagsKHR flags) {
    const VkCompositeAlphaFlagBitsKHR choices[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (size_t i = 0; i < sizeof(choices) / sizeof(choices[0]); ++i) {
        if (flags & choices[i]) return choices[i];
    }
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

int main(void) {
    const char *inst_exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "VD_Capture_Test";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = inst_exts;

    VkInstance instance = VK_NULL_HANDLE;
    CHECK(vkCreateInstance(&ici, NULL, &instance), "vkCreateInstance");

    PFN_vkCreateHeadlessSurfaceEXT create_headless = (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT");
    if (!create_headless) { fprintf(stderr, "VK_EXT_headless_surface unavailable\n"); return 3; }
    VkHeadlessSurfaceCreateInfoEXT hs = {0};
    hs.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    CHECK(create_headless(instance, &hs, NULL, &surface), "vkCreateHeadlessSurfaceEXT");

    uint32_t gpu_count = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, NULL), "vkEnumeratePhysicalDevices(count)");
    if (!gpu_count) { fprintf(stderr, "no Vulkan physical devices\n"); return 4; }
    VkPhysicalDevice *gpus = calloc(gpu_count, sizeof(*gpus));
    CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, gpus), "vkEnumeratePhysicalDevices(list)");
    VkPhysicalDevice gpu = gpus[0];
    free(gpus);

    uint32_t q_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &q_count, NULL);
    VkQueueFamilyProperties *qprops = calloc(q_count, sizeof(*qprops));
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &q_count, qprops);
    uint32_t q_family = UINT32_MAX;
    for (uint32_t i = 0; i < q_count; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &present);
        if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) { q_family = i; break; }
    }
    free(qprops);
    if (q_family == UINT32_MAX) { fprintf(stderr, "no graphics+present queue\n"); return 5; }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = q_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    const char *dev_exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    VkDevice device = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(gpu, &dci, NULL, &device), "vkCreateDevice");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, q_family, 0, &queue);

    VkSurfaceCapabilitiesKHR caps;
    CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &caps), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        fprintf(stderr, "surface lacks TRANSFER_DST for deterministic clear test\n");
        return 6;
    }

    uint32_t fmt_count = 0;
    CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &fmt_count, NULL), "surface formats count");
    VkSurfaceFormatKHR *formats = calloc(fmt_count, sizeof(*formats));
    CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &fmt_count, formats), "surface formats list");
    VkSurfaceFormatKHR chosen = {0};
    int found = 0;
    for (uint32_t i = 0; i < fmt_count; ++i) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM) { chosen = formats[i]; found = 1; break; }
    }
    free(formats);
    if (!found) { fprintf(stderr, "B8G8R8A8_UNORM unavailable\n"); return 7; }

    VkExtent2D extent;
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        extent.width = clamp_u32(320, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = clamp_u32(320, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent.width < 16 || extent.height < 16) { fprintf(stderr, "surface extent too small\n"); return 8; }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount && image_count > caps.maxImageCount) image_count = caps.maxImageCount;
    VkSwapchainCreateInfoKHR sci = {0};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = surface;
    sci.minImageCount = image_count;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = choose_alpha(caps.supportedCompositeAlpha);
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    CHECK(vkCreateSwapchainKHR(device, &sci, NULL, &swapchain), "vkCreateSwapchainKHR");
    CHECK(vkGetSwapchainImagesKHR(device, swapchain, &image_count, NULL), "swapchain image count");
    VkImage *images = calloc(image_count, sizeof(*images));
    CHECK(vkGetSwapchainImagesKHR(device, swapchain, &image_count, images), "swapchain images");
    unsigned char *initialized = calloc(image_count, 1);

    VkCommandPoolCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = q_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    CHECK(vkCreateCommandPool(device, &cpci, NULL, &pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cbai = {0};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd), "vkAllocateCommandBuffers");

    VkSemaphoreCreateInfo sem_ci = {0}; sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore acquired = VK_NULL_HANDLE, rendered = VK_NULL_HANDLE;
    CHECK(vkCreateSemaphore(device, &sem_ci, NULL, &acquired), "vkCreateSemaphore(acquired)");
    CHECK(vkCreateSemaphore(device, &sem_ci, NULL, &rendered), "vkCreateSemaphore(rendered)");
    VkFenceCreateInfo fence_ci = {0}; fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence fence = VK_NULL_HANDLE;
    CHECK(vkCreateFence(device, &fence_ci, NULL, &fence), "vkCreateFence");

    for (int frame = 0; frame < 4; ++frame) {
        CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        CHECK(vkResetFences(device, 1, &fence), "vkResetFences");
        uint32_t idx = 0;
        CHECK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx), "vkAcquireNextImageKHR");
        CHECK(vkResetCommandPool(device, pool, 0), "vkResetCommandPool");
        VkCommandBufferBeginInfo begin = {0}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");

        VkImageMemoryBarrier to_dst = {0};
        to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_dst.srcAccessMask = 0;
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = initialized[idx] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = images[idx];
        to_dst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_dst.subresourceRange.levelCount = 1;
        to_dst.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_dst);

        VkClearColorValue color = {{0.10f, 0.60f, 0.20f, 1.0f}};
        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

        VkImageMemoryBarrier to_present = to_dst;
        to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_present.dstAccessMask = 0;
        to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &to_present);
        CHECK(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit = {0};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &acquired;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &rendered;
        CHECK(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit(render)");

        VkPresentInfoKHR present = {0};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &rendered;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &idx;
        VkResult pr = vkQueuePresentKHR(queue, &present);
        if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) { fprintf(stderr, "vkQueuePresentKHR failed: %d\n", (int)pr); return 9; }
        initialized[idx] = 1;
        usleep(250000);
    }

    vkDeviceWaitIdle(device);
    printf("CAPTURE_TEST_OK format=%u extent=%ux%u images=%u\n", chosen.format, extent.width, extent.height, image_count);
    fflush(stdout);

    vkDestroyFence(device, fence, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroySemaphore(device, acquired, NULL);
    vkDestroyCommandPool(device, pool, NULL);
    free(initialized);
    free(images);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
