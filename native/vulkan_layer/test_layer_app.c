#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    printf("[APP] Starting Vulkan test client (PID %d)...\n", (int)getpid());
    fflush(stdout);

    int test_swapchain = 0;
    int layer_count = 0;
    const char *enabled_layers[] = {"VK_LAYER_VD_capture"};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--with-layer") == 0) {
            layer_count = 1;
            printf("[APP] Explicitly requesting layer VK_LAYER_VD_capture\n");
        } else if (strcmp(argv[i], "--test-swapchain") == 0) {
            test_swapchain = 1;
        }
    }
    if (getenv("FORCE_VD_LAYER")) {
        layer_count = 1;
    }
    if (getenv("TEST_SWAPCHAIN")) {
        test_swapchain = 1;
    }

    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "VD_Vulkan_Test_App";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "VD_Test";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    const char *inst_exts[4];
    uint32_t inst_ext_count = 0;
    if (test_swapchain) {
        inst_exts[inst_ext_count++] = VK_KHR_SURFACE_EXTENSION_NAME;
        inst_exts[inst_ext_count++] = VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME;
    }

    VkInstanceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = layer_count;
    create_info.ppEnabledLayerNames = layer_count > 0 ? enabled_layers : NULL;
    create_info.enabledExtensionCount = inst_ext_count;
    create_info.ppEnabledExtensionNames = inst_ext_count > 0 ? inst_exts : NULL;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = vkCreateInstance(&create_info, NULL, &instance);
    printf("[APP] vkCreateInstance result: %d, instance: %p\n", (int)res, (void*)instance);
    fflush(stdout);
    if (res != VK_SUCCESS) return 1;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (test_swapchain) {
        PFN_vkCreateHeadlessSurfaceEXT pfnCreateHeadlessSurfaceEXT =
            (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT");
        if (pfnCreateHeadlessSurfaceEXT) {
            VkHeadlessSurfaceCreateInfoEXT surf_info;
            memset(&surf_info, 0, sizeof(surf_info));
            surf_info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
            res = pfnCreateHeadlessSurfaceEXT(instance, &surf_info, NULL, &surface);
            printf("[APP] vkCreateHeadlessSurfaceEXT result: %d, surface: %p\n", (int)res, (void*)surface);
        } else {
            printf("[APP] WARNING: vkCreateHeadlessSurfaceEXT not available\n");
            test_swapchain = 0;
        }
    }

    uint32_t gpu_count = 0;
    res = vkEnumeratePhysicalDevices(instance, &gpu_count, NULL);
    printf("[APP] vkEnumeratePhysicalDevices count: %u, res: %d\n", gpu_count, (int)res);
    fflush(stdout);

    if (gpu_count > 0) {
        VkPhysicalDevice *gpus = (VkPhysicalDevice *)malloc(gpu_count * sizeof(VkPhysicalDevice));
        if (gpus) {
            vkEnumeratePhysicalDevices(instance, &gpu_count, gpus);

            VkPhysicalDeviceProperties props;
            memset(&props, 0, sizeof(props));
            vkGetPhysicalDeviceProperties(gpus[0], &props);
            printf("[APP] Selected GPU 0: %s (Driver: %u, API: %u)\n",
                   props.deviceName, props.driverVersion, props.apiVersion);
            fflush(stdout);

            float priority = 1.0f;
            VkDeviceQueueCreateInfo q_info;
            memset(&q_info, 0, sizeof(q_info));
            q_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q_info.queueFamilyIndex = 0;
            q_info.queueCount = 1;
            q_info.pQueuePriorities = &priority;

            const char *dev_exts[4];
            uint32_t dev_ext_count = 0;
            if (test_swapchain && surface != VK_NULL_HANDLE) {
                dev_exts[dev_ext_count++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
            }

            VkDeviceCreateInfo dev_info;
            memset(&dev_info, 0, sizeof(dev_info));
            dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            dev_info.queueCreateInfoCount = 1;
            dev_info.pQueueCreateInfos = &q_info;
            dev_info.enabledExtensionCount = dev_ext_count;
            dev_info.ppEnabledExtensionNames = dev_ext_count > 0 ? dev_exts : NULL;

            VkDevice device = VK_NULL_HANDLE;
            VkResult dev_res = vkCreateDevice(gpus[0], &dev_info, NULL, &device);
            printf("[APP] vkCreateDevice result: %d, device: %p\n", (int)dev_res, (void*)device);
            fflush(stdout);

            if (dev_res == VK_SUCCESS) {
                VkQueue queue = VK_NULL_HANDLE;
                vkGetDeviceQueue(device, 0, 0, &queue);
                printf("[APP] vkGetDeviceQueue queue: %p\n", (void*)queue);
                fflush(stdout);

                if (test_swapchain && surface != VK_NULL_HANDLE) {
                    PFN_vkCreateSwapchainKHR pfnCreateSwapchain =
                        (PFN_vkCreateSwapchainKHR)vkGetDeviceProcAddr(device, "vkCreateSwapchainKHR");
                    PFN_vkDestroySwapchainKHR pfnDestroySwapchain =
                        (PFN_vkDestroySwapchainKHR)vkGetDeviceProcAddr(device, "vkDestroySwapchainKHR");
                    PFN_vkGetSwapchainImagesKHR pfnGetSwapchainImages =
                        (PFN_vkGetSwapchainImagesKHR)vkGetDeviceProcAddr(device, "vkGetSwapchainImagesKHR");
                    PFN_vkQueuePresentKHR pfnQueuePresent =
                        (PFN_vkQueuePresentKHR)vkGetDeviceProcAddr(device, "vkQueuePresentKHR");

                    if (pfnCreateSwapchain && pfnQueuePresent) {
                        VkSwapchainCreateInfoKHR sc_info;
                        memset(&sc_info, 0, sizeof(sc_info));
                        sc_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
                        sc_info.surface = surface;
                        sc_info.minImageCount = 3;
                        sc_info.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
                        sc_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
                        sc_info.imageExtent.width = 1280;
                        sc_info.imageExtent.height = 720;
                        sc_info.imageArrayLayers = 1;
                        sc_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                        sc_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                        sc_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
                        sc_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
                        sc_info.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                        sc_info.clipped = VK_TRUE;

                        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
                        VkResult sc_res = pfnCreateSwapchain(device, &sc_info, NULL, &swapchain);
                        printf("[APP] vkCreateSwapchainKHR: res=%d swapchain=%p\n", (int)sc_res, (void*)swapchain);

                        if (sc_res == VK_SUCCESS) {
                            uint32_t img_count = 0;
                            if (pfnGetSwapchainImages) {
                                pfnGetSwapchainImages(device, swapchain, &img_count, NULL);
                                printf("[APP] Swapchain images: %u\n", img_count);
                            }

                            uint32_t img_idx = 0;
                            VkPresentInfoKHR present_info;
                            memset(&present_info, 0, sizeof(present_info));
                            present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
                            present_info.swapchainCount = 1;
                            present_info.pSwapchains = &swapchain;
                            present_info.pImageIndices = &img_idx;

                            for (int frame = 0; frame < 30; frame++) {
                                img_idx = frame % (img_count > 0 ? img_count : 1);
                                pfnQueuePresent(queue, &present_info);
                                usleep(1000); // 1ms simulated frame time
                            }
                            printf("[APP] Completed 30 presentation frames\n");

                            if (pfnDestroySwapchain) {
                                pfnDestroySwapchain(device, swapchain, NULL);
                                printf("[APP] vkDestroySwapchainKHR completed\n");
                            }
                        }
                    }
                }

                vkDestroyDevice(device, NULL);
                printf("[APP] vkDestroyDevice completed\n");
                fflush(stdout);
            }
            free(gpus);
        }
    }

    if (surface != VK_NULL_HANDLE) {
        PFN_vkDestroySurfaceKHR pfnDestroySurfaceKHR =
            (PFN_vkDestroySurfaceKHR)vkGetInstanceProcAddr(instance, "vkDestroySurfaceKHR");
        if (pfnDestroySurfaceKHR) pfnDestroySurfaceKHR(instance, surface, NULL);
    }

    vkDestroyInstance(instance, NULL);
    printf("[APP] vkDestroyInstance completed successfully\n");
    fflush(stdout);
    return 0;
}
