#include "vd_capture_layer.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <signal.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

struct InstanceData;

struct InstanceData {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr next_gipa = nullptr;
    PFN_GetPhysicalDeviceProcAddr next_gpdpa = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkEnumeratePhysicalDeviceGroups vkEnumeratePhysicalDeviceGroups = nullptr;
    PFN_vkEnumeratePhysicalDeviceGroupsKHR vkEnumeratePhysicalDeviceGroupsKHR = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
};

struct DeviceData {
    VkDevice device = VK_NULL_HANDLE;
    InstanceData *instance_data = nullptr;
    PFN_vkGetDeviceProcAddr next_gdpa = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkGetDeviceQueue2 vkGetDeviceQueue2 = nullptr;
    PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR = nullptr;
    PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR = nullptr;
    PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR = nullptr;
    PFN_vkQueuePresentKHR vkQueuePresentKHR = nullptr;
};

struct SwapchainInfo {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    uint32_t image_count = 0;
};

// Global state and mutexes
// Use unique_ptr to guarantee stable memory addresses for InstanceData and DeviceData across rehashes
std::mutex g_state_mutex;
std::unordered_map<void *, std::unique_ptr<InstanceData>> g_instances;
std::unordered_map<void *, InstanceData *> g_physical_device_to_instance;
std::unordered_map<void *, std::unique_ptr<DeviceData>> g_devices;
std::unordered_map<void *, DeviceData *> g_queue_to_device;
std::unordered_map<VkSwapchainKHR, SwapchainInfo> g_swapchains;
DeviceData *g_primary_device = nullptr;
InstanceData *g_primary_instance = nullptr;

// Reporting and telemetry
std::string g_base_dir;
std::string g_log_path;
std::string g_status_path;
std::string g_shm_path;

int g_shm_fd = -1;
VdLayerShmHeader *g_shm_header = nullptr;
FILE *g_log_file = nullptr;

std::atomic<uint64_t> g_frame_counter{0};
std::mutex g_fps_mutex;
uint64_t g_last_fps_time_ns = 0;
uint64_t g_last_fps_frame_count = 0;
float g_current_fps = 0.0f;
bool g_is_disabled = false;
bool g_initialized = false;
bool g_verbose_log = false;

uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void log_msg(const char *format, ...) {
    if (g_is_disabled) return;
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    uint64_t now = get_time_ns();
    double sec = (double)now / 1e9;

    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (g_log_file) {
        fprintf(g_log_file, "[VD_LAYER %.3f] %s\n", sec, buffer);
        fflush(g_log_file);
    }
}

void log_debug(const char *format, ...) {
    if (!g_verbose_log || g_is_disabled) return;
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    uint64_t now = get_time_ns();
    double sec = (double)now / 1e9;

    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (g_log_file) {
        fprintf(g_log_file, "[VD_LAYER %.3f] %s\n", sec, buffer);
        fflush(g_log_file);
    }
}

std::string get_process_name() {
    char comm[64] = {0};
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) {
        if (fgets(comm, sizeof(comm), f)) {
            comm[strcspn(comm, "\r\n")] = '\0';
        }
        fclose(f);
    }
    if (comm[0] != '\0') return std::string(comm);

    char path[256] = {0};
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len > 0) {
        path[len] = '\0';
        const char *slash = strrchr(path, '/');
        return slash ? std::string(slash + 1) : std::string(path);
    }
    return "unknown";
}

void init_layer_runtime() {
    if (g_initialized) return;
    g_initialized = true;

    const char *disable_env = getenv("DISABLE_VD_LAYER");
    if (disable_env && (strcmp(disable_env, "1") == 0 || strcmp(disable_env, "true") == 0)) {
        g_is_disabled = true;
        return;
    }

    const char *verbose_env = getenv("VD_LAYER_DEBUG");
    if (!verbose_env) verbose_env = getenv("VD_LAYER_VERBOSE");
    if (verbose_env && (strcmp(verbose_env, "1") == 0 || strcmp(verbose_env, "true") == 0)) {
        g_verbose_log = true;
    }

    const char *custom_dir = getenv("VD_LAYER_DIR");
    if (custom_dir && custom_dir[0] != '\0') {
        g_base_dir = custom_dir;
    } else {
        const char *xdg_data = getenv("XDG_DATA_HOME");
        if (xdg_data && xdg_data[0] != '\0') {
            g_base_dir = std::string(xdg_data) + "/vulkan";
        } else {
            const char *home = getenv("HOME");
            if (home && home[0] != '\0') {
                g_base_dir = std::string(home) + "/.var/app/org.vinegarhq.Sober/data/vulkan";
            } else {
                g_base_dir = "/home/oae/.var/app/org.vinegarhq.Sober/data/vulkan";
            }
        }
    }

    // Ensure directory exists
    mkdir(g_base_dir.c_str(), 0755);

    g_log_path = g_base_dir + "/vd_layer.log";
    g_status_path = g_base_dir + "/vd_layer_status.json";
    g_shm_path = g_base_dir + "/vd_layer_shm.dat";

    g_log_file = fopen(g_log_path.c_str(), "a");
    pid_t pid = getpid();
    std::string proc_name = get_process_name();
    log_msg("=== VD Vulkan Capture Layer initialized for PID %d (%s) ===", (int)pid, proc_name.c_str());
    log_msg("Directory: %s (Verbose debug: %s)", g_base_dir.c_str(), g_verbose_log ? "ON" : "OFF");

    // Setup Shared Memory File
    g_shm_fd = open(g_shm_path.c_str(), O_RDWR | O_CREAT, 0666);
    if (g_shm_fd >= 0) {
        size_t shm_size = sizeof(VdLayerShmHeader);
        if (ftruncate(g_shm_fd, shm_size) == 0) {
            void *addr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
            if (addr != MAP_FAILED) {
                g_shm_header = (VdLayerShmHeader *)addr;
                
                // If existing header belongs to another active process that is still running, avoid clobbering
                bool is_stale = true;
                if (g_shm_header->magic == VD_LAYER_SHM_MAGIC && g_shm_header->pid != 0 && g_shm_header->pid != (uint32_t)pid) {
                    if (kill((pid_t)g_shm_header->pid, 0) == 0) {
                        is_stale = false; // another active process is running
                    }
                }

                if (is_stale || g_shm_header->magic != VD_LAYER_SHM_MAGIC) {
                    memset(g_shm_header, 0, shm_size);
                    g_shm_header->magic = VD_LAYER_SHM_MAGIC;
                    g_shm_header->version = 1;
                    g_shm_header->struct_size = sizeof(VdLayerShmHeader);
                    g_shm_header->status_flags = 1; // Active
                    g_shm_header->pid = (uint32_t)pid;
                    g_shm_header->sequence = 0;
                    g_shm_header->init_timestamp_ns = get_time_ns();
                    g_shm_header->last_present_ns = g_shm_header->init_timestamp_ns;
                    strncpy(g_shm_header->process_name, proc_name.c_str(), sizeof(g_shm_header->process_name) - 1);
                    msync(g_shm_header, sizeof(VdLayerShmHeader), MS_ASYNC);
                } else {
                    // Update PID and flags
                    g_shm_header->pid = (uint32_t)pid;
                    g_shm_header->status_flags |= 1;
                    strncpy(g_shm_header->process_name, proc_name.c_str(), sizeof(g_shm_header->process_name) - 1);
                }
                log_msg("Shared memory mapped at %s (size %zu)", g_shm_path.c_str(), shm_size);
            } else {
                log_msg("mmap failed for %s", g_shm_path.c_str());
            }
        } else {
            log_msg("ftruncate failed for %s", g_shm_path.c_str());
        }
    } else {
        log_msg("open failed for shm file %s", g_shm_path.c_str());
    }

    g_last_fps_time_ns = get_time_ns();
}

void write_status_json(bool active = true) {
    if (g_is_disabled || g_status_path.empty()) return;

    pid_t pid = getpid();
    std::string tmp_path = g_status_path + "." + std::to_string(pid) + ".tmp";
    FILE *f = fopen(tmp_path.c_str(), "w");
    if (!f) return;

    std::string proc_name = get_process_name();
    uint64_t now = get_time_ns();

    uint32_t width = 0, height = 0, format = 0, present_mode = 0, img_count = 0;
    float fps = 0.0f;
    {
        std::lock_guard<std::mutex> lock(g_fps_mutex);
        fps = g_current_fps;
    }

    if (g_shm_header) {
        width = g_shm_header->swapchain_width;
        height = g_shm_header->swapchain_height;
        format = g_shm_header->swapchain_format;
        present_mode = g_shm_header->swapchain_present_mode;
        img_count = g_shm_header->swapchain_image_count;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"layer_name\": \"%s\",\n", VD_LAYER_NAME);
    fprintf(f, "  \"version\": %d,\n", VD_LAYER_IMPL_VERSION);
    fprintf(f, "  \"active\": %s,\n", active ? "true" : "false");
    fprintf(f, "  \"pid\": %d,\n", (int)pid);
    fprintf(f, "  \"process_name\": \"%s\",\n", proc_name.c_str());
    fprintf(f, "  \"frames_presented\": %llu,\n", (unsigned long long)g_frame_counter.load(std::memory_order_relaxed));
    fprintf(f, "  \"fps\": %.2f,\n", fps);
    fprintf(f, "  \"last_present_ns\": %llu,\n", (unsigned long long)now);
    fprintf(f, "  \"swapchain\": {\n");
    fprintf(f, "    \"width\": %u,\n", width);
    fprintf(f, "    \"height\": %u,\n", height);
    fprintf(f, "    \"format\": %u,\n", format);
    fprintf(f, "    \"present_mode\": %u,\n", present_mode);
    fprintf(f, "    \"image_count\": %u\n", img_count);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);

    rename(tmp_path.c_str(), g_status_path.c_str());
}

VkLayerInstanceCreateInfo *get_instance_chain_info(const VkInstanceCreateInfo *pCreateInfo, VkLayerFunction func) {
    auto *chain_info = (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
    while (chain_info) {
        if (chain_info->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
            chain_info->function == func) {
            return chain_info;
        }
        chain_info = (VkLayerInstanceCreateInfo *)chain_info->pNext;
    }
    return nullptr;
}

VkLayerDeviceCreateInfo *get_device_chain_info(const VkDeviceCreateInfo *pCreateInfo, VkLayerFunction func) {
    auto *chain_info = (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
    while (chain_info) {
        if (chain_info->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
            chain_info->function == func) {
            return chain_info;
        }
        chain_info = (VkLayerDeviceCreateInfo *)chain_info->pNext;
    }
    return nullptr;
}

} // namespace

// Intercepted Vulkan Functions Declarations
VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateInstance(
    const VkInstanceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkInstance *pInstance) {
    init_layer_runtime();
    log_msg("vd_vkCreateInstance called");

    VkLayerInstanceCreateInfo *chain_info = get_instance_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain_info || !chain_info->u.pLayerInfo) {
        log_msg("ERROR: No instance link info found in pCreateInfo");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr next_gipa = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_GetPhysicalDeviceProcAddr next_gpdpa = chain_info->u.pLayerInfo->pfnNextGetPhysicalDeviceProcAddr;

    PFN_vkCreateInstance next_create_instance =
        (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!next_create_instance) {
        log_msg("ERROR: Failed to resolve next vkCreateInstance");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance link info pointer for next layer down the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult res = next_create_instance(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) {
        log_msg("next_create_instance failed with code %d", (int)res);
        return res;
    }

    auto data = std::make_unique<InstanceData>();
    data->instance = *pInstance;
    data->next_gipa = next_gipa;
    data->next_gpdpa = next_gpdpa;
    data->vkDestroyInstance = (PFN_vkDestroyInstance)next_gipa(*pInstance, "vkDestroyInstance");
    data->vkCreateDevice = (PFN_vkCreateDevice)next_gipa(*pInstance, "vkCreateDevice");
    data->vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)next_gipa(*pInstance, "vkEnumeratePhysicalDevices");
    data->vkEnumeratePhysicalDeviceGroups = (PFN_vkEnumeratePhysicalDeviceGroups)next_gipa(*pInstance, "vkEnumeratePhysicalDeviceGroups");
    data->vkEnumeratePhysicalDeviceGroupsKHR = (PFN_vkEnumeratePhysicalDeviceGroupsKHR)next_gipa(*pInstance, "vkEnumeratePhysicalDeviceGroupsKHR");
    data->vkEnumerateDeviceExtensionProperties =
        (PFN_vkEnumerateDeviceExtensionProperties)next_gipa(*pInstance, "vkEnumerateDeviceExtensionProperties");

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        InstanceData *ptr = data.get();
        g_instances[(void *)(*pInstance)] = std::move(data);
        g_primary_instance = ptr;
    }

    log_msg("Instance %p created successfully", (void *)(*pInstance));
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroyInstance(
    VkInstance instance,
    const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroyInstance called for %p", (void *)instance);

    PFN_vkDestroyInstance next_destroy = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find((void *)instance);
        if (it != g_instances.end()) {
            next_destroy = it->second->vkDestroyInstance;
            InstanceData *inst_ptr = it->second.get();

            // Clean up physical devices mapped to this instance
            for (auto p_it = g_physical_device_to_instance.begin(); p_it != g_physical_device_to_instance.end();) {
                if (p_it->second == inst_ptr) {
                    p_it = g_physical_device_to_instance.erase(p_it);
                } else {
                    ++p_it;
                }
            }
            g_instances.erase(it);
            if (g_primary_instance == inst_ptr) {
                g_primary_instance = g_instances.empty() ? nullptr : g_instances.begin()->second.get();
            }
        }
    }

    if (g_instances.empty()) {
        if (g_shm_header) {
            g_shm_header->sequence++;
            g_shm_header->status_flags = 0; // inactive
            g_shm_header->sequence++;
            msync(g_shm_header, sizeof(VdLayerShmHeader), MS_ASYNC);
        }
        write_status_json(false);
    }

    if (next_destroy) {
        next_destroy(instance, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumeratePhysicalDevices(
    VkInstance instance,
    uint32_t *pPhysicalDeviceCount,
    VkPhysicalDevice *pPhysicalDevices) {
    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find((void *)instance);
        if (it != g_instances.end()) idata = it->second.get();
        else idata = g_primary_instance;
    }

    if (!idata || !idata->vkEnumeratePhysicalDevices) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult res = idata->vkEnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
    if ((res == VK_SUCCESS || res == VK_INCOMPLETE) && pPhysicalDevices && pPhysicalDeviceCount) {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        for (uint32_t i = 0; i < *pPhysicalDeviceCount; ++i) {
            if (pPhysicalDevices[i] != VK_NULL_HANDLE) {
                g_physical_device_to_instance[(void *)pPhysicalDevices[i]] = idata;
            }
        }
    }
    return res;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumeratePhysicalDeviceGroups(
    VkInstance instance,
    uint32_t *pPhysicalDeviceGroupCount,
    VkPhysicalDeviceGroupProperties *pPhysicalDeviceGroupProperties) {
    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find((void *)instance);
        if (it != g_instances.end()) idata = it->second.get();
        else idata = g_primary_instance;
    }

    PFN_vkEnumeratePhysicalDeviceGroups func = idata ? idata->vkEnumeratePhysicalDeviceGroups : nullptr;
    if (!func && idata) func = idata->vkEnumeratePhysicalDeviceGroupsKHR;
    if (!func) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = func(instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties);
    if ((res == VK_SUCCESS || res == VK_INCOMPLETE) && pPhysicalDeviceGroupProperties && pPhysicalDeviceGroupCount) {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        for (uint32_t g = 0; g < *pPhysicalDeviceGroupCount; ++g) {
            for (uint32_t i = 0; i < pPhysicalDeviceGroupProperties[g].physicalDeviceCount; ++i) {
                VkPhysicalDevice dev = pPhysicalDeviceGroupProperties[g].physicalDevices[i];
                if (dev != VK_NULL_HANDLE) {
                    g_physical_device_to_instance[(void *)dev] = idata;
                }
            }
        }
    }
    return res;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDevice *pDevice) {
    log_msg("vd_vkCreateDevice called for physicalDevice %p", (void *)physicalDevice);

    VkLayerDeviceCreateInfo *chain_info = get_device_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain_info || !chain_info->u.pLayerInfo) {
        log_msg("ERROR: No device link info found in pCreateInfo");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr next_gipa = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_gdpa = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;

    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_physical_device_to_instance.find((void *)physicalDevice);
        if (it != g_physical_device_to_instance.end()) idata = it->second;
        else idata = g_primary_instance;
    }

    VkInstance inst = idata ? idata->instance : VK_NULL_HANDLE;
    PFN_vkCreateDevice next_create_device = nullptr;
    if (next_gipa) {
        next_create_device = (PFN_vkCreateDevice)next_gipa(inst, "vkCreateDevice");
    }
    if (!next_create_device && idata) {
        next_create_device = idata->vkCreateDevice;
    }
    if (!next_create_device) {
        log_msg("ERROR: Failed to resolve next vkCreateDevice");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance link info pointer for next layer down the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult res = next_create_device(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (res != VK_SUCCESS) {
        log_msg("next_create_device failed with code %d", (int)res);
        return res;
    }

    auto data = std::make_unique<DeviceData>();
    data->device = *pDevice;
    data->instance_data = idata;
    data->next_gdpa = next_gdpa;
    data->vkDestroyDevice = (PFN_vkDestroyDevice)next_gdpa(*pDevice, "vkDestroyDevice");
    data->vkGetDeviceQueue = (PFN_vkGetDeviceQueue)next_gdpa(*pDevice, "vkGetDeviceQueue");
    data->vkGetDeviceQueue2 = (PFN_vkGetDeviceQueue2)next_gdpa(*pDevice, "vkGetDeviceQueue2");
    data->vkCreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)next_gdpa(*pDevice, "vkCreateSwapchainKHR");
    data->vkDestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)next_gdpa(*pDevice, "vkDestroySwapchainKHR");
    data->vkGetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)next_gdpa(*pDevice, "vkGetSwapchainImagesKHR");
    data->vkQueuePresentKHR = (PFN_vkQueuePresentKHR)next_gdpa(*pDevice, "vkQueuePresentKHR");

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        DeviceData *ptr = data.get();
        g_devices[(void *)(*pDevice)] = std::move(data);
        g_primary_device = ptr;
    }

    log_msg("Device %p created successfully (Swapchain support: %s)",
            (void *)(*pDevice),
            g_primary_device->vkCreateSwapchainKHR ? "YES" : "NO");
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroyDevice called for %p", (void *)device);

    PFN_vkDestroyDevice next_destroy = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find((void *)device);
        if (it != g_devices.end()) {
            next_destroy = it->second->vkDestroyDevice;
            DeviceData *dev_ptr = it->second.get();

            // Clean up queue mappings for this device
            for (auto q_it = g_queue_to_device.begin(); q_it != g_queue_to_device.end();) {
                if (q_it->second == dev_ptr) {
                    q_it = g_queue_to_device.erase(q_it);
                } else {
                    ++q_it;
                }
            }
            g_devices.erase(it);
            if (g_primary_device == dev_ptr) {
                g_primary_device = g_devices.empty() ? nullptr : g_devices.begin()->second.get();
            }
        }
    }

    if (next_destroy) {
        next_destroy(device, pAllocator);
    }
}

VKAPI_ATTR void VKAPI_CALL vd_vkGetDeviceQueue(
    VkDevice device,
    uint32_t queueFamilyIndex,
    uint32_t queueIndex,
    VkQueue *pQueue) {
    DeviceData *ddata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find((void *)device);
        if (it != g_devices.end()) ddata = it->second.get();
        else ddata = g_primary_device;
    }

    if (ddata && ddata->vkGetDeviceQueue) {
        ddata->vkGetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
        if (pQueue && *pQueue != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_queue_to_device[(void *)(*pQueue)] = ddata;
            log_debug("Registered queue %p to device %p", (void *)(*pQueue), (void *)device);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vd_vkGetDeviceQueue2(
    VkDevice device,
    const VkDeviceQueueInfo2 *pQueueInfo,
    VkQueue *pQueue) {
    DeviceData *ddata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find((void *)device);
        if (it != g_devices.end()) ddata = it->second.get();
        else ddata = g_primary_device;
    }

    if (ddata && ddata->vkGetDeviceQueue2) {
        ddata->vkGetDeviceQueue2(device, pQueueInfo, pQueue);
        if (pQueue && *pQueue != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_queue_to_device[(void *)(*pQueue)] = ddata;
            log_debug("Registered queue2 %p to device %p", (void *)(*pQueue), (void *)device);
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateSwapchainKHR(
    VkDevice device,
    const VkSwapchainCreateInfoKHR *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkSwapchainKHR *pSwapchain) {
    log_msg("vd_vkCreateSwapchainKHR called: %ux%u format=%u mode=%u minImages=%u",
            pCreateInfo->imageExtent.width,
            pCreateInfo->imageExtent.height,
            pCreateInfo->imageFormat,
            pCreateInfo->presentMode,
            pCreateInfo->minImageCount);

    DeviceData *ddata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find((void *)device);
        if (it != g_devices.end()) ddata = it->second.get();
        else ddata = g_primary_device;
    }

    if (!ddata || !ddata->vkCreateSwapchainKHR) {
        log_msg("ERROR: DeviceData or vkCreateSwapchainKHR missing for device %p", (void *)device);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult res = ddata->vkCreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
    if (res != VK_SUCCESS) {
        log_msg("vkCreateSwapchainKHR failed with code %d", (int)res);
        return res;
    }

    uint32_t image_count = 0;
    if (ddata->vkGetSwapchainImagesKHR) {
        ddata->vkGetSwapchainImagesKHR(device, *pSwapchain, &image_count, nullptr);
    }

    SwapchainInfo sc;
    sc.swapchain = *pSwapchain;
    sc.device = device;
    sc.width = pCreateInfo->imageExtent.width;
    sc.height = pCreateInfo->imageExtent.height;
    sc.format = pCreateInfo->imageFormat;
    sc.present_mode = pCreateInfo->presentMode;
    sc.image_count = image_count;

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_swapchains[*pSwapchain] = sc;
    }

    if (g_shm_header) {
        g_shm_header->sequence++;
        g_shm_header->status_flags |= 2; // Swapchain ready
        g_shm_header->swapchain_width = sc.width;
        g_shm_header->swapchain_height = sc.height;
        g_shm_header->swapchain_format = (uint32_t)sc.format;
        g_shm_header->swapchain_present_mode = (uint32_t)sc.present_mode;
        g_shm_header->swapchain_image_count = sc.image_count;
        g_shm_header->sequence++;
        msync(g_shm_header, sizeof(VdLayerShmHeader), MS_ASYNC);
    }

    log_msg("Swapchain %p created: %ux%u, format %u, images %u",
            (void *)(*pSwapchain), sc.width, sc.height, sc.format, sc.image_count);
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroySwapchainKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroySwapchainKHR called for %p", (void *)swapchain);

    DeviceData *ddata = nullptr;
    PFN_vkDestroySwapchainKHR next_destroy = nullptr;
    bool has_active_swapchain = false;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_swapchains.erase(swapchain);
        has_active_swapchain = !g_swapchains.empty();

        auto it = g_devices.find((void *)device);
        if (it != g_devices.end()) {
            ddata = it->second.get();
            next_destroy = ddata->vkDestroySwapchainKHR;
        } else if (g_primary_device) {
            next_destroy = g_primary_device->vkDestroySwapchainKHR;
        }
    }

    if (g_shm_header && !has_active_swapchain) {
        g_shm_header->sequence++;
        g_shm_header->status_flags &= ~2; // Swapchain inactive
        g_shm_header->sequence++;
        msync(g_shm_header, sizeof(VdLayerShmHeader), MS_ASYNC);
    }
    write_status_json(true);

    if (next_destroy) {
        next_destroy(device, swapchain, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkQueuePresentKHR(
    VkQueue queue,
    const VkPresentInfoKHR *pPresentInfo) {
    DeviceData *ddata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_queue_to_device.find((void *)queue);
        if (it != g_queue_to_device.end()) {
            ddata = it->second;
        } else {
            ddata = g_primary_device;
        }
    }

    if (!ddata || !ddata->vkQueuePresentKHR) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    uint64_t frame = g_frame_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t now_ns = get_time_ns();

    // Fast status update in mmap header with sequence counter
    if (g_shm_header) {
        g_shm_header->sequence++;
        g_shm_header->frame_count = frame;
        g_shm_header->last_present_ns = now_ns;
        if (pPresentInfo && pPresentInfo->swapchainCount > 0 && pPresentInfo->pImageIndices) {
            g_shm_header->current_image_index = pPresentInfo->pImageIndices[0];
        }
        g_shm_header->sequence++;
    }

    // Periodic FPS calculation and status file dump
    bool should_dump_json = (frame == 1);
    {
        std::lock_guard<std::mutex> lock(g_fps_mutex);
        if (frame % 60 == 0 || (now_ns - g_last_fps_time_ns) >= 1000000000ULL) {
            uint64_t elapsed_ns = now_ns - g_last_fps_time_ns;
            if (elapsed_ns > 0) {
                uint64_t frames_delta = frame - g_last_fps_frame_count;
                g_current_fps = (float)((double)frames_delta * 1e9 / (double)elapsed_ns);
                if (g_shm_header) {
                    g_shm_header->current_fps = g_current_fps;
                }
            }
            g_last_fps_time_ns = now_ns;
            g_last_fps_frame_count = frame;
            should_dump_json = true;
        }
    }

    if (should_dump_json) {
        write_status_json(true);
        if (frame % 300 == 0) {
            log_msg("Present heartbeat: frame %llu (%.1f fps)",
                    (unsigned long long)frame, g_current_fps);
        }
    }

    return ddata->vkQueuePresentKHR(queue, pPresentInfo);
}

// Layer Properties Functions
VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateInstanceLayerProperties(
    uint32_t *pPropertyCount,
    VkLayerProperties *pProperties) {
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pProperties) {
        *pPropertyCount = 1;
        return VK_SUCCESS;
    }
    if (*pPropertyCount < 1) {
        return VK_INCOMPLETE;
    }
    *pPropertyCount = 1;
    strncpy(pProperties[0].layerName, VD_LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
    pProperties[0].layerName[VK_MAX_EXTENSION_NAME_SIZE - 1] = '\0';
    pProperties[0].specVersion = VD_LAYER_SPEC_VERSION;
    pProperties[0].implementationVersion = VD_LAYER_IMPL_VERSION;
    strncpy(pProperties[0].description, VD_LAYER_DESCRIPTION, VK_MAX_DESCRIPTION_SIZE - 1);
    pProperties[0].description[VK_MAX_DESCRIPTION_SIZE - 1] = '\0';
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateInstanceExtensionProperties(
    const char *pLayerName,
    uint32_t *pPropertyCount,
    VkExtensionProperties *pProperties) {
    (void)pProperties;
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (pLayerName && strcmp(pLayerName, VD_LAYER_NAME) == 0) {
        *pPropertyCount = 0;
        return VK_SUCCESS;
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t *pPropertyCount,
    VkLayerProperties *pProperties) {
    (void)physicalDevice;
    return vd_vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice,
    const char *pLayerName,
    uint32_t *pPropertyCount,
    VkExtensionProperties *pProperties) {
    (void)pProperties;
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (pLayerName && strcmp(pLayerName, VD_LAYER_NAME) == 0) {
        *pPropertyCount = 0;
        return VK_SUCCESS;
    }

    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_physical_device_to_instance.find((void *)physicalDevice);
        if (it != g_physical_device_to_instance.end()) idata = it->second;
        else idata = g_primary_instance;
    }

    if (idata && idata->vkEnumerateDeviceExtensionProperties) {
        return idata->vkEnumerateDeviceExtensionProperties(
            physicalDevice, pLayerName, pPropertyCount, pProperties);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vd_vk_layerGetPhysicalDeviceProcAddr(
    VkInstance instance,
    const char *pName) {
    if (!pName) return nullptr;
    log_debug("vd_vk_layerGetPhysicalDeviceProcAddr: instance=%p, pName=%s", (void *)instance, pName);

    if (strcmp(pName, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)vd_vkCreateDevice;
    if (strcmp(pName, "vkEnumerateDeviceLayerProperties") == 0)
        return (PFN_vkVoidFunction)vd_vkEnumerateDeviceLayerProperties;
    if (strcmp(pName, "vkEnumerateDeviceExtensionProperties") == 0)
        return (PFN_vkVoidFunction)vd_vkEnumerateDeviceExtensionProperties;

    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find((void *)instance);
        if (it != g_instances.end()) idata = it->second.get();
        else idata = g_primary_instance;
    }
    if (idata) {
        if (idata->next_gpdpa) {
            return idata->next_gpdpa(instance, pName);
        }
        if (idata->next_gipa) {
            return idata->next_gipa(instance, pName);
        }
    }
    return nullptr;
}

// Entry Point GetProcAddr Functions
extern "C" {

VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(
    VkInstance instance,
    const char *pName) {
    if (!pName) return nullptr;
    log_debug("vkGetInstanceProcAddr: instance=%p, pName=%s", (void *)instance, pName);

    // Global entry points (callable with instance == VK_NULL_HANDLE)
    if (strcmp(pName, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (strcmp(pName, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (strcmp(pName, "vkNegotiateLoaderLayerInterfaceVersion") == 0)
        return (PFN_vkVoidFunction)vkNegotiateLoaderLayerInterfaceVersion;
    if (strcmp(pName, "vkCreateInstance") == 0) return (PFN_vkVoidFunction)vd_vkCreateInstance;
    if (strcmp(pName, "vkEnumerateInstanceLayerProperties") == 0)
        return (PFN_vkVoidFunction)vd_vkEnumerateInstanceLayerProperties;
    if (strcmp(pName, "vkEnumerateInstanceExtensionProperties") == 0)
        return (PFN_vkVoidFunction)vd_vkEnumerateInstanceExtensionProperties;
    if (strcmp(pName, "vk_layerGetPhysicalDeviceProcAddr") == 0)
        return (PFN_vkVoidFunction)vd_vk_layerGetPhysicalDeviceProcAddr;

    // If instance is null, do not return instance- or device-level functions
    if (instance == VK_NULL_HANDLE) {
        return nullptr;
    }

    if (strcmp(pName, "vkDestroyInstance") == 0) return (PFN_vkVoidFunction)vd_vkDestroyInstance;
    if (strcmp(pName, "vkEnumeratePhysicalDevices") == 0) return (PFN_vkVoidFunction)vd_vkEnumeratePhysicalDevices;
    if (strcmp(pName, "vkEnumeratePhysicalDeviceGroups") == 0) return (PFN_vkVoidFunction)vd_vkEnumeratePhysicalDeviceGroups;
    if (strcmp(pName, "vkEnumeratePhysicalDeviceGroupsKHR") == 0) return (PFN_vkVoidFunction)vd_vkEnumeratePhysicalDeviceGroups;
    if (strcmp(pName, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)vd_vkCreateDevice;
    if (strcmp(pName, "vkEnumerateDeviceLayerProperties") == 0)
        return (PFN_vkVoidFunction)vd_vkEnumerateDeviceLayerProperties;
    if (strcmp(pName, "vkEnumerateDeviceExtensionProperties") == 0)
        return (PFN_vkVoidFunction)vd_vkEnumerateDeviceExtensionProperties;

    // Device level functions that may be intercepted via instance GIPA
    if (strcmp(pName, "vkDestroyDevice") == 0) return (PFN_vkVoidFunction)vd_vkDestroyDevice;
    if (strcmp(pName, "vkGetDeviceQueue") == 0) return (PFN_vkVoidFunction)vd_vkGetDeviceQueue;
    if (strcmp(pName, "vkGetDeviceQueue2") == 0) return (PFN_vkVoidFunction)vd_vkGetDeviceQueue2;
    if (strcmp(pName, "vkCreateSwapchainKHR") == 0) return (PFN_vkVoidFunction)vd_vkCreateSwapchainKHR;
    if (strcmp(pName, "vkDestroySwapchainKHR") == 0) return (PFN_vkVoidFunction)vd_vkDestroySwapchainKHR;
    if (strcmp(pName, "vkQueuePresentKHR") == 0) return (PFN_vkVoidFunction)vd_vkQueuePresentKHR;

    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find((void *)instance);
        if (it != g_instances.end()) idata = it->second.get();
        else idata = g_primary_instance;
    }

    if (idata && idata->next_gipa) {
        return idata->next_gipa(instance, pName);
    }
    return nullptr;
}

VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(
    VkDevice device,
    const char *pName) {
    if (!pName) return nullptr;

    if (strcmp(pName, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (strcmp(pName, "vkDestroyDevice") == 0) return (PFN_vkVoidFunction)vd_vkDestroyDevice;
    if (strcmp(pName, "vkGetDeviceQueue") == 0) return (PFN_vkVoidFunction)vd_vkGetDeviceQueue;
    if (strcmp(pName, "vkGetDeviceQueue2") == 0) return (PFN_vkVoidFunction)vd_vkGetDeviceQueue2;

    DeviceData *ddata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find((void *)device);
        if (it != g_devices.end()) ddata = it->second.get();
        else ddata = g_primary_device;
    }

    // Only return swapchain hooks if swapchain was enabled and resolved on this device
    if (strcmp(pName, "vkCreateSwapchainKHR") == 0) {
        return (ddata && ddata->vkCreateSwapchainKHR) ? (PFN_vkVoidFunction)vd_vkCreateSwapchainKHR : nullptr;
    }
    if (strcmp(pName, "vkDestroySwapchainKHR") == 0) {
        return (ddata && ddata->vkDestroySwapchainKHR) ? (PFN_vkVoidFunction)vd_vkDestroySwapchainKHR : nullptr;
    }
    if (strcmp(pName, "vkQueuePresentKHR") == 0) {
        return (ddata && ddata->vkQueuePresentKHR) ? (PFN_vkVoidFunction)vd_vkQueuePresentKHR : nullptr;
    }

    if (ddata && ddata->next_gdpa) {
        return ddata->next_gdpa(device, pName);
    }
    return nullptr;
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(
    VkNegotiateLayerInterface *pVersionStruct) {
    init_layer_runtime();
    log_msg("vkNegotiateLoaderLayerInterfaceVersion called, loader version: %u",
            pVersionStruct ? pVersionStruct->loaderLayerInterfaceVersion : 0);

    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        log_msg("ERROR: Invalid VkNegotiateLayerInterface struct");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
    } else if (pVersionStruct->loaderLayerInterfaceVersion == 1) {
        pVersionStruct->loaderLayerInterfaceVersion = 1;
    } else {
        log_msg("ERROR: Unsupported loader interface version %u", pVersionStruct->loaderLayerInterfaceVersion);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    pVersionStruct->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = vd_vk_layerGetPhysicalDeviceProcAddr;

    log_msg("Negotiation successful (interface version 2, pfnGetPhysicalDeviceProcAddr bound)");
    return VK_SUCCESS;
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t *pPropertyCount,
    VkLayerProperties *pProperties) {
    return vd_vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char *pLayerName,
    uint32_t *pPropertyCount,
    VkExtensionProperties *pProperties) {
    return vd_vkEnumerateInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties);
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t *pPropertyCount,
    VkLayerProperties *pProperties) {
    return vd_vkEnumerateDeviceLayerProperties(physicalDevice, pPropertyCount, pProperties);
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice,
    const char *pLayerName,
    uint32_t *pPropertyCount,
    VkExtensionProperties *pProperties) {
    return vd_vkEnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
}

VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(
    VkInstance instance,
    const char *pName) {
    return vd_vk_layerGetPhysicalDeviceProcAddr(instance, pName);
}

} // extern "C"
