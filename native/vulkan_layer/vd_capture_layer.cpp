#include "vd_capture_layer.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <iterator>
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

constexpr uint64_t kDefaultCaptureIntervalNs = 200000000ULL; // 5 FPS
constexpr uint32_t kMaxCaptureSemaphores = 8;

struct InstanceData {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr next_gipa = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkEnumeratePhysicalDeviceGroups vkEnumeratePhysicalDeviceGroups = nullptr;
    PFN_vkEnumeratePhysicalDeviceGroupsKHR vkEnumeratePhysicalDeviceGroupsKHR = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
};

struct CaptureResources {
    bool initialized = false;
    bool permanently_disabled = false;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    bool memory_coherent = false;
    VkFence fence = VK_NULL_HANDLE;
    std::vector<VkSemaphore> present_semaphores;
};

struct DeviceData {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
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

    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkResetCommandPool vkResetCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkMapMemory vkMapMemory = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory = nullptr;
    PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkResetFences vkResetFences = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;
    PFN_vkCreateSemaphore vkCreateSemaphore = nullptr;
    PFN_vkDestroySemaphore vkDestroySemaphore = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer = nullptr;
};

struct QueueData {
    DeviceData *device_data = nullptr;
    uint32_t family_index = 0;
    std::mutex capture_mutex;
    CaptureResources capture;
};

struct SwapchainInfo {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    uint32_t image_count = 0;
    std::vector<VkImage> images;
    bool capture_supported = false;
    uint64_t last_capture_ns = 0;
};

std::mutex g_state_mutex;
std::mutex g_log_mutex;
std::mutex g_init_mutex;
std::mutex g_fps_mutex;
std::unordered_map<void *, std::unique_ptr<InstanceData>> g_instances;
std::unordered_map<void *, InstanceData *> g_physical_device_to_instance;
std::unordered_map<void *, std::unique_ptr<DeviceData>> g_devices;
std::unordered_map<void *, std::unique_ptr<QueueData>> g_queues;
std::unordered_map<VkSwapchainKHR, SwapchainInfo> g_swapchains;
DeviceData *g_primary_device = nullptr;
InstanceData *g_primary_instance = nullptr;

std::string g_base_dir;
std::string g_log_path;
std::string g_status_path;
std::string g_shm_path;
int g_shm_fd = -1;
void *g_shm_mapping = nullptr;
size_t g_shm_mapping_size = 0;
VdLayerShmHeader *g_shm_header = nullptr;
uint8_t *g_shm_pixels = nullptr;
FILE *g_log_file = nullptr;

std::atomic<uint64_t> g_frame_counter{0};
std::atomic<uint64_t> g_capture_counter{0};
uint64_t g_last_fps_time_ns = 0;
uint64_t g_last_fps_frame_count = 0;
float g_current_fps = 0.0f;
bool g_is_disabled = false;
bool g_initialized = false;
bool g_verbose_log = false;
bool g_capture_enabled = false;
uint64_t g_capture_interval_ns = kDefaultCaptureIntervalNs;

uint64_t get_time_ns() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

bool env_true(const char *name) {
    const char *value = getenv(name);
    if (!value) return false;
    return strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "yes") == 0 || strcmp(value, "on") == 0;
}

void log_line(bool debug_only, const char *format, va_list args) {
    if (g_is_disabled || (debug_only && !g_verbose_log)) return;
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    const double sec = static_cast<double>(get_time_ns()) / 1e9;

    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_file) {
        fprintf(g_log_file, "[VD_LAYER %.3f] %s\n", sec, buffer);
        fflush(g_log_file);
    }
    if (g_verbose_log) {
        fprintf(stderr, "[VD_LAYER %.3f] %s\n", sec, buffer);
        fflush(stderr);
    }
}

void log_msg(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_line(false, format, args);
    va_end(args);
}

void log_debug(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_line(true, format, args);
    va_end(args);
}

std::string get_process_name() {
    char comm[64] = {0};
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) {
        if (fgets(comm, sizeof(comm), f)) comm[strcspn(comm, "\r\n")] = '\0';
        fclose(f);
    }
    if (comm[0]) return std::string(comm);

    char path[256] = {0};
    const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len > 0) {
        path[len] = '\0';
        const char *slash = strrchr(path, '/');
        return slash ? std::string(slash + 1) : std::string(path);
    }
    return "unknown";
}

void shm_begin_write() {
    if (g_shm_header) ++g_shm_header->sequence;
}

void shm_end_write() {
    if (g_shm_header) ++g_shm_header->sequence;
}

void increment_dropped_frames() {
    if (!g_shm_header) return;
    shm_begin_write();
    ++g_shm_header->dropped_frames;
    shm_end_write();
}

void init_layer_runtime() {
    std::lock_guard<std::mutex> init_lock(g_init_mutex);
    if (g_initialized) return;
    g_initialized = true;

    if (env_true("DISABLE_VD_LAYER")) {
        g_is_disabled = true;
        return;
    }

    g_verbose_log = env_true("VD_LAYER_DEBUG") || env_true("VD_LAYER_VERBOSE");
    g_capture_enabled = env_true("VD_CAPTURE_ENABLE");
    if (const char *interval_ms = getenv("VD_CAPTURE_INTERVAL_MS")) {
        char *end = nullptr;
        const unsigned long parsed = strtoul(interval_ms, &end, 10);
        if (end && *end == '\0') g_capture_interval_ns = static_cast<uint64_t>(parsed) * 1000000ULL;
    }

    const char *custom_dir = getenv("VD_LAYER_DIR");
    if (custom_dir && *custom_dir) {
        g_base_dir = custom_dir;
    } else if (const char *xdg_data = getenv("XDG_DATA_HOME"); xdg_data && *xdg_data) {
        g_base_dir = std::string(xdg_data) + "/vulkan";
    } else if (const char *home = getenv("HOME"); home && *home) {
        g_base_dir = std::string(home) + "/.var/app/org.vinegarhq.Sober/data/vulkan";
    } else {
        g_base_dir = "/home/oae/.var/app/org.vinegarhq.Sober/data/vulkan";
    }

    mkdir(g_base_dir.c_str(), 0755);
    g_log_path = g_base_dir + "/vd_layer.log";
    g_status_path = g_base_dir + "/vd_layer_status.json";
    g_shm_path = g_base_dir + "/vd_layer_shm.dat";
    g_log_file = fopen(g_log_path.c_str(), "a");

    const pid_t pid = getpid();
    const std::string proc_name = get_process_name();
    log_msg("=== VD Vulkan Capture Layer initialized for PID %d (%s) ===", static_cast<int>(pid), proc_name.c_str());
    log_msg("Directory: %s (Verbose: %s, Capture: %s, Interval: %.1f ms)",
            g_base_dir.c_str(), g_verbose_log ? "ON" : "OFF", g_capture_enabled ? "ON" : "OFF",
            static_cast<double>(g_capture_interval_ns) / 1000000.0);

    g_shm_fd = open(g_shm_path.c_str(), O_RDWR | O_CREAT, 0666);
    if (g_shm_fd >= 0) {
        g_shm_mapping_size = sizeof(VdLayerShmHeader) + VD_CAPTURE_MAX_BYTES;
        if (ftruncate(g_shm_fd, static_cast<off_t>(g_shm_mapping_size)) == 0) {
            g_shm_mapping = mmap(nullptr, g_shm_mapping_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
            if (g_shm_mapping != MAP_FAILED) {
                g_shm_header = static_cast<VdLayerShmHeader *>(g_shm_mapping);
                g_shm_pixels = static_cast<uint8_t *>(g_shm_mapping) + sizeof(VdLayerShmHeader);

                bool stale = true;
                if (g_shm_header->magic == VD_LAYER_SHM_MAGIC && g_shm_header->pid != 0 &&
                    g_shm_header->pid != static_cast<uint32_t>(pid)) {
                    stale = kill(static_cast<pid_t>(g_shm_header->pid), 0) != 0;
                }

                if (stale || g_shm_header->magic != VD_LAYER_SHM_MAGIC ||
                    g_shm_header->version != VD_LAYER_SHM_VERSION ||
                    g_shm_header->struct_size != sizeof(VdLayerShmHeader)) {
                    memset(g_shm_mapping, 0, g_shm_mapping_size);
                }

                shm_begin_write();
                g_shm_header->magic = VD_LAYER_SHM_MAGIC;
                g_shm_header->version = VD_LAYER_SHM_VERSION;
                g_shm_header->struct_size = sizeof(VdLayerShmHeader);
                g_shm_header->status_flags = VD_STATUS_ACTIVE | (g_capture_enabled ? VD_STATUS_CAPTURE_ENABLED : 0u);
                g_shm_header->pid = static_cast<uint32_t>(pid);
                g_shm_header->init_timestamp_ns = get_time_ns();
                g_shm_header->last_present_ns = g_shm_header->init_timestamp_ns;
                g_shm_header->roi_data_offset = sizeof(VdLayerShmHeader);
                strncpy(g_shm_header->process_name, proc_name.c_str(), sizeof(g_shm_header->process_name) - 1);
                g_shm_header->process_name[sizeof(g_shm_header->process_name) - 1] = '\0';
                shm_end_write();
                msync(g_shm_mapping, sizeof(VdLayerShmHeader), MS_ASYNC);
                log_msg("Shared memory mapped at %s (header=%zu, pixels=%u)",
                        g_shm_path.c_str(), sizeof(VdLayerShmHeader), VD_CAPTURE_MAX_BYTES);
            } else {
                g_shm_mapping = nullptr;
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
    const pid_t pid = getpid();
    const std::string tmp_path = g_status_path + "." + std::to_string(pid) + ".tmp";
    FILE *f = fopen(tmp_path.c_str(), "w");
    if (!f) return;

    float fps = 0.0f;
    {
        std::lock_guard<std::mutex> lock(g_fps_mutex);
        fps = g_current_fps;
    }

    const std::string proc_name = get_process_name();
    const uint32_t width = g_shm_header ? g_shm_header->swapchain_width : 0;
    const uint32_t height = g_shm_header ? g_shm_header->swapchain_height : 0;
    const uint32_t format = g_shm_header ? g_shm_header->swapchain_format : 0;
    const uint32_t mode = g_shm_header ? g_shm_header->swapchain_present_mode : 0;
    const uint32_t images = g_shm_header ? g_shm_header->swapchain_image_count : 0;
    const uint64_t captures = g_shm_header ? g_shm_header->roi_capture_count : 0;

    fprintf(f, "{\n");
    fprintf(f, "  \"layer_name\": \"%s\",\n", VD_LAYER_NAME);
    fprintf(f, "  \"version\": %d,\n", VD_LAYER_IMPL_VERSION);
    fprintf(f, "  \"active\": %s,\n", active ? "true" : "false");
    fprintf(f, "  \"capture_enabled\": %s,\n", g_capture_enabled ? "true" : "false");
    fprintf(f, "  \"pid\": %d,\n", static_cast<int>(pid));
    fprintf(f, "  \"process_name\": \"%s\",\n", proc_name.c_str());
    fprintf(f, "  \"frames_presented\": %llu,\n", static_cast<unsigned long long>(g_frame_counter.load()));
    fprintf(f, "  \"captures\": %llu,\n", static_cast<unsigned long long>(captures));
    fprintf(f, "  \"fps\": %.2f,\n", fps);
    fprintf(f, "  \"swapchain\": {\"width\": %u, \"height\": %u, \"format\": %u, \"present_mode\": %u, \"image_count\": %u},\n",
            width, height, format, mode, images);
    if (g_shm_header) {
        fprintf(f, "  \"roi\": {\"x\": %u, \"y\": %u, \"width\": %u, \"height\": %u, \"stride\": %u, \"size\": %u}\n",
                g_shm_header->roi_x, g_shm_header->roi_y, g_shm_header->roi_width,
                g_shm_header->roi_height, g_shm_header->roi_stride, g_shm_header->roi_data_size);
    } else {
        fprintf(f, "  \"roi\": null\n");
    }
    fprintf(f, "}\n");
    fclose(f);
    rename(tmp_path.c_str(), g_status_path.c_str());
}

VkLayerInstanceCreateInfo *get_instance_chain_info(const VkInstanceCreateInfo *create_info, VkLayerFunction func) {
    auto *chain = reinterpret_cast<VkLayerInstanceCreateInfo *>(const_cast<void *>(create_info->pNext));
    while (chain) {
        if (chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && chain->function == func) return chain;
        chain = reinterpret_cast<VkLayerInstanceCreateInfo *>(const_cast<void *>(chain->pNext));
    }
    return nullptr;
}

VkLayerDeviceCreateInfo *get_device_chain_info(const VkDeviceCreateInfo *create_info, VkLayerFunction func) {
    auto *chain = reinterpret_cast<VkLayerDeviceCreateInfo *>(const_cast<void *>(create_info->pNext));
    while (chain) {
        if (chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && chain->function == func) return chain;
        chain = reinterpret_cast<VkLayerDeviceCreateInfo *>(const_cast<void *>(chain->pNext));
    }
    return nullptr;
}

bool format_supported_for_capture(VkFormat format) {
    return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB ||
           format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB;
}

uint32_t find_host_memory_type(DeviceData *ddata, uint32_t type_bits, bool *coherent) {
    if (!ddata || !ddata->instance_data || !ddata->instance_data->vkGetPhysicalDeviceMemoryProperties) return UINT32_MAX;
    VkPhysicalDeviceMemoryProperties props{};
    ddata->instance_data->vkGetPhysicalDeviceMemoryProperties(ddata->physical_device, &props);

    uint32_t visible_fallback = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        const VkMemoryPropertyFlags flags = props.memoryTypes[i].propertyFlags;
        if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
            if (coherent) *coherent = true;
            return i;
        }
        if (visible_fallback == UINT32_MAX) visible_fallback = i;
    }
    if (visible_fallback != UINT32_MAX && coherent) *coherent = false;
    return visible_fallback;
}

void destroy_capture_resources(QueueData *qdata) {
    if (!qdata || !qdata->device_data) return;
    DeviceData *d = qdata->device_data;
    CaptureResources &c = qdata->capture;
    if (c.mapped && d->vkUnmapMemory && c.staging_memory) d->vkUnmapMemory(d->device, c.staging_memory);
    c.mapped = nullptr;
    for (VkSemaphore sem : c.present_semaphores) {
        if (sem && d->vkDestroySemaphore) d->vkDestroySemaphore(d->device, sem, nullptr);
    }
    c.present_semaphores.clear();
    if (c.fence && d->vkDestroyFence) d->vkDestroyFence(d->device, c.fence, nullptr);
    if (c.staging_buffer && d->vkDestroyBuffer) d->vkDestroyBuffer(d->device, c.staging_buffer, nullptr);
    if (c.staging_memory && d->vkFreeMemory) d->vkFreeMemory(d->device, c.staging_memory, nullptr);
    if (c.command_pool && d->vkDestroyCommandPool) d->vkDestroyCommandPool(d->device, c.command_pool, nullptr);
    c = CaptureResources{};
}

bool init_capture_resources(QueueData *qdata) {
    if (!qdata || !qdata->device_data) return false;
    DeviceData *d = qdata->device_data;
    CaptureResources &c = qdata->capture;
    if (c.initialized) return true;
    if (c.permanently_disabled) return false;

    const bool funcs_ok = d->vkCreateCommandPool && d->vkDestroyCommandPool && d->vkResetCommandPool &&
        d->vkAllocateCommandBuffers && d->vkCreateBuffer && d->vkDestroyBuffer &&
        d->vkGetBufferMemoryRequirements && d->vkAllocateMemory && d->vkFreeMemory &&
        d->vkBindBufferMemory && d->vkMapMemory && d->vkCreateFence && d->vkDestroyFence &&
        d->vkResetFences && d->vkWaitForFences && d->vkCreateSemaphore && d->vkDestroySemaphore &&
        d->vkQueueSubmit && d->vkBeginCommandBuffer && d->vkEndCommandBuffer &&
        d->vkCmdPipelineBarrier && d->vkCmdCopyImageToBuffer;
    if (!funcs_ok) {
        log_msg("Capture disabled: required Vulkan device functions are missing");
        c.permanently_disabled = true;
        return false;
    }

    auto fail = [&]() {
        log_msg("Capture resource initialization failed; capture disabled for queue family %u", qdata->family_index);
        destroy_capture_resources(qdata);
        qdata->capture.permanently_disabled = true;
        increment_dropped_frames();
        return false;
    };

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = qdata->family_index;
    if (d->vkCreateCommandPool(d->device, &pool_info, nullptr, &c.command_pool) != VK_SUCCESS) return fail();

    VkCommandBufferAllocateInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = c.command_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    if (d->vkAllocateCommandBuffers(d->device, &cmd_info, &c.command_buffer) != VK_SUCCESS) return fail();

    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = VD_CAPTURE_MAX_BYTES;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d->vkCreateBuffer(d->device, &buffer_info, nullptr, &c.staging_buffer) != VK_SUCCESS) return fail();

    VkMemoryRequirements requirements{};
    d->vkGetBufferMemoryRequirements(d->device, c.staging_buffer, &requirements);
    c.memory_coherent = false;
    const uint32_t memory_type = find_host_memory_type(d, requirements.memoryTypeBits, &c.memory_coherent);
    if (memory_type == UINT32_MAX) return fail();

    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = requirements.size;
    alloc_info.memoryTypeIndex = memory_type;
    if (d->vkAllocateMemory(d->device, &alloc_info, nullptr, &c.staging_memory) != VK_SUCCESS) return fail();
    if (d->vkBindBufferMemory(d->device, c.staging_buffer, c.staging_memory, 0) != VK_SUCCESS) return fail();
    if (d->vkMapMemory(d->device, c.staging_memory, 0, VD_CAPTURE_MAX_BYTES, 0, &c.mapped) != VK_SUCCESS) return fail();

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (d->vkCreateFence(d->device, &fence_info, nullptr, &c.fence) != VK_SUCCESS) return fail();

    VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    c.present_semaphores.reserve(kMaxCaptureSemaphores);
    for (uint32_t i = 0; i < kMaxCaptureSemaphores; ++i) {
        VkSemaphore sem = VK_NULL_HANDLE;
        if (d->vkCreateSemaphore(d->device, &sem_info, nullptr, &sem) != VK_SUCCESS) return fail();
        c.present_semaphores.push_back(sem);
    }

    c.initialized = true;
    log_msg("Capture resources ready on queue family %u (host coherent=%s)", qdata->family_index, c.memory_coherent ? "YES" : "NO");
    return true;
}

void register_queue(DeviceData *ddata, VkQueue queue, uint32_t family_index) {
    if (!ddata || !queue) return;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto &slot = g_queues[reinterpret_cast<void *>(queue)];
        if (!slot) slot = std::make_unique<QueueData>();
        slot->device_data = ddata;
        slot->family_index = family_index;
    }
    log_debug("Registered queue %p to device %p (family %u)", reinterpret_cast<void *>(queue), reinterpret_cast<void *>(ddata->device), family_index);
}

void publish_roi(const SwapchainInfo &sc, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const void *pixels, uint64_t now_ns) {
    if (!g_shm_header || !g_shm_pixels || !pixels) return;
    const uint32_t stride = width * VD_CAPTURE_BYTES_PER_PIXEL;
    const uint32_t size = stride * height;
    if (size > VD_CAPTURE_MAX_BYTES) return;

    shm_begin_write();
    memcpy(g_shm_pixels, pixels, size);
    g_shm_header->status_flags |= VD_STATUS_CAPTURE_ENABLED | VD_STATUS_PIXELS_READY;
    g_shm_header->roi_x = x;
    g_shm_header->roi_y = y;
    g_shm_header->roi_width = width;
    g_shm_header->roi_height = height;
    g_shm_header->roi_stride = stride;
    g_shm_header->roi_data_offset = sizeof(VdLayerShmHeader);
    g_shm_header->roi_data_size = size;
    g_shm_header->roi_capture_count = g_capture_counter.fetch_add(1) + 1;
    g_shm_header->roi_capture_timestamp_ns = now_ns;
    g_shm_header->swapchain_format = static_cast<uint32_t>(sc.format);
    shm_end_write();
    msync(g_shm_mapping, sizeof(VdLayerShmHeader) + size, MS_ASYNC);
}

bool capture_before_present(VkQueue queue, QueueData *qdata, const VkPresentInfoKHR *present_info, VkResult *present_result, uint64_t now_ns) {
    if (!g_capture_enabled || !qdata || !qdata->device_data || !present_info || !present_result) return false;
    if (present_info->swapchainCount != 1 || !present_info->pSwapchains || !present_info->pImageIndices) return false;

    const VkSwapchainKHR swapchain = present_info->pSwapchains[0];
    const uint32_t image_index = present_info->pImageIndices[0];
    SwapchainInfo sc;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_swapchains.find(swapchain);
        if (it == g_swapchains.end() || !it->second.capture_supported) return false;
        if (g_capture_interval_ns > 0 && now_ns - it->second.last_capture_ns < g_capture_interval_ns) return false;
        if (image_index >= it->second.images.size() || image_index >= kMaxCaptureSemaphores) return false;
        it->second.last_capture_ns = now_ns;
        sc = it->second;
    }

    DeviceData *d = qdata->device_data;
    std::lock_guard<std::mutex> capture_lock(qdata->capture_mutex);
    if (!init_capture_resources(qdata)) return false;
    CaptureResources &c = qdata->capture;

    const uint32_t roi_width = std::min<uint32_t>(VD_CAPTURE_MAX_WIDTH, sc.width);
    const uint32_t roi_height = std::min<uint32_t>(VD_CAPTURE_MAX_HEIGHT, sc.height);
    const uint32_t roi_x = (sc.width - roi_width) / 2;
    const uint32_t roi_y = (sc.height - roi_height) / 2;
    const uint32_t roi_size = roi_width * roi_height * VD_CAPTURE_BYTES_PER_PIXEL;

    if (d->vkWaitForFences(d->device, 1, &c.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
        d->vkResetCommandPool(d->device, c.command_pool, 0) != VK_SUCCESS) {
        increment_dropped_frames();
        return false;
    }

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (d->vkBeginCommandBuffer(c.command_buffer, &begin_info) != VK_SUCCESS) {
        increment_dropped_frames();
        return false;
    }

    VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_transfer.srcAccessMask = 0;
    to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_transfer.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.image = sc.images[image_index];
    to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    to_transfer.subresourceRange.baseMipLevel = 0;
    to_transfer.subresourceRange.levelCount = 1;
    to_transfer.subresourceRange.baseArrayLayer = 0;
    to_transfer.subresourceRange.layerCount = 1;
    d->vkCmdPipelineBarrier(c.command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &to_transfer);

    VkBufferImageCopy copy{};
    copy.bufferOffset = 0;
    copy.bufferRowLength = 0;
    copy.bufferImageHeight = 0;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = {static_cast<int32_t>(roi_x), static_cast<int32_t>(roi_y), 0};
    copy.imageExtent = {roi_width, roi_height, 1};
    d->vkCmdCopyImageToBuffer(c.command_buffer, sc.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              c.staging_buffer, 1, &copy);

    VkImageMemoryBarrier to_present = to_transfer;
    to_present.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    d->vkCmdPipelineBarrier(c.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &to_present);

    if (d->vkEndCommandBuffer(c.command_buffer) != VK_SUCCESS) {
        increment_dropped_frames();
        return false;
    }

    std::vector<VkPipelineStageFlags> wait_stages(present_info->waitSemaphoreCount, VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkSemaphore capture_done = c.present_semaphores[image_index];
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = present_info->waitSemaphoreCount;
    submit.pWaitSemaphores = present_info->pWaitSemaphores;
    submit.pWaitDstStageMask = wait_stages.empty() ? nullptr : wait_stages.data();
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &c.command_buffer;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &capture_done;

    if (d->vkResetFences(d->device, 1, &c.fence) != VK_SUCCESS) {
        increment_dropped_frames();
        return false;
    }
    const VkResult submit_result = d->vkQueueSubmit(queue, 1, &submit, c.fence);
    if (submit_result != VK_SUCCESS) {
        log_msg("Capture vkQueueSubmit failed: %d; disabling capture resources for this queue", static_cast<int>(submit_result));
        increment_dropped_frames();
        destroy_capture_resources(qdata);
        qdata->capture.permanently_disabled = true;
        return false;
    }

    const VkResult wait_result = d->vkWaitForFences(d->device, 1, &c.fence, VK_TRUE, UINT64_MAX);
    if (wait_result == VK_SUCCESS) {
        if (!c.memory_coherent && d->vkInvalidateMappedMemoryRanges) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = c.staging_memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            d->vkInvalidateMappedMemoryRanges(d->device, 1, &range);
        }
        publish_roi(sc, roi_x, roi_y, roi_width, roi_height, c.mapped, now_ns);
        log_debug("Captured ROI %ux%u at (%u,%u), image=%u, bytes=%u", roi_width, roi_height, roi_x, roi_y, image_index, roi_size);
    } else {
        log_msg("Capture fence wait failed: %d", static_cast<int>(wait_result));
        increment_dropped_frames();
    }

    VkPresentInfoKHR chained_present = *present_info;
    chained_present.waitSemaphoreCount = 1;
    chained_present.pWaitSemaphores = &capture_done;
    *present_result = d->vkQueuePresentKHR(queue, &chained_present);
    return true;
}

} // namespace

VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                                                     const VkAllocationCallbacks *pAllocator,
                                                     VkInstance *pInstance) {
    init_layer_runtime();
    log_msg("vd_vkCreateInstance called");
    VkLayerInstanceCreateInfo *chain = get_instance_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain || !chain->u.pLayerInfo) {
        log_msg("ERROR: No instance link info found in pCreateInfo");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance next_create = reinterpret_cast<PFN_vkCreateInstance>(next_gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    VkResult result = next_create(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) return result;

    auto data = std::make_unique<InstanceData>();
    data->instance = *pInstance;
    data->next_gipa = next_gipa;
    data->vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(next_gipa(*pInstance, "vkDestroyInstance"));
    data->vkCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(next_gipa(*pInstance, "vkCreateDevice"));
    data->vkEnumeratePhysicalDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(next_gipa(*pInstance, "vkEnumeratePhysicalDevices"));
    data->vkEnumeratePhysicalDeviceGroups = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceGroups>(next_gipa(*pInstance, "vkEnumeratePhysicalDeviceGroups"));
    data->vkEnumeratePhysicalDeviceGroupsKHR = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceGroupsKHR>(next_gipa(*pInstance, "vkEnumeratePhysicalDeviceGroupsKHR"));
    data->vkEnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(next_gipa(*pInstance, "vkEnumerateDeviceExtensionProperties"));
    data->vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(next_gipa(*pInstance, "vkGetPhysicalDeviceMemoryProperties"));
    data->vkGetPhysicalDeviceSurfaceCapabilitiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(next_gipa(*pInstance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        InstanceData *ptr = data.get();
        g_instances[reinterpret_cast<void *>(*pInstance)] = std::move(data);
        g_primary_instance = ptr;
    }
    log_msg("Instance %p created successfully", reinterpret_cast<void *>(*pInstance));
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroyInstance called for %p", reinterpret_cast<void *>(instance));
    PFN_vkDestroyInstance next_destroy = nullptr;
    bool no_instances = false;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find(reinterpret_cast<void *>(instance));
        if (it != g_instances.end()) {
            next_destroy = it->second->vkDestroyInstance;
            InstanceData *ptr = it->second.get();
            for (auto pit = g_physical_device_to_instance.begin(); pit != g_physical_device_to_instance.end();) {
                pit = pit->second == ptr ? g_physical_device_to_instance.erase(pit) : std::next(pit);
            }
            g_instances.erase(it);
            if (g_primary_instance == ptr) g_primary_instance = g_instances.empty() ? nullptr : g_instances.begin()->second.get();
        }
        no_instances = g_instances.empty();
    }
    if (no_instances && g_shm_header) {
        shm_begin_write();
        g_shm_header->status_flags = 0;
        shm_end_write();
        write_status_json(false);
    }
    if (next_destroy) next_destroy(instance, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumeratePhysicalDevices(VkInstance instance, uint32_t *count, VkPhysicalDevice *devices) {
    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find(reinterpret_cast<void *>(instance));
        idata = it != g_instances.end() ? it->second.get() : g_primary_instance;
    }
    if (!idata || !idata->vkEnumeratePhysicalDevices) return VK_ERROR_INITIALIZATION_FAILED;
    const VkResult result = idata->vkEnumeratePhysicalDevices(instance, count, devices);
    if ((result == VK_SUCCESS || result == VK_INCOMPLETE) && devices && count) {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        for (uint32_t i = 0; i < *count; ++i) if (devices[i]) g_physical_device_to_instance[reinterpret_cast<void *>(devices[i])] = idata;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumeratePhysicalDeviceGroups(VkInstance instance, uint32_t *count, VkPhysicalDeviceGroupProperties *groups) {
    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find(reinterpret_cast<void *>(instance));
        idata = it != g_instances.end() ? it->second.get() : g_primary_instance;
    }
    PFN_vkEnumeratePhysicalDeviceGroups fn = idata ? idata->vkEnumeratePhysicalDeviceGroups : nullptr;
    if (!fn && idata) fn = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceGroups>(idata->vkEnumeratePhysicalDeviceGroupsKHR);
    if (!fn) return VK_ERROR_INITIALIZATION_FAILED;
    const VkResult result = fn(instance, count, groups);
    if ((result == VK_SUCCESS || result == VK_INCOMPLETE) && groups && count) {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        for (uint32_t g = 0; g < *count; ++g) {
            for (uint32_t i = 0; i < groups[g].physicalDeviceCount; ++i) {
                if (groups[g].physicalDevices[i]) g_physical_device_to_instance[reinterpret_cast<void *>(groups[g].physicalDevices[i])] = idata;
            }
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateDevice(VkPhysicalDevice physicalDevice,
                                                   const VkDeviceCreateInfo *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator,
                                                   VkDevice *pDevice) {
    log_msg("vd_vkCreateDevice called for physicalDevice %p", reinterpret_cast<void *>(physicalDevice));
    VkLayerDeviceCreateInfo *chain = get_device_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain || !chain->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;

    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_physical_device_to_instance.find(reinterpret_cast<void *>(physicalDevice));
        idata = it != g_physical_device_to_instance.end() ? it->second : g_primary_instance;
    }
    const VkInstance instance = idata ? idata->instance : VK_NULL_HANDLE;
    PFN_vkCreateDevice next_create = next_gipa ? reinterpret_cast<PFN_vkCreateDevice>(next_gipa(instance, "vkCreateDevice")) : nullptr;
    if (!next_create && idata) next_create = idata->vkCreateDevice;
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    VkResult result = next_create(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS) return result;

    auto data = std::make_unique<DeviceData>();
    data->physical_device = physicalDevice;
    data->device = *pDevice;
    data->instance_data = idata;
    data->next_gdpa = next_gdpa;
#define VD_RESOLVE_DEVICE(name) data->name = reinterpret_cast<PFN_##name>(next_gdpa(*pDevice, #name))
    VD_RESOLVE_DEVICE(vkDestroyDevice);
    VD_RESOLVE_DEVICE(vkGetDeviceQueue);
    VD_RESOLVE_DEVICE(vkGetDeviceQueue2);
    VD_RESOLVE_DEVICE(vkCreateSwapchainKHR);
    VD_RESOLVE_DEVICE(vkDestroySwapchainKHR);
    VD_RESOLVE_DEVICE(vkGetSwapchainImagesKHR);
    VD_RESOLVE_DEVICE(vkQueuePresentKHR);
    VD_RESOLVE_DEVICE(vkCreateCommandPool);
    VD_RESOLVE_DEVICE(vkDestroyCommandPool);
    VD_RESOLVE_DEVICE(vkResetCommandPool);
    VD_RESOLVE_DEVICE(vkAllocateCommandBuffers);
    VD_RESOLVE_DEVICE(vkCreateBuffer);
    VD_RESOLVE_DEVICE(vkDestroyBuffer);
    VD_RESOLVE_DEVICE(vkGetBufferMemoryRequirements);
    VD_RESOLVE_DEVICE(vkAllocateMemory);
    VD_RESOLVE_DEVICE(vkFreeMemory);
    VD_RESOLVE_DEVICE(vkBindBufferMemory);
    VD_RESOLVE_DEVICE(vkMapMemory);
    VD_RESOLVE_DEVICE(vkUnmapMemory);
    VD_RESOLVE_DEVICE(vkInvalidateMappedMemoryRanges);
    VD_RESOLVE_DEVICE(vkCreateFence);
    VD_RESOLVE_DEVICE(vkDestroyFence);
    VD_RESOLVE_DEVICE(vkResetFences);
    VD_RESOLVE_DEVICE(vkWaitForFences);
    VD_RESOLVE_DEVICE(vkCreateSemaphore);
    VD_RESOLVE_DEVICE(vkDestroySemaphore);
    VD_RESOLVE_DEVICE(vkQueueSubmit);
    VD_RESOLVE_DEVICE(vkBeginCommandBuffer);
    VD_RESOLVE_DEVICE(vkEndCommandBuffer);
    VD_RESOLVE_DEVICE(vkCmdPipelineBarrier);
    VD_RESOLVE_DEVICE(vkCmdCopyImageToBuffer);
#undef VD_RESOLVE_DEVICE

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        DeviceData *ptr = data.get();
        g_devices[reinterpret_cast<void *>(*pDevice)] = std::move(data);
        g_primary_device = ptr;
    }
    log_msg("Device %p created successfully (Swapchain support: %s)", reinterpret_cast<void *>(*pDevice), g_primary_device->vkCreateSwapchainKHR ? "YES" : "NO");
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroyDevice called for %p", reinterpret_cast<void *>(device));
    PFN_vkDestroyDevice next_destroy = nullptr;
    DeviceData *target = nullptr;
    std::vector<std::unique_ptr<QueueData>> queues_to_destroy;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        if (it != g_devices.end()) {
            target = it->second.get();
            next_destroy = target->vkDestroyDevice;
            for (auto qit = g_queues.begin(); qit != g_queues.end();) {
                if (qit->second->device_data == target) {
                    queues_to_destroy.push_back(std::move(qit->second));
                    qit = g_queues.erase(qit);
                } else {
                    ++qit;
                }
            }
        }
    }
    for (auto &q : queues_to_destroy) {
        std::lock_guard<std::mutex> lock(q->capture_mutex);
        destroy_capture_resources(q.get());
    }
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        if (it != g_devices.end()) {
            DeviceData *ptr = it->second.get();
            g_devices.erase(it);
            if (g_primary_device == ptr) g_primary_device = g_devices.empty() ? nullptr : g_devices.begin()->second.get();
        }
    }
    if (next_destroy) next_destroy(device, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vd_vkGetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue *queue) {
    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        d = it != g_devices.end() ? it->second.get() : g_primary_device;
    }
    if (d && d->vkGetDeviceQueue) {
        d->vkGetDeviceQueue(device, family, index, queue);
        if (queue && *queue) register_queue(d, *queue, family);
    }
}

VKAPI_ATTR void VKAPI_CALL vd_vkGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *info, VkQueue *queue) {
    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        d = it != g_devices.end() ? it->second.get() : g_primary_device;
    }
    if (d && d->vkGetDeviceQueue2) {
        d->vkGetDeviceQueue2(device, info, queue);
        if (queue && *queue && info) register_queue(d, *queue, info->queueFamilyIndex);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateSwapchainKHR(VkDevice device,
                                                        const VkSwapchainCreateInfoKHR *pCreateInfo,
                                                        const VkAllocationCallbacks *pAllocator,
                                                        VkSwapchainKHR *pSwapchain) {
    log_msg("vd_vkCreateSwapchainKHR called: %ux%u format=%u mode=%u minImages=%u usage=0x%x",
            pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height, pCreateInfo->imageFormat,
            pCreateInfo->presentMode, pCreateInfo->minImageCount, pCreateInfo->imageUsage);

    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        d = it != g_devices.end() ? it->second.get() : g_primary_device;
    }
    if (!d || !d->vkCreateSwapchainKHR) return VK_ERROR_INITIALIZATION_FAILED;

    VkSwapchainCreateInfoKHR local = *pCreateInfo;
    const VkSwapchainCreateInfoKHR *create_info = pCreateInfo;
    bool capture_supported = false;
    bool added_transfer_usage = false;
    if (g_capture_enabled && format_supported_for_capture(pCreateInfo->imageFormat) && d->instance_data &&
        d->instance_data->vkGetPhysicalDeviceSurfaceCapabilitiesKHR) {
        VkSurfaceCapabilitiesKHR caps{};
        if (d->instance_data->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->physical_device, pCreateInfo->surface, &caps) == VK_SUCCESS &&
            (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            local.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            create_info = &local;
            capture_supported = true;
            added_transfer_usage = !(pCreateInfo->imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        } else {
            log_msg("Capture disabled for swapchain: surface does not support TRANSFER_SRC");
        }
    } else if (g_capture_enabled && !format_supported_for_capture(pCreateInfo->imageFormat)) {
        log_msg("Capture disabled for swapchain: unsupported 4-byte format %u", pCreateInfo->imageFormat);
    }

    VkResult result = d->vkCreateSwapchainKHR(device, create_info, pAllocator, pSwapchain);
    if (result != VK_SUCCESS && added_transfer_usage) {
        log_msg("Swapchain creation with TRANSFER_SRC failed (%d); retrying untouched create info", static_cast<int>(result));
        capture_supported = false;
        result = d->vkCreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
    }
    if (result != VK_SUCCESS) return result;

    SwapchainInfo sc;
    sc.swapchain = *pSwapchain;
    sc.device = device;
    sc.surface = pCreateInfo->surface;
    sc.width = pCreateInfo->imageExtent.width;
    sc.height = pCreateInfo->imageExtent.height;
    sc.format = pCreateInfo->imageFormat;
    sc.present_mode = pCreateInfo->presentMode;
    sc.capture_supported = capture_supported;

    if (d->vkGetSwapchainImagesKHR) {
        uint32_t count = 0;
        if (d->vkGetSwapchainImagesKHR(device, *pSwapchain, &count, nullptr) == VK_SUCCESS && count > 0) {
            sc.images.resize(count);
            if (d->vkGetSwapchainImagesKHR(device, *pSwapchain, &count, sc.images.data()) == VK_SUCCESS) {
                sc.images.resize(count);
                sc.image_count = count;
            } else {
                sc.images.clear();
            }
        }
    }
    if (sc.images.empty()) sc.capture_supported = false;

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_swapchains[*pSwapchain] = sc;
    }
    if (g_shm_header) {
        shm_begin_write();
        g_shm_header->status_flags |= VD_STATUS_SWAPCHAIN_READY;
        if (sc.capture_supported && g_capture_enabled) g_shm_header->status_flags |= VD_STATUS_CAPTURE_ENABLED;
        g_shm_header->swapchain_width = sc.width;
        g_shm_header->swapchain_height = sc.height;
        g_shm_header->swapchain_format = static_cast<uint32_t>(sc.format);
        g_shm_header->swapchain_present_mode = static_cast<uint32_t>(sc.present_mode);
        g_shm_header->swapchain_image_count = sc.image_count;
        shm_end_write();
    }
    log_msg("Swapchain %p created: %ux%u format=%u images=%u capture=%s",
            reinterpret_cast<void *>(*pSwapchain), sc.width, sc.height, sc.format, sc.image_count,
            sc.capture_supported ? "YES" : "NO");
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroySwapchainKHR called for %p", reinterpret_cast<void *>(swapchain));
    PFN_vkDestroySwapchainKHR next_destroy = nullptr;
    bool any = false;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_swapchains.erase(swapchain);
        any = !g_swapchains.empty();
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        DeviceData *d = it != g_devices.end() ? it->second.get() : g_primary_device;
        if (d) next_destroy = d->vkDestroySwapchainKHR;
    }
    if (g_shm_header && !any) {
        shm_begin_write();
        g_shm_header->status_flags &= ~(VD_STATUS_SWAPCHAIN_READY | VD_STATUS_PIXELS_READY);
        shm_end_write();
    }
    write_status_json(true);
    if (next_destroy) next_destroy(device, swapchain, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo) {
    QueueData *qdata = nullptr;
    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto qit = g_queues.find(reinterpret_cast<void *>(queue));
        if (qit != g_queues.end()) {
            qdata = qit->second.get();
            d = qdata->device_data;
        } else {
            d = g_primary_device;
        }
    }
    if (!d || !d->vkQueuePresentKHR) return VK_ERROR_INITIALIZATION_FAILED;

    const uint64_t frame = g_frame_counter.fetch_add(1) + 1;
    const uint64_t now_ns = get_time_ns();
    if (g_shm_header) {
        shm_begin_write();
        g_shm_header->frame_count = frame;
        g_shm_header->last_present_ns = now_ns;
        if (pPresentInfo && pPresentInfo->swapchainCount > 0 && pPresentInfo->pImageIndices)
            g_shm_header->current_image_index = pPresentInfo->pImageIndices[0];
        shm_end_write();
    }

    bool dump = frame == 1;
    {
        std::lock_guard<std::mutex> lock(g_fps_mutex);
        if (frame % 60 == 0 || now_ns - g_last_fps_time_ns >= 1000000000ULL) {
            const uint64_t elapsed = now_ns - g_last_fps_time_ns;
            if (elapsed) {
                g_current_fps = static_cast<float>(static_cast<double>(frame - g_last_fps_frame_count) * 1e9 / static_cast<double>(elapsed));
                if (g_shm_header) g_shm_header->current_fps = g_current_fps;
            }
            g_last_fps_time_ns = now_ns;
            g_last_fps_frame_count = frame;
            dump = true;
        }
    }

    VkResult chained_result = VK_SUCCESS;
    if (qdata && capture_before_present(queue, qdata, pPresentInfo, &chained_result, now_ns)) {
        if (dump) write_status_json(true);
        if (frame % 300 == 0) log_msg("Present heartbeat: frame %llu (%.1f fps, captures=%llu)",
                                      static_cast<unsigned long long>(frame), g_current_fps,
                                      static_cast<unsigned long long>(g_capture_counter.load()));
        return chained_result;
    }

    if (dump) write_status_json(true);
    if (frame % 300 == 0) log_msg("Present heartbeat: frame %llu (%.1f fps, captures=%llu)",
                                  static_cast<unsigned long long>(frame), g_current_fps,
                                  static_cast<unsigned long long>(g_capture_counter.load()));
    return d->vkQueuePresentKHR(queue, pPresentInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateInstanceLayerProperties(uint32_t *count, VkLayerProperties *props) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (!props) { *count = 1; return VK_SUCCESS; }
    if (*count < 1) return VK_INCOMPLETE;
    *count = 1;
    memset(&props[0], 0, sizeof(props[0]));
    strncpy(props[0].layerName, VD_LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
    props[0].specVersion = VD_LAYER_SPEC_VERSION;
    props[0].implementationVersion = VD_LAYER_IMPL_VERSION;
    strncpy(props[0].description, VD_LAYER_DESCRIPTION, VK_MAX_DESCRIPTION_SIZE - 1);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateInstanceExtensionProperties(const char *layer, uint32_t *count, VkExtensionProperties *) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (layer && strcmp(layer, VD_LAYER_NAME) == 0) { *count = 0; return VK_SUCCESS; }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateDeviceLayerProperties(VkPhysicalDevice, uint32_t *count, VkLayerProperties *props) {
    return vd_vkEnumerateInstanceLayerProperties(count, props);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *layer, uint32_t *count, VkExtensionProperties *props) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (layer && strcmp(layer, VD_LAYER_NAME) == 0) { *count = 0; return VK_SUCCESS; }
    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_physical_device_to_instance.find(reinterpret_cast<void *>(physicalDevice));
        idata = it != g_physical_device_to_instance.end() ? it->second : g_primary_instance;
    }
    return idata && idata->vkEnumerateDeviceExtensionProperties
        ? idata->vkEnumerateDeviceExtensionProperties(physicalDevice, layer, count, props)
        : VK_SUCCESS;
}

extern "C" {

VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    if (!name) return nullptr;
    log_debug("vkGetInstanceProcAddr: instance=%p name=%s", reinterpret_cast<void *>(instance), name);
    if (strcmp(name, "vkGetInstanceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vkGetInstanceProcAddr);
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceProcAddr);
    if (strcmp(name, "vkNegotiateLoaderLayerInterfaceVersion") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vkNegotiateLoaderLayerInterfaceVersion);
    if (strcmp(name, "vkCreateInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkCreateInstance);
    if (strcmp(name, "vkEnumerateInstanceLayerProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumerateInstanceLayerProperties);
    if (strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumerateInstanceExtensionProperties);
    if (!instance) return nullptr;

    if (strcmp(name, "vkDestroyInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroyInstance);
    if (strcmp(name, "vkEnumeratePhysicalDevices") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumeratePhysicalDevices);
    if (strcmp(name, "vkEnumeratePhysicalDeviceGroups") == 0 || strcmp(name, "vkEnumeratePhysicalDeviceGroupsKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumeratePhysicalDeviceGroups);
    if (strcmp(name, "vkCreateDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkCreateDevice);
    if (strcmp(name, "vkEnumerateDeviceLayerProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumerateDeviceLayerProperties);
    if (strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumerateDeviceExtensionProperties);
    if (strcmp(name, "vkDestroyDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroyDevice);
    if (strcmp(name, "vkGetDeviceQueue") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkGetDeviceQueue);
    if (strcmp(name, "vkGetDeviceQueue2") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkGetDeviceQueue2);
    if (strcmp(name, "vkCreateSwapchainKHR") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkCreateSwapchainKHR);
    if (strcmp(name, "vkDestroySwapchainKHR") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroySwapchainKHR);
    if (strcmp(name, "vkQueuePresentKHR") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkQueuePresentKHR);

    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find(reinterpret_cast<void *>(instance));
        idata = it != g_instances.end() ? it->second.get() : g_primary_instance;
    }
    return idata && idata->next_gipa ? idata->next_gipa(instance, name) : nullptr;
}

VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    if (!name) return nullptr;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceProcAddr);
    if (strcmp(name, "vkDestroyDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroyDevice);
    if (strcmp(name, "vkGetDeviceQueue") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkGetDeviceQueue);
    if (strcmp(name, "vkGetDeviceQueue2") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkGetDeviceQueue2);

    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        d = it != g_devices.end() ? it->second.get() : g_primary_device;
    }
    if (strcmp(name, "vkCreateSwapchainKHR") == 0) return d && d->vkCreateSwapchainKHR ? reinterpret_cast<PFN_vkVoidFunction>(vd_vkCreateSwapchainKHR) : nullptr;
    if (strcmp(name, "vkDestroySwapchainKHR") == 0) return d && d->vkDestroySwapchainKHR ? reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroySwapchainKHR) : nullptr;
    if (strcmp(name, "vkQueuePresentKHR") == 0) return d && d->vkQueuePresentKHR ? reinterpret_cast<PFN_vkVoidFunction>(vd_vkQueuePresentKHR) : nullptr;
    return d && d->next_gdpa ? d->next_gdpa(device, name) : nullptr;
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *version) {
    init_layer_runtime();
    log_msg("vkNegotiateLoaderLayerInterfaceVersion called, loader version: %u", version ? version->loaderLayerInterfaceVersion : 0);
    if (!version || version->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT || version->loaderLayerInterfaceVersion < 1)
        return VK_ERROR_INITIALIZATION_FAILED;
    version->loaderLayerInterfaceVersion = std::min<uint32_t>(version->loaderLayerInterfaceVersion, 2);
    version->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    version->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
    version->pfnGetPhysicalDeviceProcAddr = nullptr;
    log_msg("Negotiation successful (interface version %u, pfnGetPhysicalDeviceProcAddr = nullptr)", version->loaderLayerInterfaceVersion);
    return VK_SUCCESS;
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *count, VkLayerProperties *props) {
    return vd_vkEnumerateInstanceLayerProperties(count, props);
}
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *layer, uint32_t *count, VkExtensionProperties *props) {
    return vd_vkEnumerateInstanceExtensionProperties(layer, count, props);
}
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physical, uint32_t *count, VkLayerProperties *props) {
    return vd_vkEnumerateDeviceLayerProperties(physical, count, props);
}
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physical, const char *layer, uint32_t *count, VkExtensionProperties *props) {
    return vd_vkEnumerateDeviceExtensionProperties(physical, layer, count, props);
}

} // extern "C"
