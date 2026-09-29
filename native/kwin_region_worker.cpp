#include <KPipeWire/pipewiresourcestream.h>

#include <QCoreApplication>
#include <QByteArray>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QTimer>

#include <wayland-client.h>
#include "xdg-shell-client-protocol.h"
#include "zkde-screencast-client-protocol.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <poll.h>
#include <string>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kReferenceWidth = 1920;
constexpr int kReferenceHeight = 1080;
constexpr uint32_t kHiddenPointer = 1;

struct Options {
    int x = 800;
    int y = 420;
    int width = 320;
    int height = 240;
    int fps = 60;
    int priority = 10;
};

struct OutputInfo {
    wl_output *proxy = nullptr;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int scale = 1;
    int transform = WL_OUTPUT_TRANSFORM_NORMAL;
    bool hasMode = false;
    std::string name;
};

struct WaylandState {
    wl_display *display = nullptr;
    wl_registry *registry = nullptr;
    wl_compositor *compositor = nullptr;
    xdg_wm_base *wmBase = nullptr;
    xdg_surface *xdgSurface = nullptr;
    xdg_toplevel *toplevel = nullptr;
    wl_surface *surface = nullptr;
    zkde_screencast_unstable_v1 *screencast = nullptr;
    zkde_screencast_stream_unstable_v1 *stream = nullptr;
    std::vector<std::unique_ptr<OutputInfo>> outputs;
    uint32_t screencastVersion = 0;
    uint64_t objectSerial = 0;
    bool hasSerial = false;
    std::string error;
};

volatile sig_atomic_t g_interrupted = 0;

void signalHandler(int)
{
    g_interrupted = 1;
}

bool writeAll(const void *data, size_t size)
{
    const auto *bytes = static_cast<const char *>(data);
    while (size > 0) {
        const ssize_t written = ::write(STDOUT_FILENO, bytes, size);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return false;
        }
        bytes += written;
        size -= static_cast<size_t>(written);
    }
    return true;
}

int64_t clockNanos(clockid_t clock)
{
    timespec value{};
    if (::clock_gettime(clock, &value) != 0) {
        return 0;
    }
    return int64_t(value.tv_sec) * 1'000'000'000LL + value.tv_nsec;
}

bool parseInteger(const char *text, int *value)
{
    if (!text || !*text) {
        return false;
    }
    const char *end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, *value);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parseOptions(int argc, char **argv, Options *options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            std::puts("usage: kwin-region-worker --roi X,Y,W,H --fps N --priority N");
            std::exit(0);
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", argv[i]);
            return false;
        }
        const char *value = argv[++i];
        if (arg == "--roi") {
            char tail = '\0';
            if (std::sscanf(value, "%d,%d,%d,%d%c", &options->x, &options->y,
                            &options->width, &options->height, &tail) != 4) {
                std::fprintf(stderr, "Invalid ROI: %s\n", value);
                return false;
            }
        } else if (arg == "--fps") {
            if (!parseInteger(value, &options->fps)) {
                std::fprintf(stderr, "Invalid FPS: %s\n", value);
                return false;
            }
        } else if (arg == "--priority") {
            if (!parseInteger(value, &options->priority)) {
                std::fprintf(stderr, "Invalid priority: %s\n", value);
                return false;
            }
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    if (options->x < 0 || options->y < 0 || options->width <= 0 || options->height <= 0 ||
        options->fps < 1 || options->fps > 240 || options->priority < 0 ||
        options->priority > 19) {
        std::fprintf(stderr, "ROI, FPS, or priority is out of range\n");
        return false;
    }
    return true;
}

void wmPing(void *, xdg_wm_base *base, uint32_t serial)
{
    xdg_wm_base_pong(base, serial);
}

const xdg_wm_base_listener kWmListener = {wmPing};

void surfaceConfigured(void *, xdg_surface *surface, uint32_t serial)
{
    xdg_surface_ack_configure(surface, serial);
}

const xdg_surface_listener kSurfaceListener = {surfaceConfigured};

void outputGeometry(void *data, wl_output *, int32_t x, int32_t y, int32_t,
                    int32_t, int32_t, const char *, const char *, int32_t transform)
{
    auto *output = static_cast<OutputInfo *>(data);
    output->x = x;
    output->y = y;
    output->transform = transform;
}

void outputMode(void *data, wl_output *, uint32_t flags, int32_t width, int32_t height,
                int32_t)
{
    if (!(flags & WL_OUTPUT_MODE_CURRENT)) {
        return;
    }
    auto *output = static_cast<OutputInfo *>(data);
    output->width = width;
    output->height = height;
    output->hasMode = width > 0 && height > 0;
}

void outputDone(void *, wl_output *) {}

void outputScale(void *data, wl_output *, int32_t scale)
{
    auto *output = static_cast<OutputInfo *>(data);
    output->scale = std::max(1, scale);
}

void outputName(void *data, wl_output *, const char *name)
{
    static_cast<OutputInfo *>(data)->name = name ? name : "";
}

void outputDescription(void *, wl_output *, const char *) {}

const wl_output_listener kOutputListener = {
    outputGeometry, outputMode, outputDone, outputScale, outputName, outputDescription
};

void registryGlobal(void *data, wl_registry *registry, uint32_t name,
                    const char *interface, uint32_t version)
{
    auto *state = static_cast<WaylandState *>(data);
    if (std::strcmp(interface, "wl_compositor") == 0) {
        state->compositor = static_cast<wl_compositor *>(wl_registry_bind(
            registry, name, &wl_compositor_interface, std::min(version, 4u)));
    } else if (std::strcmp(interface, "xdg_wm_base") == 0) {
        state->wmBase = static_cast<xdg_wm_base *>(wl_registry_bind(
            registry, name, &xdg_wm_base_interface, std::min(version, 6u)));
        xdg_wm_base_add_listener(state->wmBase, &kWmListener, nullptr);
    } else if (std::strcmp(interface, "wl_output") == 0) {
        auto output = std::make_unique<OutputInfo>();
        output->proxy = static_cast<wl_output *>(wl_registry_bind(
            registry, name, &wl_output_interface, std::min(version, 4u)));
        wl_output_add_listener(output->proxy, &kOutputListener, output.get());
        state->outputs.push_back(std::move(output));
    } else if (std::strcmp(interface, "zkde_screencast_unstable_v1") == 0) {
        state->screencastVersion = version;
        state->screencast = static_cast<zkde_screencast_unstable_v1 *>(wl_registry_bind(
            registry, name, &zkde_screencast_unstable_v1_interface,
            std::min(version, 6u)));
    }
}

void registryGlobalRemove(void *, wl_registry *, uint32_t) {}

const wl_registry_listener kRegistryListener = {registryGlobal, registryGlobalRemove};

void streamClosed(void *data, zkde_screencast_stream_unstable_v1 *)
{
    auto *state = static_cast<WaylandState *>(data);
    state->error = "KWin closed the region stream";
}

void streamCreated(void *, zkde_screencast_stream_unstable_v1 *, uint32_t) {}

void streamFailed(void *data, zkde_screencast_stream_unstable_v1 *, const char *error)
{
    auto *state = static_cast<WaylandState *>(data);
    state->error = error ? error : "KWin failed to create the region stream";
}

void streamSerial(void *data, zkde_screencast_stream_unstable_v1 *, uint32_t high,
                  uint32_t low)
{
    auto *state = static_cast<WaylandState *>(data);
    state->objectSerial = (uint64_t(high) << 32) | low;
    state->hasSerial = true;
}

const zkde_screencast_stream_unstable_v1_listener kStreamListener = {
    streamClosed, streamCreated, streamFailed, streamSerial
};

OutputInfo *selectOutput(WaylandState *state)
{
    OutputInfo *fallback = nullptr;
    for (const auto &item : state->outputs) {
        if (!item->hasMode) {
            continue;
        }
        if (!fallback) {
            fallback = item.get();
        }
        if (item->x == 0 && item->y == 0) {
            return item.get();
        }
    }
    return fallback;
}

bool createRegionStream(WaylandState *state, const Options &options, OutputInfo *output,
                        int *sourceWidth, int *sourceHeight)
{
    if (!state->display || !state->compositor || !state->wmBase) {
        std::fprintf(stderr, "Wayland compositor or xdg-shell is unavailable\n");
        return false;
    }
    state->surface = wl_compositor_create_surface(state->compositor);
    state->xdgSurface = xdg_wm_base_get_xdg_surface(state->wmBase, state->surface);
    state->toplevel = xdg_surface_get_toplevel(state->xdgSurface);
    xdg_toplevel_set_app_id(state->toplevel, "vd-region-capture");
    xdg_surface_add_listener(state->xdgSurface, &kSurfaceListener, nullptr);
    wl_surface_commit(state->surface);
    if (wl_display_roundtrip(state->display) < 0) {
        std::fprintf(stderr, "Wayland identity roundtrip failed: %s\n", std::strerror(errno));
        return false;
    }
    if (!state->screencast || state->screencastVersion < 6) {
        std::fprintf(stderr, "KWin region capture needs zkde_screencast_unstable_v1 v6\n");
        return false;
    }
    if (!output || !output->hasMode || output->scale < 1) {
        std::fprintf(stderr, "Cannot determine the active display mode\n");
        return false;
    }

    const bool rotated = output->transform == WL_OUTPUT_TRANSFORM_90 ||
        output->transform == WL_OUTPUT_TRANSFORM_270 ||
        output->transform == WL_OUTPUT_TRANSFORM_FLIPPED_90 ||
        output->transform == WL_OUTPUT_TRANSFORM_FLIPPED_270;
    *sourceWidth = rotated ? output->height : output->width;
    *sourceHeight = rotated ? output->width : output->height;
    const double logicalWidth = double(*sourceWidth) / output->scale;
    const double logicalHeight = double(*sourceHeight) / output->scale;

    const int regionX = output->x + int(std::lround(options.x * logicalWidth / kReferenceWidth));
    const int regionY = output->y + int(std::lround(options.y * logicalHeight / kReferenceHeight));
    const int regionWidth = int(std::lround(options.width * logicalWidth / kReferenceWidth));
    const int regionHeight = int(std::lround(options.height * logicalHeight / kReferenceHeight));
    if (regionWidth < 1 || regionHeight < 1 || regionX < output->x || regionY < output->y ||
        regionX + regionWidth > output->x + int(std::lround(logicalWidth)) ||
        regionY + regionHeight > output->y + int(std::lround(logicalHeight))) {
        std::fprintf(stderr, "Requested ROI does not fit on display %s\n", output->name.c_str());
        return false;
    }

    const double scale = double(options.width) / regionWidth;
    const wl_fixed_t requestedScale = wl_fixed_from_double(scale);
    state->stream = zkde_screencast_unstable_v1_stream_region(
        state->screencast, regionX, regionY, uint32_t(regionWidth), uint32_t(regionHeight),
        requestedScale, kHiddenPointer);
    zkde_screencast_stream_unstable_v1_add_listener(state->stream, &kStreamListener, state);
    if (wl_display_flush(state->display) < 0) {
        std::fprintf(stderr, "Cannot send KWin region request: %s\n", std::strerror(errno));
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!state->hasSerial && state->error.empty()) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) {
            break;
        }
        pollfd fd{wl_display_get_fd(state->display), POLLIN, 0};
        const int ready = ::poll(&fd, 1, int(remaining));
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready <= 0) {
            break;
        }
        if (wl_display_dispatch(state->display) < 0) {
            std::fprintf(stderr, "KWin region event failed: %s\n", std::strerror(errno));
            return false;
        }
    }
    if (!state->error.empty()) {
        std::fprintf(stderr, "KWin region request failed: %s\n", state->error.c_str());
        return false;
    }
    if (!state->hasSerial) {
        std::fprintf(stderr, "KWin did not return a PipeWire stream serial\n");
        return false;
    }

    std::fprintf(stderr,
        "KWin region: output=%s %dx%d scale=%d roi=%d,%d %dx%d stream=%dx%d\n",
        output->name.c_str(), *sourceWidth, *sourceHeight, output->scale, regionX, regionY,
        regionWidth, regionHeight, options.width, options.height);
    return true;
}

bool writeFrame(const PipeWireFrame &frame, const Options &options, int sourceWidth,
                int sourceHeight, uint64_t sequence)
{
    if (!frame.dataFrame) {
        return true;
    }
    const int64_t receivedNs = clockNanos(CLOCK_MONOTONIC);
    QImage image = frame.dataFrame->toImage().convertToFormat(QImage::Format_BGR888);
    if (image.isNull()) {
        std::fprintf(stderr, "Cannot convert PipeWire frame to BGR\n");
        return false;
    }
    if (image.width() != options.width || image.height() != options.height) {
        image = image.scaled(options.width, options.height, Qt::IgnoreAspectRatio,
                             Qt::FastTransformation);
    }

    const int stride = options.width * 3;
    QByteArray payload;
    payload.resize(stride * options.height);
    for (int y = 0; y < options.height; ++y) {
        std::memcpy(payload.data() + y * stride, image.constScanLine(y), stride);
    }

    const int64_t cpuNs = clockNanos(CLOCK_PROCESS_CPUTIME_ID);
    const auto mediaNs = frame.presentationTimestamp
        ? frame.presentationTimestamp->count() : int64_t(0);
    QJsonObject header;
    header.insert("seq", double(sequence));
    header.insert("width", options.width);
    header.insert("height", options.height);
    header.insert("source_width", sourceWidth);
    header.insert("source_height", sourceHeight);
    header.insert("bytes", payload.size());
    header.insert("stride", stride);
    header.insert("worker_cpu_time_ns", double(cpuNs));
    header.insert("pts_ns", frame.presentationTimestamp
        ? QJsonValue(double(mediaNs)) : QJsonValue(QJsonValue::Null));
    header.insert("negotiated_caps", QString("PipeWire BGR %1x%2")
        .arg(options.width).arg(options.height));
    header.insert("requested_fps", options.fps);
    header.insert("media_monotonic_ns", frame.presentationTimestamp
        ? QJsonValue(double(mediaNs)) : QJsonValue(QJsonValue::Null));
    header.insert("received_ns", double(receivedNs));
    header.insert("timestamp_kind", frame.presentationTimestamp
        ? QString("pipewire_presentation_timestamp") : QString("arrival_only"));
    header.insert("synthetic", false);
    const QByteArray json = QJsonDocument(header).toJson(QJsonDocument::Compact);
    if (json.size() > 16384) {
        std::fprintf(stderr, "Capture frame header is unexpectedly large\n");
        return false;
    }
    const uint32_t headerSize = htonl(uint32_t(json.size()));
    return writeAll(&headerSize, sizeof(headerSize)) &&
        writeAll(json.constData(), size_t(json.size())) &&
        writeAll(payload.constData(), size_t(payload.size()));
}

void destroyWayland(WaylandState *state)
{
    if (state->stream) {
        zkde_screencast_stream_unstable_v1_close(state->stream);
    }
    if (state->screencast) {
        zkde_screencast_unstable_v1_destroy(state->screencast);
    }
    if (state->toplevel) {
        xdg_toplevel_destroy(state->toplevel);
    }
    if (state->xdgSurface) {
        xdg_surface_destroy(state->xdgSurface);
    }
    if (state->surface) {
        wl_surface_destroy(state->surface);
    }
    if (state->wmBase) {
        xdg_wm_base_destroy(state->wmBase);
    }
    if (state->compositor) {
        wl_compositor_destroy(state->compositor);
    }
    for (const auto &output : state->outputs) {
        if (output->proxy) {
            wl_output_destroy(output->proxy);
        }
    }
    if (state->registry) {
        wl_registry_destroy(state->registry);
    }
    if (state->display) {
        wl_display_flush(state->display);
        wl_display_disconnect(state->display);
    }
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!parseOptions(argc, argv, &options)) {
        return 2;
    }
    errno = 0;
    const int currentPriority = getpriority(PRIO_PROCESS, 0);
    if (errno == 0 && options.priority > currentPriority &&
        setpriority(PRIO_PROCESS, 0, options.priority) != 0) {
        std::fprintf(stderr, "Could not set capture priority: %s\n", std::strerror(errno));
    }

    QCoreApplication app(argc, argv);
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    WaylandState wayland;
    wayland.display = wl_display_connect(nullptr);
    if (!wayland.display) {
        std::fprintf(stderr, "Cannot connect to the Wayland display\n");
        return 1;
    }
    wayland.registry = wl_display_get_registry(wayland.display);
    wl_registry_add_listener(wayland.registry, &kRegistryListener, &wayland);
    if (wl_display_roundtrip(wayland.display) < 0) {
        std::fprintf(stderr, "Cannot read Wayland globals: %s\n", std::strerror(errno));
        destroyWayland(&wayland);
        return 1;
    }
    if (wl_display_roundtrip(wayland.display) < 0) {
        std::fprintf(stderr, "Cannot read Wayland output geometry: %s\n", std::strerror(errno));
        destroyWayland(&wayland);
        return 1;
    }
    OutputInfo *output = selectOutput(&wayland);
    int sourceWidth = 0;
    int sourceHeight = 0;
    if (!createRegionStream(&wayland, options, output, &sourceWidth, &sourceHeight)) {
        destroyWayland(&wayland);
        return 1;
    }

    PipeWireSourceStream stream;
    uint64_t sequence = 0;
    QObject::connect(&stream, &PipeWireSourceStream::streamReady,
                     [&stream]() { stream.setActive(true); });
    QObject::connect(&stream, &PipeWireSourceStream::stateChanged,
                     [&app, &stream](pw_stream_state state, pw_stream_state) {
        if (state == PW_STREAM_STATE_ERROR || state == PW_STREAM_STATE_UNCONNECTED) {
            std::fprintf(stderr, "PipeWire stream stopped: %s\n",
                         stream.error().toUtf8().constData());
            app.quit();
        }
    });
    QObject::connect(&stream, &PipeWireSourceStream::frameReceived,
                     [&app, &options, sourceWidth, sourceHeight, &sequence,
                      &wayland](const PipeWireFrame &frame) {
        if (!writeFrame(frame, options, sourceWidth, sourceHeight, sequence++)) {
            wayland.error = "Could not write a frame to the solver";
            app.quit();
        }
    });
    stream.setMaxFramerate(Fraction{uint32_t(options.fps), 1});
    if (!stream.createStream(quint64(wayland.objectSerial), -1)) {
        std::fprintf(stderr, "Cannot connect to KWin PipeWire stream: %s\n",
                     stream.error().toUtf8().constData());
        destroyWayland(&wayland);
        return 1;
    }

    QTimer stopPoll;
    QObject::connect(&stopPoll, &QTimer::timeout, [&app, &wayland]() {
        if (g_interrupted || !wayland.error.empty()) {
            app.quit();
        }
    });
    stopPoll.start(20);
    const int result = app.exec();
    stream.setActive(false);
    if (!wayland.error.empty()) {
        std::fprintf(stderr, "CAPTURE ERROR: %s\n", wayland.error.c_str());
    }
    destroyWayland(&wayland);
    return result == 0 && wayland.error.empty() ? 0 : 1;
}
