#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using Image = std::vector<std::vector<std::uint8_t>>;
using Kernel = std::vector<std::vector<float>>;
using FlatImage = std::vector<std::uint8_t>;

constexpr int kIterations = 100;
constexpr int kWarmupIterations = 5;
constexpr int kKernelSizes[] = { 3, 5, 7, 9 };
constexpr double kClockGHz = 4.4;

// Reordering float additions can change a truncated result by 1.
constexpr int kTolerance = 1;

__attribute__((noinline)) void apply_kernel_v0(
    const Image& image, const Kernel& kernel, Image& output) {
    size_t width = image[0].size();
    size_t height = image.size();
    size_t k = kernel.size();

    size_t radius = k / 2;

    for (size_t y = radius; y < height - radius; ++y) {
        for (size_t x = radius; x < width - radius; ++x) {
            float sum = 0;

            for (size_t ky = 0; ky < k; ++ky) {
                for (size_t kx = 0; kx < k; ++kx) {
                    size_t iy = y + ky - radius;
                    size_t ix = x + kx - radius;

                    sum += image[iy][ix] * kernel[ky][kx];
                }
            }

            output[y][x] = sum;
        }
    }
}

template <size_t K>
__attribute__((noinline)) void apply_kernel_v1_fixed(
    const Image& image, const Kernel& kernel, Image& output) {
    size_t width = image[0].size();
    size_t height = image.size();

    constexpr size_t radius = K / 2;

    for (size_t y = radius; y < height - radius; ++y) {
        for (size_t x = radius; x < width - radius; ++x) {
            float sum = 0;

            for (size_t ky = 0; ky < K; ++ky) {
                for (size_t kx = 0; kx < K; ++kx) {
                    size_t iy = y + ky - radius;
                    size_t ix = x + kx - radius;

                    sum += image[iy][ix] * kernel[ky][kx];
                }
            }

            output[y][x] = sum;
        }
    }
}

template <size_t K>
__attribute__((noinline)) void apply_kernel_v2_fixed(
    const FlatImage& flat_image, size_t width, size_t height, const Kernel& kernel, Image& output) {
    // K == 0 retains the runtime-size fallback for other kernels.
    const size_t k = K == 0 ? kernel.size() : K;
    const size_t radius = k / 2;

    for (size_t y = radius; y < height - radius; ++y) {
        for (size_t x = radius; x < width - radius; ++x) {
            float sum = 0;

            for (size_t ky = 0; ky < k; ++ky) {
                for (size_t kx = 0; kx < k; ++kx) {
                    size_t iy = y + ky - radius;
                    size_t ix = x + kx - radius;

                    sum += flat_image[iy * width + ix] * kernel[ky][kx];
                }
            }

            output[y][x] = sum;
        }
    }
}

template <size_t K>
__attribute__((noinline)) void apply_kernel_v3_fixed(
    const std::vector<float>& flat_image_float, size_t width, size_t height,
    const Kernel& kernel, Image& output) {
    // K == 0 retains the runtime-size fallback for other kernels.
    const size_t k = K == 0 ? kernel.size() : K;
    const size_t radius = k / 2;

    for (size_t y = radius; y < height - radius; ++y) {
        for (size_t x = radius; x < width - radius; ++x) {
            float sum = 0;

            for (size_t ky = 0; ky < k; ++ky) {
                for (size_t kx = 0; kx < k; ++kx) {
                    size_t iy = y + ky - radius;
                    size_t ix = x + kx - radius;

                    sum += flat_image_float[iy * width + ix] * kernel[ky][kx];
                }
            }

            output[y][x] = sum;
        }
    }
}

void apply_kernel_v1(const Image& image, const Kernel& kernel, Image& output) {
    switch (kernel.size()) {
    case 3:
        return apply_kernel_v1_fixed<3>(image, kernel, output);
    case 5:
        return apply_kernel_v1_fixed<5>(image, kernel, output);
    case 7:
        return apply_kernel_v1_fixed<7>(image, kernel, output);
    case 9:
        return apply_kernel_v1_fixed<9>(image, kernel, output);
    default:
        return apply_kernel_v0(image, kernel, output);
    }
}

void apply_kernel_v2(
    const FlatImage& image, size_t width, size_t height, const Kernel& kernel, Image& output) {
    switch (kernel.size()) {
    case 3:
        return apply_kernel_v2_fixed<3>(image, width, height, kernel, output);
    case 5:
        return apply_kernel_v2_fixed<5>(image, width, height, kernel, output);
    case 7:
        return apply_kernel_v2_fixed<7>(image, width, height, kernel, output);
    case 9:
        return apply_kernel_v2_fixed<9>(image, width, height, kernel, output);
    default:
        return apply_kernel_v2_fixed<0>(image, width, height, kernel, output);
    }
}

void apply_kernel_v3(const std::vector<float>& image, size_t width, size_t height,
    const Kernel& kernel, Image& output) {
    switch (kernel.size()) {
    case 3:
        return apply_kernel_v3_fixed<3>(image, width, height, kernel, output);
    case 5:
        return apply_kernel_v3_fixed<5>(image, width, height, kernel, output);
    case 7:
        return apply_kernel_v3_fixed<7>(image, width, height, kernel, output);
    case 9:
        return apply_kernel_v3_fixed<9>(image, width, height, kernel, output);
    default:
        return apply_kernel_v3_fixed<0>(image, width, height, kernel, output);
    }
}

FlatImage flatten(const Image& image) {
    const size_t width = image[0].size();
    FlatImage flat(image.size() * width);
    for (size_t y = 0; y < image.size(); ++y) {
        std::memcpy(flat.data() + y * width, image[y].data(), width);
    }
    return flat;
}

template <typename T> void do_not_optimize(T const& value) {
    asm volatile("" : : "g"(&value) : "memory");
}

std::optional<Image> load_grayscale(const char* path) {
    int width, height, channels;
    unsigned char* data = stbi_load(path, &width, &height, &channels, 1);
    if (data == nullptr) {
        std::fprintf(stderr, "failed to load %s: %s\n", path, stbi_failure_reason());
        return std::nullopt;
    }

    Image image(height, std::vector<std::uint8_t>(width));
    for (int y = 0; y < height; ++y) {
        std::memcpy(image[y].data(), data + static_cast<size_t>(y) * width, width);
    }

    stbi_image_free(data);
    return image;
}

Kernel box_kernel(int k) { return Kernel(k, std::vector<float>(k, 1.0f / (k * k))); }

Image blank_like(const Image& image) {
    return Image(image.size(), std::vector<std::uint8_t>(image[0].size()));
}

struct Mismatch {
    size_t count = 0;
    int max_diff = 0;
};

Mismatch compare(const Image& expected, const Image& actual) {
    Mismatch result;
    for (size_t y = 0; y < expected.size(); ++y) {
        for (size_t x = 0; x < expected[y].size(); ++x) {
            int diff = std::abs(int(expected[y][x]) - int(actual[y][x]));
            result.max_diff = std::max(result.max_diff, diff);
            if (diff > kTolerance) {
                ++result.count;
            }
        }
    }
    return result;
}

struct Timing {
    double min_us;
    double median_us;
};

template <typename Run> Timing time_variant(Run run) {
    for (int i = 0; i < kWarmupIterations; ++i) {
        run();
    }

    std::vector<double> samples(kIterations);
    for (double& sample : samples) {
        auto start = std::chrono::steady_clock::now();
        run();
        auto end = std::chrono::steady_clock::now();
        sample = std::chrono::duration<double, std::micro>(end - start).count();
    }

    std::ranges::sort(samples);
    return { samples.front(), samples[samples.size() / 2] };
}

int main(int argc, char** argv) {
    std::string_view filter = argc > 1 ? argv[1] : "";

#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    std::optional<Image> image = load_grayscale("image.png");
    if (!image) {
        return 1;
    }

    size_t width = (*image)[0].size();
    size_t height = image->size();
    // Prepare the alternative input layouts before any warmup or timing.
    FlatImage flat_image = flatten(*image);
    std::vector<float> flat_image_float(flat_image.begin(), flat_image.end());
    std::printf("image %zux%zu, %d iterations, cycles assume %.1f GHz\n\n", width, height,
        kIterations, kClockGHz);
    std::printf("%-10s %3s %10s %10s %10s %12s %8s\n", "variant", "k", "min us", "median us",
        "ns/pixel", "cycles/pixel", "speedup");

    bool all_correct = true;
    for (int k : kKernelSizes) {
        Kernel kernel = box_kernel(k);
        Image reference = blank_like(*image);
        apply_kernel_v0(*image, kernel, reference);

        size_t radius = k / 2;
        double pixels = double(width - 2 * radius) * double(height - 2 * radius);
        std::optional<double> reference_us;

        Image out = blank_like(*image);
        auto benchmark = [&](std::string_view name, auto run) {
            if (name.find(filter) == std::string_view::npos) {
                return;
            }

            for (auto& row : out) {
                std::ranges::fill(row, 0);
            }
            Timing timing = time_variant([&] {
                run();
                do_not_optimize(out);
            });
            Mismatch mismatch = compare(reference, out);

            if (name == "v0") {
                reference_us = timing.min_us;
            }

            double ns_per_pixel = timing.min_us * 1000.0 / pixels;
            std::printf("%-10.*s %3d %10.1f %10.1f %10.3f %12.2f", int(name.size()), name.data(), k,
                timing.min_us, timing.median_us, ns_per_pixel, ns_per_pixel * kClockGHz);
            if (reference_us) {
                std::printf(" %7.2fx", *reference_us / timing.min_us);
            } else {
                std::printf(" %8s", "-");
            }
            if (mismatch.count > 0) {
                all_correct = false;
                std::printf("  WRONG: %zu pixels off by more than %d (max %d)", mismatch.count,
                    kTolerance, mismatch.max_diff);
            }
            std::printf("\n");
        };

        // Each variant binds its own arguments; the timer only needs a callable.
        benchmark("v0", [&] { apply_kernel_v0(*image, kernel, out); });
        benchmark("v1", [&] { apply_kernel_v1(*image, kernel, out); });
        benchmark("v2", [&] { apply_kernel_v2(flat_image, width, height, kernel, out); });
        benchmark("v3", [&] { apply_kernel_v3(flat_image_float, width, height, kernel, out); });
        std::printf("\n");
    }

    return all_correct ? 0 : 1;
}
