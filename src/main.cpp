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
#include <Accelerate/Accelerate.h>
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
__attribute__((noinline)) void apply_kernel_v3_fixed(const std::vector<float>& flat_image_float,
    size_t width, size_t height, const Kernel& kernel, Image& output) {
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

template <size_t K>
__attribute__((noinline)) void apply_kernel_v4_fixed(const std::vector<float>& flat_image_float,
    size_t width, size_t height, const Kernel& kernel, std::vector<float>& output_float) {
    const size_t k = K == 0 ? kernel.size() : K;
    const size_t radius = k / 2;

    for (size_t ky = 0; ky < k; ++ky) {
        for (size_t kx = 0; kx < k; ++kx) {
            for (size_t y = radius; y < height - radius; ++y) {
                for (size_t x = radius; x < width - radius; ++x) {
                    size_t iy = y + ky - radius;
                    size_t ix = x + kx - radius;
                    output_float[y * width + x]
                        += flat_image_float[iy * width + ix] * kernel[ky][kx];
                }
            }
        }
    }

}

template <size_t K>
__attribute__((noinline)) void apply_kernel_v5_fixed(const std::vector<float>& flat_image_float,
    size_t width, size_t height, const Kernel& kernel, std::vector<float>& output_float) {
    const size_t k = K == 0 ? kernel.size() : K;
    const size_t radius = k / 2;

    for (size_t ky = 0; ky < k; ++ky) {
        for (size_t kx = 0; kx < k; ++kx) {
            for (size_t y = radius; y < height - radius; ++y) {
                for (size_t x = radius; x < width - radius; ++x) {
                    size_t iy = y + ky - radius;
                    size_t ix = x + kx - radius;
                    output_float[y * width + x]
                        += flat_image_float[iy * width + ix] * kernel[ky][kx];
                }
            }
        }
    }
}

template <size_t K>
__attribute__((noinline)) void apply_kernel_v6_fixed(const std::vector<float>& flat_image_float,
    size_t width, size_t height, const Kernel& kernel, std::vector<float>& output_float) {
    const size_t k = K == 0 ? kernel.size() : K;
    const size_t radius = k / 2;

    for (size_t ky = 0; ky < k; ++ky) {
        for (size_t kx = 0; kx < k; ++kx) {
            for (size_t y = radius; y < height - radius; ++y) {
                for (size_t x = radius; x < width - radius; ++x) {
                    size_t iy = y + ky - radius;
                    size_t ix = x + kx - radius;
                    output_float[y * width + x]
                        += flat_image_float[iy * width + ix] * kernel[ky][kx];
                }
            }
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

void apply_kernel_v4(const std::vector<float>& image, size_t width, size_t height,
    const Kernel& kernel, std::vector<float>& output) {
    switch (kernel.size()) {
    case 3:
        return apply_kernel_v4_fixed<3>(image, width, height, kernel, output);
    case 5:
        return apply_kernel_v4_fixed<5>(image, width, height, kernel, output);
    case 7:
        return apply_kernel_v4_fixed<7>(image, width, height, kernel, output);
    case 9:
        return apply_kernel_v4_fixed<9>(image, width, height, kernel, output);
    default:
        return apply_kernel_v4_fixed<0>(image, width, height, kernel, output);
    }
}

void apply_kernel_v5(const std::vector<float>& image, size_t width, size_t height,
    const Kernel& kernel, std::vector<float>& output) {
    switch (kernel.size()) {
    case 3:
        return apply_kernel_v5_fixed<3>(image, width, height, kernel, output);
    case 5:
        return apply_kernel_v5_fixed<5>(image, width, height, kernel, output);
    case 7:
        return apply_kernel_v5_fixed<7>(image, width, height, kernel, output);
    case 9:
        return apply_kernel_v5_fixed<9>(image, width, height, kernel, output);
    default:
        return apply_kernel_v5_fixed<0>(image, width, height, kernel, output);
    }
}

void apply_kernel_v6(const std::vector<float>& image, size_t width, size_t height,
    const Kernel& kernel, std::vector<float>& output) {
    switch (kernel.size()) {
    case 3:
        return apply_kernel_v6_fixed<3>(image, width, height, kernel, output);
    case 5:
        return apply_kernel_v6_fixed<5>(image, width, height, kernel, output);
    case 7:
        return apply_kernel_v6_fixed<7>(image, width, height, kernel, output);
    case 9:
        return apply_kernel_v6_fixed<9>(image, width, height, kernel, output);
    default:
        return apply_kernel_v6_fixed<0>(image, width, height, kernel, output);
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

// Match the byte-output variants' truncation and tolerance, outside timing.
Mismatch compare(const Image& expected, const std::vector<float>& actual) {
    Mismatch result;
    const size_t width = expected[0].size();
    for (size_t y = 0; y < expected.size(); ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto value = static_cast<std::uint8_t>(actual[y * width + x]);
            int diff = std::abs(int(expected[y][x]) - int(value));
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

template <typename Run, typename Setup> Timing time_variant(Run run, Setup setup) {
    for (int i = 0; i < kWarmupIterations; ++i) {
        setup();
        run();
    }

    std::vector<double> samples(kIterations);
    for (double& sample : samples) {
        setup(); // Per-invocation preparation is excluded from the measurement.
        auto start = std::chrono::steady_clock::now();
        run();
        auto end = std::chrono::steady_clock::now();
        sample = std::chrono::duration<double, std::micro>(end - start).count();
    }

    std::ranges::sort(samples);
    return { samples.front(), samples[samples.size() / 2] };
}

// Each invocation owns its preparation so end-to-end includes allocation and cleanup.
struct Prepared {
    FlatImage bytes;
    std::vector<float> floats;
    Image output;
    std::vector<float> float_output;
#ifdef __APPLE__
    std::vector<float> coefficients;
    std::vector<std::uint8_t> scratch;
#endif
};

constexpr std::string_view kVariantNames[] = {
    "v0", "v1", "v2", "v3", "v4", "v5", "v6",
#ifdef __APPLE__
    "vimage",
#endif
};

#ifdef __APPLE__
vImage_Error convolve_vimage(Prepared& work, size_t width, size_t height,
    size_t k, vImage_Flags extra_flags = kvImageNoFlags) {
    const size_t radius = k / 2;
    vImage_Buffer src { work.floats.data(), height, width, width * sizeof(float) };
    vImage_Buffer dst { work.float_output.data() + radius * width + radius,
        height - 2 * radius, width - 2 * radius, width * sizeof(float) };
    return vImageConvolve_PlanarF(&src, &dst, work.scratch.data(), radius, radius,
        work.coefficients.data(), static_cast<uint32_t>(k), static_cast<uint32_t>(k),
        0.0f, kvImageEdgeExtend | kvImageDoNotTile | extra_flags);
}

void check_vimage(vImage_Error error) {
    if (error < 0) {
        std::fprintf(stderr, "vImage failed: %ld\n", long(error));
        std::exit(1);
    }
}
#endif

Prepared prepare(size_t variant, const Image& image, const Kernel& kernel) {
    Prepared work;
    const size_t width = image[0].size();
    const size_t height = image.size();
    if (variant == 2) {
        work.bytes = flatten(image);
    } else if (variant >= 3) {
        work.floats.resize(width * height);
        for (size_t y = 0; y < height; ++y) {
            std::copy(image[y].begin(), image[y].end(), work.floats.begin() + y * width);
        }
    }
    if (variant < 4) {
        work.output = blank_like(image);
    } else {
        work.float_output.resize(width * height);
    }
#ifdef __APPLE__
    if (variant == 7) {
        for (const auto& row : kernel) {
            work.coefficients.insert(work.coefficients.end(), row.begin(), row.end());
        }
        const auto bytes = convolve_vimage(work, width, height, kernel.size(),
            kvImageGetTempBufferSize);
        check_vimage(bytes);
        work.scratch.resize(std::max<size_t>(1, size_t(bytes)));
    }
#endif
    return work;
}

void run_kernel(size_t variant, const Image& image, const Kernel& kernel, Prepared& work) {
    const size_t width = image[0].size();
    const size_t height = image.size();
    switch (variant) {
    case 0: apply_kernel_v0(image, kernel, work.output); break;
    case 1: apply_kernel_v1(image, kernel, work.output); break;
    case 2: apply_kernel_v2(work.bytes, width, height, kernel, work.output); break;
    case 3: apply_kernel_v3(work.floats, width, height, kernel, work.output); break;
    case 4: apply_kernel_v4(work.floats, width, height, kernel, work.float_output); break;
    case 5: apply_kernel_v5(work.floats, width, height, kernel, work.float_output); break;
    case 6: apply_kernel_v6(work.floats, width, height, kernel, work.float_output); break;
#ifdef __APPLE__
    case 7: check_vimage(convolve_vimage(work, width, height, kernel.size())); break;
#endif
    }
    do_not_optimize(work.output);
    do_not_optimize(work.float_output);
}

Image finish(Prepared& work, const Image& image) {
    if (!work.output.empty()) {
        return std::move(work.output);
    }
    Image result = blank_like(image);
    const size_t width = image[0].size();
    for (size_t y = 0; y < image.size(); ++y) {
        for (size_t x = 0; x < width; ++x) {
            result[y][x] = static_cast<std::uint8_t>(work.float_output[y * width + x]);
        }
    }
    return result;
}

int main(int argc, char** argv) {
    std::string_view filter = argc > 1 ? argv[1] : "";
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    auto image = load_grayscale("image.png");
    if (!image) {
        return 1;
    }
    const size_t width = (*image)[0].size();
    const size_t height = image->size();
    std::printf("image %zux%zu, %d iterations\n", width, height, kIterations);
    bool all_correct = true;
    for (bool end_to_end : { false, true }) {
        std::printf("\n%s\n", end_to_end ? "END-TO-END" : "KERNEL-ONLY");
        std::printf("%s\n", end_to_end
            ? "Nested byte input -> nested byte output; preparation, allocation, conversion and temporary cleanup timed."
            : "Prepared native buffers; allocation, clearing and separate conversion passes excluded.");
        if (!end_to_end) {
            std::printf("v0-v3 write bytes (fused conversion timed); v4-v6/vimage write floats.\n");
        }
        std::printf("%-10s %3s %10s %10s %10s %8s\n", "variant", "k", "min us",
            "median us", "ns/pixel", "speedup");
        for (int k : kKernelSizes) {
            Kernel kernel = box_kernel(k);
            Image reference = blank_like(*image);
            apply_kernel_v0(*image, kernel, reference);
            const size_t radius = k / 2;
            const double pixels = double(width - 2 * radius) * double(height - 2 * radius);
            std::optional<double> reference_us;
            for (size_t variant = 0; variant < std::size(kVariantNames); ++variant) {
                const auto name = kVariantNames[variant];
                if (name.find(filter) == std::string_view::npos) {
                    continue;
                }
                Timing timing;
                Mismatch mismatch;
                if (end_to_end) {
                    Image result;
                    timing = time_variant([&] {
                        auto work = prepare(variant, *image, kernel);
                        run_kernel(variant, *image, kernel, work);
                        result = finish(work, *image);
                        do_not_optimize(result);
                        // work's temporary buffers are released before the timer stops.
                    }, [] {});
                    mismatch = compare(reference, result);
                } else {
                    auto work = prepare(variant, *image, kernel);
                    timing = time_variant([&] {
                        run_kernel(variant, *image, kernel, work);
                    }, [&] {
                        // Accumulating kernels require a fresh zero output each invocation.
                        if (variant >= 4 && variant <= 6) {
                            std::ranges::fill(work.float_output, 0.0f);
                            do_not_optimize(work.float_output);
                        }
                    });
                    mismatch = variant < 4 ? compare(reference, work.output)
                                           : compare(reference, work.float_output);
                }
                if (variant == 0) {
                    reference_us = timing.min_us;
                }
                std::printf("%-10.*s %3d %10.1f %10.1f %10.3f", int(name.size()), name.data(),
                    k, timing.min_us, timing.median_us, timing.min_us * 1000.0 / pixels);
                if (reference_us) {
                    std::printf(" %7.2fx", *reference_us / timing.min_us);
                } else {
                    std::printf(" %8s", "-");
                }
                if (mismatch.count > 0) {
                    all_correct = false;
                    std::printf("  WRONG: %zu pixels off by more than %d (max %d)",
                        mismatch.count, kTolerance, mismatch.max_diff);
                }
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
    return all_correct ? 0 : 1;
}
