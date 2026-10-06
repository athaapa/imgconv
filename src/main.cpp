#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

constexpr int kIterations = 100;
constexpr int kWarmupIterations = 5;

using ApplyKernelFunction = std::function<void(std::vector<std::vector<std::uint8_t>>&,
    std::vector<std::vector<float>>&, std::vector<std::vector<std::uint8_t>>&)>;

__attribute__((noinline)) void apply_kernel_v0(std::vector<std::vector<std::uint8_t>>& image,
    std::vector<std::vector<float>>& kernel, std::vector<std::vector<std::uint8_t>>& output) {
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

template <typename T> void do_not_optimize(T const& value) {
    asm volatile("" : : "g"(&value) : "memory");
}

std::optional<std::chrono::microseconds> apply_blur(int k, ApplyKernelFunction func) {
    int width, height, n;
    unsigned char* data = stbi_load("image.png", &width, &height, &n, 1);

    if (data != NULL) {
        std::vector<std::vector<std::uint8_t>> image(height, std::vector<std::uint8_t>(width));

        size_t len = width * height;
        for (int i = 0; i < len; i++) {
            std::uint8_t value = static_cast<std::uint8_t>(data[i]);
            image[i / width][i % width] = value;
        }

        float weight = 1.0f / (k * k);

        std::vector<std::vector<float>> kernel(k, std::vector<float>(k, weight));

        std::vector<std::vector<std::uint8_t>> out(height, std::vector<std::uint8_t>(width));

        for (int i = 0; i < kWarmupIterations; ++i) {
            func(image, kernel, out);
            do_not_optimize(out);
        }

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < kIterations; ++i) {
            func(image, kernel, out);
            do_not_optimize(out);
        }

        auto end = std::chrono::steady_clock::now();
        auto elapsed = end - start;

        // verification
        std::vector<std::vector<std::uint8_t>> out_v0(height, std::vector<std::uint8_t>(width));
        apply_kernel_v0(image, kernel, out_v0);
        if (out_v0 != out) {
            std::cerr << "image verification failed\n";
            return std::nullopt;
        }

        stbi_image_free(data);

        return std::chrono::duration_cast<std::chrono::microseconds>(elapsed / kIterations);
    }

    std::cerr << "failed to load image: " << stbi_failure_reason() << "\n";
    return std::nullopt;
}

bool verify(
    std::vector<std::vector<std::uint8_t>>& source, std::vector<std::vector<std::uint8_t>>& x) {
    if (source.size() != x.size())
        return false;
    if (source.size() == 0)
        return true;
    if (source[0].size() != x[0].size())
        return false;

    for (size_t i = 0; i < source.size(); ++i) {
        for (size_t j = 0; j < source[i].size(); ++j) {
            // TODO: Consider using a tolerance for floating point error
            if (source[i][j] != x[i][j])
                return false;
        }
    }

    return true;
}

int main() {
    std::cout << "Running benchmark...\n";

    std::vector<ApplyKernelFunction> functions = { apply_kernel_v0 };

    for (size_t i = 0; i < functions.size(); i++) {
        for (int k = 3; k <= 9; k += 2) {
            auto maybe_duration = apply_blur(k, functions[i]);
            if (maybe_duration.has_value()) {
                auto duration = maybe_duration.value();

                std::cout << k << "x" << k << " kernel: " << duration << '\n';
            } else {
                return 1;
            }
        }
    }

    return 0;
}
