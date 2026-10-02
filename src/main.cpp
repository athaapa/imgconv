#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include <cstdint>
#include <iostream>
#include <vector>

std::vector<std::vector<std::uint8_t>> apply_kernel(
    std::vector<std::vector<std::uint8_t>>& image, std::vector<std::vector<float>>& kernel) {
    size_t width = image[0].size();
    size_t height = image.size();
    size_t k = kernel.size();

    std::vector<std::vector<std::uint8_t>> output(width, std::vector<std::uint8_t>(width));

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

    return output;
}

int main(int argc, char* argv[]) {
    int width, height, n;
    unsigned char* data = stbi_load("image.png", &width, &height, &n, 1);

    if (data != NULL) {
        std::vector<std::vector<std::uint8_t>> image(height, std::vector<std::uint8_t>(width));

        size_t len = width * height;
        for (int i = 0; i < len; i++) {
            std::uint8_t value = static_cast<std::uint8_t>(data[i]);
            image[i / width][i % width] = value;
        }

        int k = 15;
        float weight = 1.0f / (k * k);

        std::vector<std::vector<float>> kernel(k, std::vector<float>(k, weight));

        auto out = apply_kernel(image, kernel);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                unsigned char value = static_cast<unsigned char>(out[i][j]);
                data[i * width + j] = value;
            }
        }

        stbi_write_png("new.png", width, height, 1, data, width);

        stbi_image_free(data);
    } else {
        std::cerr << "failed to load image: " << stbi_failure_reason() << "\n";
    }
}
