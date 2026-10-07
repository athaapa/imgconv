# imgconv

A C++ performance engineering project exploring image convolution on Apple silicon. I compare implementations, inspect the generated ARM64 assembly, and use benchmarks and Instruments to test predictions about vectorization, memory access, and instruction dependencies.

The experiments start with a direct convolution baseline and change one aspect at a time:

| Variant | Experiment |
| --- | --- |
| `v0` | Runtime kernel size with nested image vectors |
| `v1` | Compile-time kernel sizes |
| `v2` | Flat byte input to reduce pointer indirection |
| `v3` | Flat float input to move pixel conversion outside the convolution |

This is an ongoing learning project. The focus is understanding why a change affects performance and revising the explanation when measurements disagree.

## Run

Requires a C++20-capable Clang compiler and Make. From the repository root:

```sh
make
./imgconv       # Compare all variants
./imgconv v2    # Run one variant
```

The benchmark uses the included `image.png` with 3×3, 5×5, 7×7, and 9×9 box kernels. It reports minimum and median times across 100 iterations after warmup, plus time per output pixel and speedup against the baseline when included.

Input preparation happens outside the timed region. Outputs are checked against `v0` with a one-level pixel tolerance for floating-point differences. The cycles/pixel column estimates cycles using a fixed 4.4 GHz clock; it is not a hardware-counter measurement.
