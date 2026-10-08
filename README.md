# imgconv

I'm using a box blur to learn performance engineering on Apple silicon. My workflow is to read the ARM64 assembly, predict what a change will do, then use benchmarks and Instruments to figure out where my reasoning was wrong.

## Run

You'll need a C++20-capable Clang compiler and Make. On macOS, the build also links Apple's Accelerate framework for the vImage comparison. Run these commands from the repository root:

```sh
make
./imgconv          # Compare all variants
./imgconv v2       # Run v2
./imgconv vimage   # Run Apple's implementation (macOS only)
```

The benchmark uses the included `image.png` and box kernels of size 3, 5, 7, and 9. Each run prints kernel-only and end-to-end results, with 5 warmup iterations and 100 timed iterations per case. The optional argument filters variant names in both tables.

## Variants

| Variant | Change |
| --- | --- |
| `v0` | Baseline: runtime kernel size, nested byte vectors |
| `v1` | Make the kernel size a template parameter |
| `v2` | Flatten the byte input to reduce pointer loads |
| `v3` | Convert the flat input to floats before convolution |
| `v4` | Move the output-pixel loop inside the kernel-tap loops |
| `v5`, `v6` | Currently use the same convolution loops and preparation as `v4` |
| `vimage` | Apple's `vImageConvolve_PlanarF`, run single-threaded |

## What the timings include

**Kernel-only** measures convolution on prepared buffers. Input conversion, allocation, accumulator clearing, kernel packing, and vImage scratch setup happen outside the timer. The variants keep their own output formats: `v0`–`v3` write bytes, including the cast inside the loop; `v4`–`v6` and `vimage` write floats. Separate output conversion passes are excluded, so these timings compare kernels with different input and output formats.

**End-to-end** starts with the same nested byte image and produces a nested byte image for each variant. It includes preparation, allocation, zero initialization, convolution, conversion, and temporary-buffer cleanup. It also includes replacing the previous result. Image decoding and kernel generation happen outside the timer.

Both tables use the same interior output pixels. The benchmark reports minimum and median times; ns/pixel and speedup use the minimum. Speedup compares against `v0` for the same kernel size and timing scope, and appears only when the filter includes `v0`.

After timing, the benchmark checks each output against `v0`, allowing a difference of one intensity level for floating-point rounding. The current `v4`, `v5`, and `v6` implementations should give similar results in both tables because they do the same work.
