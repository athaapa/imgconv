# imgconv

image convolution

**purpose**: learn the performance workflow (work out the limit, measure, predict, explain) and ARM SIMD basics on a small problem, so I can apply them to mohg

scope:
- one image
- greyscale
- box blur only at k=3
- single-threaded, one performance core
- direct convolution

i define done as:
- [ ] calculated and checked theoretical limit
- [ ] table that shows each version and its cycles/pixel and an explanation for what changed
- [ ] an explanation of the remaining gap between my version and the limit
- [ ] a short writeup

**end date**: oct 8 
