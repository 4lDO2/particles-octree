# Particle simulation challenge solution

## Practical

Run the code using

```sh
cargo run --release -- /path/to/positions.xyz
```

This code works with Rust Stable or latest nightly.

## Algorithm

The algorithm is based on exploiting the inherently recursive structure of octrees. Space is partitioned wrt XYZ, resulting in boxes of width 0.5^depth. This allows starting with (root, root) and then recursively decomposing each node into 64 possible pairs of nodes, filtering out those where the shortest distance exceeds the specified 0.05m limit. This is inherently parallelizable, especially considering the octree is immutable during the whole process.

## Performance

On my (overclocked) 5950X CPU, the single-threaded staged variant (see singlethread branch) parses the large dataset in 20 ms, builds the tree in 13 ms, and calculates the pairs in 90 ms. The multithreaded variant improves the last step, instead taking 3.5 ms (where 90 ms originally/32 threads = 2.81 ms, which is quite close). For data with sufficiently uniform density, it would in theory be possible to achieve very good parallelism merely by splitting the highest-level (largest) nodes among the thread workers.

I first optimized the algorithm itself (i.e. the lower bound used to filter away node pairs), tried to reduce the time as much as possible when single threaded, and finally implemented work-stealing multithreading with 32 threads. There were some optimizations that intuitively seemed critical, but which unexpectedly resulted in worse perf:

- reusing allocated buffers thread-locally rather than allocating for each batch
- using a separate "finished" queue for nodes where every particle inside is *guaranteed* to be within 0.05m range.

## Caveats

This implementation uses single-precision floating points, due to their small size. The validation code verifies that there were not any conflicting-position particles in the actual data, but for larger data sets it may be necessary to use double-precision floats instead. The ideal method, at least for constructing the tree, would have been to use fixpoint coordinates (in that case, looking at the coordinate bits would directly give the subnode index, rather than having to compare floats). But without AVX512, SIMD support for integer multiplication etc. is quite limited.

Additionally, although not a significant contributor to new unnecessary particles to check, the (particle, node)-pair case treats the node as a sphere, which although valid is not the most accurate lower bound for the distance. Perhaps there is a fast SIMD code sequence for this, but again, finer-grained filtering will be done at the lower-order tree level anyway.
