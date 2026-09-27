# remex-mojo (`polarquant`)

A pure-Mojo port of the [remex](https://github.com/oaustegard/remex)
Quantizer's encode, ADC search and decode paths, shipped as a standalone
CLI binary. It reads `.npy` corpus files and writes the same `.pq`
container that `remex.load_pq()` reads.

This code lived in `remex/mojo/` until remex 1.0 and was moved here with
its history. Bare issue and PR numbers below (#5, #40, PR #37, ...) refer to
[oaustegard/remex](https://github.com/oaustegard/remex/issues).

Requirements:

- **Mojo 0.26.2.** Mojo 1.x does not parse this code yet (`MutExternalOrigin`
  and friends are gone). `pip install "mojo==0.26.2.0"` installs it.
- **remex >= 1.0** for the parity tests, which build their expected values
  with the Python library. remex 1.0 changed the `rht` construction at
  power-of-two `d` (remex #89), and this port follows it.

## What's here

```
remex-mojo/
├── README.md                # This file
├── polarquant.mojo          # CLI entrypoint (encode + search + decode)
├── src/
│   ├── mathx.mojo           # erf-based normal CDF / PDF
│   ├── rng.mojo             # xoshiro256++ + Marsaglia polar normal
│   ├── matrix.mojo          # Owning Float32 / Float64 matrices
│   ├── rotation.mojo        # Householder QR → Haar, and randomized Hadamard (RHT)
│   ├── codebook.mojo        # Lloyd-Max iteration on N(0, 1/d) + Matryoshka nested tables
│   ├── packing.mojo         # 1/2/3/4/8-bit pack and unpack
│   ├── packed_vectors.mojo  # Bit-packed in-memory storage with on-demand unpack
│   ├── npy.mojo             # .npy reader/writer (float32, 2D, C-contiguous)
│   ├── pq_format.mojo       # .pq binary format read/write
│   ├── params_format.mojo   # .params dump (R + boundaries + centroids)
│   ├── quantizer.mojo       # Quantizer struct: encode + ADC search + two-stage + decode
│   ├── ivf.mojo             # IVFCoarseIndex: data-oblivious IVF over Matryoshka coarse tier
│   └── gpu/                 # GPU/MAX kernels — scaffolding, see issue #42
│       ├── device.mojo      # is_gpu_available() probe
│       ├── encode.mojo      # gpu_encode_batch (stub)
│       └── adc.mojo         # gpu_adc_search (stub)
├── tests/
│   ├── test_rng.mojo
│   ├── test_rotation.mojo
│   ├── test_rht.mojo               # RHT byte parity vs Python + orthogonality
│   ├── test_rht_encode.mojo        # RHT encode parity, --seed and --params routes
│   ├── build_rht_fixture.py        # Python fixture builder for the two above
│   ├── test_codebook.mojo
│   ├── test_packing.mojo
│   ├── test_packed_vectors.mojo    # PackedVectors round-trip + at_precision parity
│   ├── test_encode.mojo            # bit-identical encode parity vs Python
│   ├── test_decode.mojo            # decode parity vs Python (full + coarse precision)
│   ├── test_search_twostage.mojo   # top-k parity for search_twostage vs Python
│   ├── test_ivf.mojo               # IVFCoarseIndex parity vs Python (cell IDs + search)
│   ├── build_ivf_fixture.py        # Python fixture builder for test_ivf.mojo
│   ├── build_fixtures.py           # builds every fixture, including the two above
│   ├── run_all.sh                  # fixtures + every CPU test
│   ├── test_gpu_encode.mojo        # encode parity for --device gpu (skipped without GPU)
│   └── test_gpu_search.mojo        # ADC parity vs CPU for --device gpu (skipped without GPU)
└── bench/
    ├── bench_encode.mojo
    ├── bench_search.mojo
    ├── bench_twostage.mojo
    ├── bench_ivf.mojo              # IVF two-stage latency at varying nprobe
    ├── bench_gpu_encode.mojo       # GPU encode timing (skipped without GPU)
    ├── bench_gpu_search.mojo       # GPU ADC timing (skipped without GPU)
    └── compare.py           # Mojo vs NumPy comparison driver
```

## Build

Install the Mojo compiler (~500 MB) and, for the tests and benchmarks,
the Python library:

```bash
pip install "mojo==0.26.2.0" "remex>=1.0"
```

Build the CLI and the bench binaries from the repository root. The CLI
links the GPU kernels, and on a host where Mojo 0.26.2 detects no GPU it
stops with "Unknown GPU architecture detected". Name one with
`--target-accelerator nvidia:80` and the build succeeds; the GPU paths
still refuse at runtime and `--device auto` falls back to the CPU.

```bash
mojo build -I . polarquant.mojo            -o polarquant
mojo build -I . bench/bench_encode.mojo    -o bench/bench_encode
mojo build -I . bench/bench_search.mojo    -o bench/bench_search
mojo build -I . bench/bench_twostage.mojo  -o bench/bench_twostage
mojo build -I . bench/bench_ivf.mojo       -o bench/bench_ivf
```

## CLI usage

```bash
# Encode an .npy of float32 vectors → .pq
./polarquant encode corpus.npy --bits 4 --seed 42 -o corpus.pq

# Same, with the randomized Hadamard rotation (cheaper to build at large d).
# --rotation is part of the encoding: pass the same value to search/decode.
./polarquant encode corpus.npy --bits 4 --seed 42 --rotation rht -o corpus.pq

# Search a single (1, d) query against a .pq, top-k
./polarquant search corpus.pq query.npy --k 10 --seed 42 --top 10

# Memory-efficient two-stage retrieval: coarse ADC scan at reduced
# precision, full-precision rerank on the top `--candidates` rows.
./polarquant search corpus.pq query.npy --k 10 --seed 42 \
    --twostage --candidates 500 --coarse-precision 2

# Reconstruct float32 vectors from a .pq → .npy. Optional --precision
# uses the nested codebook at a coarser bit width (1..bits).
./polarquant decode corpus.pq --params P.bin -o reconstructed.npy
./polarquant decode corpus.pq --params P.bin --precision 2 -o coarse.npy
```

The same `corpus.pq` round-trips through the Python library:

```python
from remex import load_pq, save_pq, Quantizer
cv = load_pq("corpus.pq")            # works either way
print(cv.n, cv.d, cv.bits, cv.compression_ratio)
```

## Two parameter modes — `--seed` vs `--params`

The encode/search path needs (R, boundaries, centroids). The CLI
provides those two ways:

| Flag | Source of (R, codebook) | Bit-identical to the matching Python `Quantizer`? |
|---|---|---|
| `--seed S` *(default RNG: numpy, rotation: haar)* | Computed in Mojo from S via PCG64 + SeedSequence + Ziggurat + Householder QR + Lloyd-Max | **Yes**, at 1–4 bits. **8-bit:** see caveat below. |
| `--seed S --rotation rht` | Computed in Mojo from S via PCG64 + SeedSequence + permute/sign/FWHT + Lloyd-Max | **R: yes.** Codes: Python now applies `rht` in operator form and takes boundaries from float32 centroids, while this port encodes through the dense matrix, so about 1e-6 of coordinates can differ. Porting `RHTOperator` would restore exact parity. |
| `--seed S --rng xoshiro` | Computed in Mojo from S via xoshiro256++ + Marsaglia + Householder QR + Lloyd-Max | **No.** Both Haar samples but from different Gaussian streams. Faster init, no Ziggurat tables. |
| `--params P.bin` | Loaded from a file written by `remex.save_params(quantizer, P)` | **Yes**, at all bit widths and for either rotation. |

`--seed` (numpy mode, the default) gives a self-contained Mojo workflow
with end-to-end byte parity vs Python: `polarquant encode X.npy --bits 4
--seed 42 -o out.pq` produces the same bytes as Python's
`save_pq(Quantizer(d, 4, seed=42, rotation="haar").encode(X))`.

`--rotation` defaults to `haar` here. remex's own default became `rht` in
1.0, so a Python `Quantizer` built without `rotation=` does not match a
`polarquant` run without `--rotation`. The mismatch is loud: the `.pq`
records its rotation, and decoding under the other one raises on both
sides. This was the goal of
issue #40 and uses the new `src/rng_numpy.mojo` module.

`--seed --rng xoshiro` is the legacy fast path. Use it when you don't
need Python parity — startup is slightly faster (no 25 KB of Ziggurat
tables to populate).

`--rotation rht` builds the randomized Hadamard transform instead of the
Haar matrix — `O(d^2 log d)` instead of the QR's `O(d^3)`, off the same
NumPy PCG64 stream Python uses, so `polarquant encode --seed 42
--rotation rht` and `Quantizer(d, bits, seed=42, rotation="rht")` agree
byte for byte. It needs an even `d`, and it only combines with the numpy
RNG (there is no xoshiro variant of the construction). The rotation is
part of the encoding: an index encoded with `--rotation rht` must be
searched and decoded with it too.

`--params` remains the canonical bridge for any case where you want
guaranteed bit parity regardless of bit width or potential drift.

### The index records its rotation

`.pq` byte 17 holds the rotation code (`0` = haar, `1` = rht; Python also
writes `2` = none, see below) and
`.params` byte 9 mirrors it. Both were reserved-and-zero before the field
existed, and haar is `0`, so every previously written file reads back as
haar with no compatibility branch — which is the correct answer for all
of them. `search` and `decode` refuse a `--rotation` that disagrees with
what the index records, instead of silently decoding against the wrong
rotation; the two constructions differ on roughly half their stored bits,
so the results would look plausible and be wrong. `--params` skips the
check, since the matrix comes from the file and there is nothing to
disagree with.

### Not ported: scalar mode

Python's `Quantizer(normalize=False)` (remex #77) quantizes coordinates
directly, storing no norms, and pairs with a third rotation code (`2` =
none, the identity). `polarquant` implements neither, and needs no
compatibility branch to stay safe: rotation code `2` fails the `> 1` check
in `load_pq`, and a scalar-mode `.pq` has no norms section, so it is
`n * 4` bytes shorter than the reader's length arithmetic expects and is
rejected as truncated. The flag that marks it lives in byte 18, which was
reserved-and-zero before, so files written by this port are unaffected in
either direction.

### 8-bit byte-parity caveat

At `--bits 8`, the Lloyd-Max codebook uses 256 levels and runs 300
iterations of refinement. Mojo's `std.math.erf` differs from libm's
`math.erf` (used by `scipy.stats.norm.cdf`) by ~1e-8 per call. Across
300 iterations × 256 levels these accumulate, shifting a handful of
codebook boundaries by ~1e-3. Net effect: ~0.1–0.2% of indices end up
1 level off. The encoded `.pq` is ~99.8% byte-identical but not
strictly so. For 1/2/3/4 bit, the codebook has fewer levels and far
more margin, so byte parity holds exactly.

If you need byte parity at 8-bit, use `--params`. (This is a Mojo
stdlib precision limitation, not a remex algorithm issue.)

## Tests

```bash
tests/run_all.sh            # builds the Python fixtures, then every CPU test
RUN_GPU=1 tests/run_all.sh  # also the two GPU tests
```

`tests/build_fixtures.py` writes everything the parity tests read into
`/tmp`. Every `Quantizer` in it names `rotation=` (the `--seed` path's
default is `haar`, remex's is `rht`) and sets `renorm=False`: this port
multiplies raw norms and does not implement remex's reconstruction-length
correction, which remex applies by default. Codes do not depend on
`renorm`; decoded vectors and search scores do.

The GPU tests do not compile on a host without a supported GPU under Mojo
0.26.2 (the pass manager fails), which is why `run_all.sh` leaves them out
unless asked.

## Benchmarks

```bash
python bench/compare.py --n 10000 --d 384 --bits 4 --queries 100 --k 10

# IVF latency at varying nprobe (auto doubling sweep up to n_cells).
./bench/bench_ivf --n 100000 --d 384 --bits 4 --queries 100 --k 10 \
    --n-bits 8 --mode rotated_prefix --candidates 500 --coarse-precision 2
```

See `bench/RESULTS.md` for current numbers.

## GPU / MAX path (`--device auto|cpu|gpu`)

**Status: kernels live, auto policy steers per stage.** Encode and
ADC-search kernels in `src/gpu/encode.mojo` and `src/gpu/adc.mojo`
landed via PRs [#65](https://github.com/oaustegard/remex/pull/65)
and [#66](https://github.com/oaustegard/remex/pull/66). The CLI now
defaults to `--device auto`, which resolves per stage based on what's
been measured:

- **encode → CPU.** Mojo CPU encode is 11.7 µs/vec at d=384; Apple Metal
  GPU is 43.1 µs/vec (3.7× slower) on the same host (PR
  [#67](https://github.com/oaustegard/remex/pull/67)). NVIDIA/AMD encode
  is unmeasured; auto stays on the safe path until that changes.
- **search → GPU when an accelerator is reachable.** GPU ADC search with
  the corpus cache is 1.30× faster than CPU on Apple Metal at d=384
  (PR [#66](https://github.com/oaustegard/remex/pull/66)); falls back
  to CPU when no accelerator is available.

Use `--device cpu` or `--device gpu` to force a backend explicitly —
useful for benchmarking or for non-Apple GPU hosts where the auto
policy may be too conservative on encode. Forcing `--device gpu` for
encode on Apple Metal still works but prints a one-line warning. See
[issue #42](https://github.com/oaustegard/remex/issues/42) for the
broader GPU acceptance criteria.

### Build

The GPU build needs MAX with a CUDA-capable backend. CPU-only hosts
(M-series Macs, generic Linux without an NVIDIA GPU) can still build
and run the CLI with `--target-accelerator nvidia:80` (see § Build) — the
GPU paths refuse at runtime via `is_gpu_available()`.

```bash
# Same Mojo install as the CPU build, plus MAX for the GPU backend.
pip install "mojo==0.26.2.0" max

# Build the CLI + GPU bench binaries (same flags as CPU).
mojo build -I . polarquant.mojo                -o polarquant
mojo build -I . bench/bench_gpu_encode.mojo    -o bench/bench_gpu_encode
mojo build -I . bench/bench_gpu_search.mojo    -o bench/bench_gpu_search
```

### Run

```bash
# Default: auto picks CPU encode, GPU search on hosts with an accelerator.
./polarquant encode corpus.npy --bits 4 --params P.bin -o corpus.pq
./polarquant search corpus.pq query.npy --k 10 --params P.bin --top 10

# Force a backend (e.g. for benchmarking).
./polarquant encode corpus.npy --bits 4 --params P.bin --device gpu -o corpus.pq
./polarquant search corpus.pq query.npy --k 10 --params P.bin --device cpu --top 10
```

### Tests

```bash
# Needs a supported GPU: on a CPU-only host these fail to compile.
RUN_GPU=1 tests/run_all.sh
```

`test_gpu_encode.mojo` reuses the `/tmp/_parity_*` fixtures already
generated for `test_encode.mojo` (see § Tests above). `test_gpu_search`
is self-contained: it builds a synthetic corpus and asserts the GPU
top-k matches the CPU `adc_search` baseline (rtol=1e-5 on scores,
identical indices).

### Acceptance for the kernel implementation

Per issue #42:

- `polarquant encode --device gpu` produces packed indices byte-identical
  to the CPU path on the same input + `(R, codebook)`, modulo a
  documented FP-order tolerance for boundary-adjacent coordinates.
- `bench/RESULTS.md § Mojo port` gains a row for GPU encode + search
  with timings benchmarked against a CuPy/PyTorch baseline on the same
  GPU host (not against the Mojo CPU bench).

### Why this is its own follow-up

GPU work needs an NVIDIA host to validate. The kernel implementation
doesn't gate on this scaffolding: anyone with a supported GPU can pick
up `src/gpu/encode.mojo` and `src/gpu/adc.mojo`, replace the `raise
Error(...)` body, and the rest of the pipeline (CLI, tests, bench)
already works.

## Notes / known gaps

- **Encode hot loop is SIMD-vectorized + NB=8 row-blocked.** The per-row
  rotation matvec and the squared-norm reduction in `encode_batch` use a
  `simd_width_of[DType.float32]()`-wide FMA + horizontal reduce — see
  `_dot_f32`, `_dot_block_8`, and `_sumsq_f64` in `src/quantizer.mojo`.
  On top of that, `encode_batch` processes 8 rows of X at a time through
  `_dot_block_8`, which fuses 8 dot products against the same R[k, :]
  into eight SIMD accumulators sharing one R-load per j-step. R memory
  traffic drops 8× — the unblocked path reloaded the full ~d²·4 bytes
  of R per row, which doesn't fit in L2 at d=384. The combined effect
  brings Mojo encode from 179 µs/vec (pre-SIMD) → 21 µs/vec (PR #37)
  → 12 µs/vec (this) at d=384, **1.26× faster than NumPy's BLAS
  `X @ R.T`** on AVX-512 (issue #41).
- **`search_twostage` is naive.** The coarse stage is a straightforward
  per-row ADC table lookup (no SIMD on the lookup gather), and the
  candidate selection is O(n*candidates). Mirrors the structure of
  `adc_search` in this port. Tighter inner-loop kernels — especially
  for the coarse-stage reduction at low precision — are a follow-up.
- **GPU kernels live; default policy is per-stage.** `--device auto`
  (the default) routes encode to CPU and ADC search to GPU when an
  accelerator is present. Apple Metal GPU encode is measured 3.7× slower
  than CPU at d=384 (PR #67); NVIDIA/AMD encode is unmeasured, and the
  auto policy stays conservative there until a baseline lands. See the
  `## GPU / MAX path` section above and issue #42.
- **Seed parity** with NumPy's `default_rng` is now bit-identical at
  1–4 bits via `src/rng_numpy.mojo` (PCG64 + SeedSequence + Ziggurat,
  issue #40). 8-bit byte parity is blocked by Mojo `std.math.erf`
  precision drift in the 256-level Lloyd-Max iteration — see the 8-bit
  caveat above; use `--params` for guaranteed parity at 8-bit.
- **UnsafePointer field aliasing.** Several spots copy struct-owned
  buffers into a fresh local allocation before passing the pointer to
  another function. Direct `struct.field[i]` reads through an
  `UnsafePointer` field can return stale/garbage values for the first
  one or two indices on Mojo 0.26.2 in this container. The copies are
  flagged in comments where they happen.
