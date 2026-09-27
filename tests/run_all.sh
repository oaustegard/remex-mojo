#!/usr/bin/env bash
# Build the Python fixtures, then run every Mojo test. Exits non-zero on the
# first failure. Needs `mojo` on PATH and `remex` importable by python3.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tests/build_fixtures.py >/dev/null

for t in test_rng test_rng_numpy test_rotation test_rht test_rht_encode \
         test_codebook test_packing test_encode test_decode \
         test_packed_vectors test_encode_seed test_search_twostage test_ivf; do
    echo "== $t"
    mojo run -I . "tests/$t.mojo"
done

# The GPU tests do not compile on a machine without a supported GPU (Mojo
# 0.26.2 fails in the pass manager), so they only run when asked.
if [[ "${RUN_GPU:-0}" == 1 ]]; then
    for t in test_gpu_encode test_gpu_search; do
        echo "== $t"
        mojo run -I . "tests/$t.mojo"
    done
fi
