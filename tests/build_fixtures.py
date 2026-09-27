"""Build every Python-side fixture the Mojo parity tests read.

    python3 tests/build_fixtures.py

Needs `remex` importable (`pip install remex`). Writes to /tmp, where the
tests look. Every Quantizer here names its rotation: polarquant's
`--seed` path defaults to `haar`, and remex 1.0 changed the Python default
to `rht`, so an unnamed rotation would build fixtures the Mojo side cannot
reproduce. Every one also sets `renorm=False`: the port multiplies raw
norms and does not implement remex's reconstruction-length correction, so
decode and search scores only agree without it. Codes are unaffected.
"""

import os
import sys

import numpy as np

from remex import PackedVectors, Quantizer, save_params, save_pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ivf_fixture  # noqa: E402
import build_rht_fixture  # noqa: E402

ROTATION = "haar"


def encode_params():
    """tests/test_encode.mojo and tests/test_gpu_encode.mojo."""
    np.random.seed(0)
    X = np.random.randn(50, 16).astype(np.float32)
    np.save("/tmp/_parity_X.npy", X)
    q = Quantizer(d=16, bits=4, seed=42, rotation=ROTATION, renorm=False)
    save_params("/tmp/_parity.params", q)
    save_pq("/tmp/_parity_ref.pq", q.encode(X))


def decode():
    """tests/test_decode.mojo: full precision and a nested coarse level."""
    np.random.seed(0)
    n, d, bits = 80, 16, 4
    coarse_precision = 2
    X = np.random.randn(n, d).astype(np.float32)
    q = Quantizer(d=d, bits=bits, seed=42, rotation=ROTATION, renorm=False)
    save_params("/tmp/_decode.params", q)
    cv = q.encode(X)
    save_pq("/tmp/_decode.pq", cv)
    np.save("/tmp/_decode_X.npy", X)
    np.save("/tmp/_decode_full.npy", q.decode(cv).astype(np.float32))
    np.save("/tmp/_decode_coarse.npy",
            q.decode(cv, precision=coarse_precision).astype(np.float32))
    np.save("/tmp/_decode_meta.npy",
            np.array([[coarse_precision]], dtype=np.float32))


def packed_vectors():
    """tests/test_packed_vectors.mojo: round-trip and at_precision."""
    np.random.seed(0)
    n, d, bits = 80, 16, 4
    target_bits = 2
    X = np.random.randn(n, d).astype(np.float32)
    q = Quantizer(d=d, bits=bits, seed=42, rotation=ROTATION, renorm=False)
    cv = q.encode(X)
    packed = PackedVectors.from_compressed(cv)
    np.save("/tmp/_pv_indices.npy", cv.indices.astype(np.float32))
    np.save("/tmp/_pv_indices_at.npy",
            packed.at_precision(target_bits).unpack_rows(0, n).astype(np.float32))
    np.save("/tmp/_pv_meta.npy",
            np.array([[n, d, bits, target_bits]], dtype=np.float32))


def encode_seed():
    """tests/test_encode_seed.mojo: R rebuilt in Mojo from the seed."""
    np.random.seed(0)
    X = np.random.randn(50, 16).astype(np.float32)
    np.save("/tmp/_seed_parity_X.npy", X)
    q = Quantizer(d=16, bits=4, seed=42, rotation=ROTATION, renorm=False)
    save_pq("/tmp/_seed_parity_ref.pq", q.encode(X))


def twostage():
    """tests/test_search_twostage.mojo: top-k parity."""
    np.random.seed(0)
    n, d, bits = 200, 16, 4
    n_q, k, candidates, coarse_precision = 8, 5, 50, 2
    X = np.random.randn(n, d).astype(np.float32)
    Q = np.random.randn(n_q, d).astype(np.float32)
    q = Quantizer(d=d, bits=bits, seed=42, rotation=ROTATION, renorm=False)
    save_params("/tmp/_twostage.params", q)
    cv = q.encode(X)
    save_pq("/tmp/_twostage.pq", cv)
    np.save("/tmp/_twostage_X.npy", X)
    np.save("/tmp/_twostage_Q.npy", Q)
    expected_idx = np.zeros((n_q, k), dtype=np.float32)
    expected_scores = np.zeros((n_q, k), dtype=np.float32)
    for i in range(n_q):
        ti, ts = q.search_twostage(cv, Q[i], k=k, candidates=candidates,
                                   coarse_precision=coarse_precision)
        expected_idx[i] = ti.astype(np.float32)
        expected_scores[i] = ts
    np.save("/tmp/_twostage_meta.npy",
            np.array([[k, candidates, coarse_precision, n_q]], dtype=np.float32))
    np.save("/tmp/_twostage_expected_idx.npy", expected_idx)
    np.save("/tmp/_twostage_expected_scores.npy", expected_scores)


def main():
    encode_params()
    decode()
    packed_vectors()
    encode_seed()
    twostage()
    build_rht_fixture.main("/tmp")
    build_ivf_fixture.main()


if __name__ == "__main__":
    main()
