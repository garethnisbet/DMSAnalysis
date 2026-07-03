"""Self-verification of the pseudo-cubic re-indexing matrices.

The 12 ``PSEUDOCUBIC_TRANSFORMS`` are transcribed from Table 1 of Nisbet et al.
(2023), J. Appl. Cryst. 56, 1046-1050.  These tests re-derive the underlying
group theory (coset decomposition of the rhombohedral point group in the cubic
holohedry — the twinning-by-pseudo-merohedry construction of Flack, 1987) and
check the hard-coded table against it, so a transcription slip cannot pass
silently.

Run standalone:
    python -m DMSAnalysis.tests.test_pseudocubic
or under pytest:
    pytest DMSAnalysis/tests/test_pseudocubic.py
"""

import numpy as np

from .. import ts_quasi as ts


def test_all_matrices_are_proper_cubic_rotations():
    O = ts.cubic_proper_rotations()
    assert len(O) == 24
    for i in range(1, len(ts.PSEUDOCUBIC_TRANSFORMS) + 1):
        M = ts.pseudocubic_matrix(i)
        assert ts._mat_in(M, O), 'matrix %d is not a proper cubic rotation' % i
        assert int(round(np.linalg.det(M))) == 1
        assert np.array_equal(M @ M.T, np.eye(3, dtype=int))


def test_first_matrix_is_identity():
    assert np.array_equal(ts.pseudocubic_matrix(1), np.eye(3, dtype=int))


def test_rhombohedral_group_is_order_6_subgroup():
    O = ts.cubic_proper_rotations()
    D3 = ts.rhombohedral_proper_group()
    assert len(D3) == 6
    assert all(ts._mat_in(g, O) for g in D3)
    # closed under multiplication
    assert all(ts._mat_in(a @ b, D3) for a in D3 for b in D3)


def test_four_domains_by_coset_decomposition():
    cosets = ts.coset_decomposition(ts.cubic_proper_rotations(),
                                    ts.rhombohedral_proper_group())
    assert len(cosets) == 4
    assert all(len(c) == 6 for c in cosets)
    # 4 canonical domain representatives, identity first
    domains = ts.pseudocubic_domains()
    assert len(domains) == 4
    assert np.array_equal(domains[0], np.eye(3, dtype=int))


def test_table_matches_coset_derivation():
    # The headline check: table is correct and complete (3 entries per domain,
    # every domain represented).  Raises AssertionError on any mismatch.
    summary = ts.verify_pseudocubic_transforms()
    assert summary == {'n_domains': 4, 'per_domain': [3, 3, 3, 3],
                       'n_matrices': 12}


def test_published_duplicate_is_flagged():
    # Matrix 9 is identical to matrix 2 as published — documented in the source
    # comment; assert it here so the duplicate is a known, tested fact rather
    # than a surprise.
    assert np.array_equal(ts.pseudocubic_matrix(9), ts.pseudocubic_matrix(2))


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('ok  %s' % fn.__name__)
    print('\n%d checks passed; %s'
          % (len(fns), ts.verify_pseudocubic_transforms()))


if __name__ == '__main__':
    _run()
