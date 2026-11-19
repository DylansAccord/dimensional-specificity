"""Zigzag persistent homology over a sequence of simplicial complexes.

Murmur's attention-graph arm computes a flag complex per transformer layer
and asks which topological features persist across the layer sequence.
Layer-to-layer complex changes are not monotonic (edges and cliques appear,
disappear, reappear), so standard persistence is the wrong tool. Zigzag
persistence (Carlsson and de Silva, 2010) is the principled answer.

Backend: Dionysus 2.x. Input: a sequence of simplicial complexes in the
(simplex, birth_value) list format produced by flag_complex.py. Output: a
dict mapping homology dimension to a list of (birth, death) intervals, one
intervals list per dimension (the persistence diagram).

Convention: each complex K_i in the sequence sits at integer time i. A
simplex's birth/death events occur at integer times corresponding to which
K_i it appears in or vanishes from. The per-simplex birth_value field on
input is currently ignored. Adding intra-complex filtration ordering on top
of the layer-level zigzag is a non-breaking extension if needed later.
"""

from collections import defaultdict
from typing import Hashable


Simplex = tuple[Hashable, ...]
SimplexWithBirth = tuple[Simplex, float]
ComplexSequence = list[list[SimplexWithBirth]]
PersistenceDiagram = dict[int, list[tuple[float, float]]]


def compute_zigzag_persistence(
    complex_sequence: ComplexSequence,
    max_dim: int,
) -> PersistenceDiagram:
    """Compute zigzag persistent homology over a sequence of complexes.

    Args:
        complex_sequence: List of complexes. Each complex is a list of
            (simplex, birth_value) tuples in the format produced by
            flag_complex.build_flag_complex. Each simplex must be a sorted
            tuple of comparable hashable node IDs. The birth_value field is
            accepted but not used.
        max_dim: Maximum homology dimension to return. Diagrams for
            dimensions 0 through max_dim inclusive are returned.

    Returns:
        Dict mapping dimension k in {0, ..., max_dim} to a list of
        (birth, death) tuples. death may be float('inf') for features
        still alive at the end of the sequence. Dimensions with no features
        map to an empty list.

    Raises:
        ValueError: If complex_sequence is empty or max_dim < 0.
    """
    if not complex_sequence:
        raise ValueError("complex_sequence must contain at least one complex")
    if max_dim < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")

    # dionysus is imported lazily: it is only needed for the attention-graph
    # (zigzag) arm, it compiles from source (see docs/install.md), and the
    # per-token confirmatory arm must not depend on it being installed.
    try:
        import dionysus as d
    except ImportError as e:
        raise ImportError(
            "dionysus is required for the attention-graph (zigzag) arm but is not "
            "installed. See docs/install.md. The per-token arm does not need it."
        ) from e

    n = len(complex_sequence)

    # Build presence map: simplex -> list of booleans, one per timestep.
    presence: dict[Simplex, list[bool]] = defaultdict(lambda: [False] * n)
    for i, complex_i in enumerate(complex_sequence):
        for simplex, _birth in complex_i:
            presence[simplex][i] = True

    # Convert each presence vector to Dionysus's alternating times list.
    # Convention: K_i sits at integer time i. times list alternates
    # [appear_t0, disappear_t1, appear_t2, ...]. Odd length = alive at end.
    def presence_to_times(p: list[bool]) -> list[float]:
        times: list[float] = []
        currently_present = False
        for i, present in enumerate(p):
            if present and not currently_present:
                times.append(float(i))
                currently_present = True
            elif not present and currently_present:
                times.append(float(i))
                currently_present = False
        return times

    # Build a Dionysus filtration containing every simplex that ever appears.
    # Order by (dimension, simplex) so faces precede cofaces, which Dionysus
    # requires for a valid zigzag input.
    all_simplices = sorted(presence.keys(), key=lambda s: (len(s), s))

    f = d.Filtration()
    for simplex in all_simplices:
        # The numeric data attached to d.Simplex doesn't affect zigzag; it's
        # metadata. Pass 0.0 as placeholder.
        f.append(d.Simplex(list(simplex), 0.0))

    times = [presence_to_times(presence[s]) for s in all_simplices]

    # Compute zigzag persistence. dgms is a list indexed by dimension.
    _zz, dgms, _cells = d.zigzag_homology_persistence(f, times)

    # Parse output into our standard format.
    result: PersistenceDiagram = {k: [] for k in range(max_dim + 1)}
    for dim, dgm in enumerate(dgms):
        if dim > max_dim:
            break
        for pt in dgm:
            birth = float(pt.birth)
            death_raw = pt.death
            # Dionysus uses float('inf') for essential classes.
            death = float(death_raw) if death_raw != float("inf") else float("inf")
            result[dim].append((birth, death))

    return result


# ----- Sanity check (run: python -m src.topology.zigzag) -----


def _sanity_check() -> None:
    """Hand-derivable cases. Prints PASS/FAIL per case; exits 1 on any failure."""
    import sys

    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        marker = "PASS" if condition else "FAIL"
        tail = f": {detail}" if (detail and not condition) else ""
        print(f"  [{marker}] {name}{tail}")
        if not condition:
            failures.append(name)

    # Case 1: single complex with two isolated vertices.
    # Expect 2 H_0 features, both born at 0, both alive at end.
    K0 = [((0,), 0.0), ((1,), 0.0)]
    dgms = compute_zigzag_persistence([K0], max_dim=1)
    h0 = dgms[0]
    check("single complex, 2 isolated vertices: |H_0| = 2", len(h0) == 2)
    check(
        "single complex: both H_0 born at 0 and alive at end",
        len(h0) == 2 and all(b == 0.0 and d == float("inf") for b, d in h0),
        detail=f"got {h0}",
    )

    # Case 2: K_0 = two isolated vertices, K_1 = same plus connecting edge.
    # Expect: 2 H_0 born at 0, one dies at 1 (merge), other alive at end.
    K0 = [((0,), 0.0), ((1,), 0.0)]
    K1 = [((0,), 0.0), ((1,), 0.0), ((0, 1), 1.0)]
    dgms = compute_zigzag_persistence([K0, K1], max_dim=1)
    h0 = dgms[0]
    check("merge sequence: |H_0| = 2", len(h0) == 2)
    deaths = sorted(d for _, d in h0)
    check(
        "merge sequence: one H_0 dies at 1, one alive at end",
        len(deaths) == 2
        and abs(deaths[0] - 1.0) < 1e-9
        and deaths[1] == float("inf"),
        detail=f"deaths = {deaths}",
    )

    # Case 3: open triangle (3 vertices, 3 edges, no 2-simplex).
    # Expect one persistent H_1 feature.
    K0 = [
        ((0,), 0.0), ((1,), 0.0), ((2,), 0.0),
        ((0, 1), 1.0), ((1, 2), 1.0), ((0, 2), 1.0),
    ]
    dgms = compute_zigzag_persistence([K0], max_dim=1)
    h1 = dgms[1]
    check("open triangle: |H_1| = 1", len(h1) == 1)
    check(
        "open triangle: H_1 born at 0, alive at end",
        len(h1) == 1 and h1[0][0] == 0.0 and h1[0][1] == float("inf"),
        detail=f"got {h1}",
    )

    # Case 4: open triangle K_0, filled triangle K_1 (2-simplex added).
    # Expect H_1 feature born at 0, dies at 1.
    K1 = K0 + [((0, 1, 2), 1.0)]
    dgms = compute_zigzag_persistence([K0, K1], max_dim=1)
    h1 = dgms[1]
    check("triangle fills at t=1: |H_1| = 1", len(h1) == 1)
    check(
        "triangle fills at t=1: H_1 born at 0, dies at 1",
        len(h1) == 1
        and h1[0][0] == 0.0
        and abs(h1[0][1] - 1.0) < 1e-9,
        detail=f"got {h1}",
    )

    # Case 5: zigzag triangle. K_0 = open triangle, K_1 = vertices only,
    # K_2 = open triangle again. Expect two H_1 intervals: [0, 1) and [2, inf).
    K0 = [
        ((0,), 0.0), ((1,), 0.0), ((2,), 0.0),
        ((0, 1), 1.0), ((1, 2), 1.0), ((0, 2), 1.0),
    ]
    K1_vertices_only = [((0,), 0.0), ((1,), 0.0), ((2,), 0.0)]
    K2 = K0
    dgms = compute_zigzag_persistence([K0, K1_vertices_only, K2], max_dim=1)
    h1 = dgms[1]
    check("zigzag triangle: |H_1| = 2", len(h1) == 2, detail=f"got {h1}")
    intervals = sorted(h1)
    check(
        "zigzag triangle: intervals = [(0, 1), (2, inf)]",
        len(intervals) == 2
        and abs(intervals[0][0] - 0.0) < 1e-9
        and abs(intervals[0][1] - 1.0) < 1e-9
        and abs(intervals[1][0] - 2.0) < 1e-9
        and intervals[1][1] == float("inf"),
        detail=f"got {intervals}",
    )

    # Case 6: input validation.
    try:
        compute_zigzag_persistence([], max_dim=1)
        check("empty sequence rejected", False, detail="no error raised")
    except ValueError:
        check("empty sequence rejected with ValueError", True)
    try:
        compute_zigzag_persistence([[((0,), 0.0)]], max_dim=-1)
        check("negative max_dim rejected", False, detail="no error raised")
    except ValueError:
        check("negative max_dim rejected with ValueError", True)

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("zigzag sanity check\n")
    _sanity_check()
