"""Flag complex construction from networkx graphs.

A flag complex (clique complex) lifts a graph to a simplicial complex by
filling in every clique as a simplex: a triangle of three mutually adjacent
nodes becomes a 2-simplex, a tetrahedron of four becomes a 3-simplex, and so
on.

This is required for Murmur's attention-graph arm. Persistent homology at
dimension k >= 2 needs k-dimensional cells, which a raw attention graph
(being a 1-skeleton) does not provide by construction. The flag lift is the
bridge that makes H_2+ testable on attention-derived structure.

Output format: a plain list of (simplex, birth_value) tuples. Each simplex
is a sorted tuple of node IDs; birth_value is a float. The persistence
backend (Dionysus, GUDHI, etc.) decides how to consume this.
"""

from itertools import combinations
from typing import Hashable, Literal

import networkx as nx


Simplex = tuple[Hashable, ...]
SimplexWithBirth = tuple[Simplex, float]


def build_flag_complex(
    graph: nx.Graph,
    max_dim: int,
    weight_attr: str = "weight",
    filtration_mode: Literal["max", "min"] = "max",
    vertex_birth: float = 0.0,
    default_edge_weight: float = 1.0,
) -> list[SimplexWithBirth]:
    """Build a flag complex from an undirected graph up to max_dim.

    Args:
        graph: An undirected, non-multi networkx Graph. Nodes must be
            hashable and pairwise comparable (e.g., ints or strings) so that
            simplex tuples can be canonically sorted.
        max_dim: Maximum simplex dimension to enumerate. 0 = vertices only,
            1 = vertices + edges, 2 = + triangles, and so on. Must be >= 0.
        weight_attr: Edge attribute holding the scalar weight. Edges lacking
            the attribute use default_edge_weight.
        filtration_mode: Aggregation across constituent edge weights when
            assigning a birth value to a k-simplex (k >= 1).
            "max" = Vietoris-Rips convention: simplex is born when its last
            (heaviest) edge appears.
            "min" = threshold-sweep convention for similarity weights:
            simplex is born when its weakest edge first clears the threshold.
        vertex_birth: Birth value assigned to every 0-simplex.
        default_edge_weight: Weight assumed for edges missing weight_attr.

    Returns:
        List of (simplex, birth_value) tuples. Each simplex is a sorted
        tuple of node IDs. Ordering of the list is not guaranteed; the
        caller sorts by birth value if a filtration sequence is needed.

    Raises:
        TypeError: If graph is directed or a multigraph.
        ValueError: If max_dim < 0 or filtration_mode is invalid.
    """
    if graph.is_directed():
        raise TypeError("graph must be undirected")
    if graph.is_multigraph():
        raise TypeError("graph must not be a multigraph")
    if max_dim < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")
    if filtration_mode not in ("max", "min"):
        raise ValueError(
            f"filtration_mode must be 'max' or 'min', got {filtration_mode!r}"
        )

    result: list[SimplexWithBirth] = []

    # 0-simplices: one per node.
    for node in graph.nodes():
        result.append(((node,), vertex_birth))

    if max_dim == 0:
        return result

    def edge_weight(u: Hashable, v: Hashable) -> float:
        data = graph.get_edge_data(u, v) or {}
        return float(data.get(weight_attr, default_edge_weight))

    # 1-simplices: one per edge.
    for u, v in graph.edges():
        simplex = tuple(sorted((u, v)))
        result.append((simplex, edge_weight(u, v)))

    if max_dim == 1:
        return result

    # k-simplices for k >= 2: every (k+1)-clique in the graph.
    # networkx.enumerate_all_cliques yields cliques in ascending size order,
    # so we can break out of the loop once we exceed max_dim.
    agg = max if filtration_mode == "max" else min
    for clique in nx.enumerate_all_cliques(graph):
        size = len(clique)
        k = size - 1
        if k < 2:
            continue  # vertices and edges already emitted above
        if k > max_dim:
            break
        simplex = tuple(sorted(clique))
        edge_weights = [edge_weight(u, v) for u, v in combinations(simplex, 2)]
        birth = agg(edge_weights)
        result.append((simplex, birth))

    return result


# ----- Sanity check (run: python -m src.topology.flag_complex) -----


def _sanity_check() -> None:
    """Hand-verifiable cases. Prints PASS/FAIL per case; exits 1 on any failure."""
    import sys

    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        marker = "PASS" if condition else "FAIL"
        tail = f": {detail}" if (detail and not condition) else ""
        print(f"  [{marker}] {name}{tail}")
        if not condition:
            failures.append(name)

    # Case 1: empty graph.
    sc = build_flag_complex(nx.Graph(), max_dim=3)
    check("empty graph yields empty complex", sc == [])

    # Case 2: two isolated nodes, max_dim=0.
    g = nx.Graph()
    g.add_nodes_from([0, 1])
    sc = build_flag_complex(g, max_dim=0)
    check("two isolated nodes", sorted(sc) == [((0,), 0.0), ((1,), 0.0)])

    # Case 3: K_3, max_dim=2. Expect 3 vertices, 3 edges, 1 triangle.
    g = nx.complete_graph(3)
    sc = build_flag_complex(g, max_dim=2)
    dims = sorted(len(s) - 1 for s, _ in sc)
    check("K_3 yields 3 + 3 + 1 simplices", dims == [0, 0, 0, 1, 1, 1, 2])

    # Case 4: K_4, max_dim=3. Expect 4 + 6 + 4 + 1 = 15 simplices.
    g = nx.complete_graph(4)
    sc = build_flag_complex(g, max_dim=3)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for s, _ in sc:
        counts[len(s) - 1] += 1
    check("K_4 dimension counts (4,6,4,1)", counts == {0: 4, 1: 6, 2: 4, 3: 1})

    # Case 5: K_4 truncated at max_dim=2 has no 3-simplex.
    sc = build_flag_complex(g, max_dim=2)
    counts = {0: 0, 1: 0, 2: 0}
    for s, _ in sc:
        counts[len(s) - 1] += 1
    check("K_4 truncated at dim=2", counts == {0: 4, 1: 6, 2: 4})

    # Case 6: open path 0-1-2 has no triangle.
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])
    sc = build_flag_complex(g, max_dim=2)
    has_triangle = any(len(s) == 3 for s, _ in sc)
    check("open path has no triangle", not has_triangle)

    # Case 7: weighted triangle, mode=max. Triangle birth = max(0.3, 0.5, 0.7) = 0.7.
    g = nx.Graph()
    g.add_edge(0, 1, weight=0.3)
    g.add_edge(1, 2, weight=0.5)
    g.add_edge(0, 2, weight=0.7)
    sc = build_flag_complex(g, max_dim=2, filtration_mode="max")
    tri = [(s, b) for s, b in sc if len(s) == 3]
    check(
        "weighted triangle mode=max birth = 0.7",
        len(tri) == 1 and abs(tri[0][1] - 0.7) < 1e-9,
        detail=f"got {tri[0][1] if tri else 'no triangle'}",
    )

    # Case 8: same triangle, mode=min. Triangle birth = min = 0.3.
    sc = build_flag_complex(g, max_dim=2, filtration_mode="min")
    tri = [(s, b) for s, b in sc if len(s) == 3]
    check(
        "weighted triangle mode=min birth = 0.3",
        len(tri) == 1 and abs(tri[0][1] - 0.3) < 1e-9,
        detail=f"got {tri[0][1] if tri else 'no triangle'}",
    )

    # Case 9: validation errors.
    try:
        build_flag_complex(nx.DiGraph(), max_dim=2)
        check("DiGraph rejected", False, detail="no error raised")
    except TypeError:
        check("DiGraph rejected with TypeError", True)
    try:
        build_flag_complex(nx.Graph(), max_dim=-1)
        check("negative max_dim rejected", False, detail="no error raised")
    except ValueError:
        check("negative max_dim rejected with ValueError", True)

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("flag_complex sanity check\n")
    _sanity_check()
