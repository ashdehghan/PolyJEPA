"""Precomputed neighborhood and community structure for pair construction.

A ``GraphIndex`` is built once per graph and shared across the probes. It holds
the 1-hop adjacency (as CPU ``LongTensor`` lists), computes 2-hop rings on
demand, and lazily detects communities (NetworkX Louvain) for the
community-sibling probe. Sibling sampling is delegated to a caller-supplied
``torch.Generator``, and community detection uses the ``seed`` passed at
construction, so results are reproducible and the partition varies with the seed.
"""

from __future__ import annotations

import torch


class GraphIndex:
    def __init__(
        self, edge_index: torch.Tensor, num_nodes: int, seed: int | None = 0
    ) -> None:
        self.num_nodes = num_nodes
        # Seed for Louvain community detection; threaded from the engine seed so
        # the partition is deterministic per seed but varies across seeds.
        self._seed = seed
        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()
        adj: list[set[int]] = [set() for _ in range(num_nodes)]
        for s, d in zip(src, dst, strict=True):
            if s != d:
                adj[s].add(d)
                adj[d].add(s)  # treat as undirected for neighborhood structure
        self._adj_sets = adj
        self._neighbors: list[torch.Tensor] = [
            torch.tensor(sorted(s), dtype=torch.long) for s in adj
        ]
        self._ring2: list[torch.Tensor] | None = None
        self._communities: list[torch.Tensor] | None = None
        self._node_community: torch.Tensor | None = None

    def neighbors(self, v: int) -> torch.Tensor:
        """Sorted 1-hop neighbors of ``v`` (excluding ``v``)."""
        return self._neighbors[v]

    @property
    def neighbor_lists(self) -> list[torch.Tensor]:
        return self._neighbors

    def ring2(self, v: int) -> torch.Tensor:
        """Nodes exactly 2 hops from ``v`` (within 2 hops, minus 1-hop, minus v)."""
        if self._ring2 is None:
            self._build_ring2()
        assert self._ring2 is not None
        return self._ring2[v]

    def _build_ring2(self) -> None:
        out: list[torch.Tensor] = []
        for v in range(self.num_nodes):
            one = self._adj_sets[v]
            two: set[int] = set()
            for u in one:
                two.update(self._adj_sets[u])
            two.discard(v)
            two.difference_update(one)
            out.append(torch.tensor(sorted(two), dtype=torch.long))
        self._ring2 = out

    def _build_communities(self) -> None:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(self.num_nodes))
        for v, nbrs in enumerate(self._adj_sets):
            for u in nbrs:
                if u > v:
                    g.add_edge(v, u)
        # Deterministic given the seed argument; Louvain is randomized otherwise.
        comms = nx.community.louvain_communities(
            g, seed=self._seed if self._seed is not None else 0
        )
        node_comm = torch.full((self.num_nodes,), -1, dtype=torch.long)
        comm_tensors: list[torch.Tensor] = []
        for cid, members in enumerate(comms):
            members_t = torch.tensor(sorted(members), dtype=torch.long)
            comm_tensors.append(members_t)
            node_comm[members_t] = cid
        self._communities = comm_tensors
        self._node_community = node_comm

    def community_of(self, v: int) -> torch.Tensor:
        """Members of ``v``'s community (including ``v``)."""
        if self._communities is None:
            self._build_communities()
        assert self._node_community is not None and self._communities is not None
        cid = int(self._node_community[v])
        return self._communities[cid]

    def sample_sibling(self, v: int, generator: torch.Generator) -> int | None:
        """A non-neighbor member of ``v``'s community, or ``None`` if none exist."""
        members = self.community_of(v)
        nbrs = set(self._adj_sets[v].union({v}))
        candidates = [int(u) for u in members.tolist() if int(u) not in nbrs]
        if not candidates:
            return None
        idx = int(torch.randint(len(candidates), (1,), generator=generator))
        return candidates[idx]
