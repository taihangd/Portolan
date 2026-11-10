"""
Category-B Paradigm 4 — Vec-Filter (VecDBFilteredANNEngine):
Scalar spatio-temporal range pre-filtering followed by Top-1 CSLS-scaled vector similarity search
(DiskANN NeurIPS 2019; HNSW TPAMI 2020; Milvus SIGMOD 2021).
"""

from typing import List, Optional, Tuple
import time
import numpy as np

from portolan.candidate_graph import SpatioTemporalCandidateGraph
from portolan.query_eval import BaseDBEngine, links_to_trajectories


class VecDBFilteredANNEngine(BaseDBEngine):
    """
    4. VecDB-Filtered-ANN (Vec-Filter):
    Scalar spatio-temporal road-gate pre-filtering followed by Top-1 CSLS-adjusted vector
    similarity search without global combinatorial assignment or travel-time likelihood.
    """

    name = "Vec-Filter"

    def __init__(self, sim_threshold: float = 0.24):
        self.sim_threshold = sim_threshold

    def recover_trajectories(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], np.ndarray, float]:
        """Link each tuple to its Top-1 road-filtered nearest CSLS visual neighbor."""
        t0 = time.perf_counter()
        n = cgraph.n_snaps
        next_ptr = np.full(n, -1, dtype=np.int32)
        in_claimed = np.zeros(n, dtype=bool)
        in_best_sim = np.full(n, -10.0, dtype=np.float32)
        in_best_u = np.full(n, -1, dtype=np.int32)

        for u in range(n):
            if active_mask is not None and not active_mask[u]:
                continue
            nbrs = cgraph.out_nbrs[u]
            if len(nbrs) == 0:
                continue
            vis = cgraph.out_csls_vis[u]
            if active_mask is not None:
                valid = active_mask[nbrs]
                if not np.any(valid):
                    continue
                nbrs = nbrs[valid]
                vis = vis[valid]
            best_idx = int(np.argmax(vis))
            best_s = float(vis[best_idx])
            if best_s >= self.sim_threshold:
                v = int(nbrs[best_idx])
                if not in_claimed[v]:
                    in_claimed[v] = True
                    in_best_sim[v] = best_s
                    in_best_u[v] = u
                    next_ptr[u] = v
                elif best_s > in_best_sim[v]:
                    old_u = int(in_best_u[v])
                    next_ptr[old_u] = -1
                    in_best_sim[v] = best_s
                    in_best_u[v] = u
                    next_ptr[u] = v

        trajs = links_to_trajectories(next_ptr, active_mask=active_mask)
        elapsed = time.perf_counter() - t0
        return trajs, next_ptr, elapsed
