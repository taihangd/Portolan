"""
Category-B Paradigm 5 — Prob-Path (TopoMarkovMAPEngine):
Topology-guided Markov probabilistic DAG inference combining CSLS-scaled directional similarity
with road-network log-normal travel-time log-likelihood via backward dynamic programming
and greedy node-disjoint path extraction (adapting MMVC MM 2022; RNTrajRec ICDE 2023; VIVID PACMMOD 2025).
"""

from typing import List, Optional, Tuple
import time
import numpy as np

from portolan.candidate_graph import SpatioTemporalCandidateGraph
from portolan.query_eval import BaseDBEngine, links_to_trajectories


class TopoMarkovMAPEngine(BaseDBEngine):
    """
    5. Topo-Markov-MAP (Prob-Path):
    Topology-guided Markov probabilistic DAG inference: computes exact backward DP MAP path
    potentials combining shared CSLS visual similarity and road-network travel-time log-likelihood,
    then extracts node-disjoint trajectories greedily in descending MAP score order.
    """

    name = "Prob-Path"

    def __init__(self, kin_weight: float = 0.085, base_prior: float = 0.20):
        self.kin_weight = kin_weight
        self.base_prior = base_prior

    def recover_trajectories(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], np.ndarray, float]:
        """Compute backward Markov DP values and extract greedy node-disjoint paths."""
        t0 = time.perf_counter()
        n = cgraph.n_snaps
        next_ptr = np.full(n, -1, dtype=np.int32)
        claimed = np.zeros(n, dtype=bool)

        dp_val = np.zeros(n, dtype=np.float32)
        best_choice = np.full(n, -1, dtype=np.int32)
        out_nbrs = cgraph.out_nbrs
        out_csls = cgraph.out_csls_vis
        out_kin = cgraph.out_kin

        for u in range(n - 1, -1, -1):
            if active_mask is not None and not active_mask[u]:
                continue
            nbrs = out_nbrs[u]
            if len(nbrs) == 0:
                continue
            vis = out_csls[u]
            kin = out_kin[u]
            if active_mask is not None:
                valid = active_mask[nbrs]
                if not np.any(valid):
                    continue
                nbrs = nbrs[valid]
                vis = vis[valid]
                kin = kin[valid]
            edge_ll = vis + self.kin_weight * kin - self.base_prior
            pos_mask = edge_ll > 0.0
            if np.any(pos_mask):
                tot = np.where(pos_mask, edge_ll + dp_val[nbrs], -1.0)
                best_idx = int(np.argmax(tot))
                if float(tot[best_idx]) > 0.0:
                    dp_val[u] = float(tot[best_idx])
                    best_choice[u] = int(nbrs[best_idx])

        roots = np.argsort(-dp_val)
        for u in roots:
            u_int = int(u)
            if active_mask is not None and not active_mask[u_int]:
                continue
            if claimed[u_int] or dp_val[u_int] <= 0.0:
                continue
            cur = u_int
            while cur >= 0 and not claimed[cur]:
                claimed[cur] = True
                nxt = int(best_choice[cur])
                if nxt >= 0 and not claimed[nxt]:
                    next_ptr[cur] = nxt
                    cur = nxt
                else:
                    nbrs_c = out_nbrs[cur]
                    if len(nbrs_c) == 0:
                        break
                    avail = ~claimed[nbrs_c]
                    if active_mask is not None:
                        avail &= active_mask[nbrs_c]
                    if not np.any(avail):
                        break
                    sc_c = out_csls[cur][avail] + self.kin_weight * out_kin[cur][avail] - self.base_prior
                    best_fb = int(np.argmax(sc_c))
                    if float(sc_c[best_fb]) > 0.0:
                        fb_v = int(nbrs_c[avail][best_fb])
                        next_ptr[cur] = fb_v
                        cur = fb_v
                    else:
                        break

        trajs = links_to_trajectories(next_ptr, active_mask=active_mask)
        elapsed = time.perf_counter() - t0
        return trajs, next_ptr, elapsed
