"""
Category-B Paradigm 1 — Global-BM (EagerHungarianEngine):
Sliding-window exact bipartite 1-to-1 assignment (Kuhn 1955; Jonker-Volgenant 1987)
materializing deterministic Top-1 trajectory links over shared CSLS+kinematic candidate scores.
"""

from typing import List, Optional, Tuple
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from portolan.candidate_graph import SpatioTemporalCandidateGraph
from portolan.query_eval import BaseDBEngine, links_to_trajectories


class EagerHungarianEngine(BaseDBEngine):
    """
    1. Eager-Hungarian (Global-BM / Global-BM-TW):
    Sliding-window exact sparse bipartite assignment materializing deterministic Top-1 links
    with a lookahead overlap buffer so boundary targets compete fairly before commitment.
    Supports both fixed tuple-count windows (`window_size=550, overlap=380`) and fixed
    temporal windows (`window_sec`, `overlap_sec`) without artificial successor index caps.
    """

    name = "Global-BM"

    def __init__(
        self,
        sim_threshold: float = 0.20,
        window_size: int = 550,
        overlap: int = 380,
        window_sec: Optional[float] = None,
        overlap_sec: float = 52.0,
    ):
        self.sim_threshold = sim_threshold
        self.window_size = window_size
        self.overlap = overlap
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        if window_sec is not None:
            self.name = "Global-BM (Time-W)"

    def recover_trajectories(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], np.ndarray, float]:
        """Execute sliding-window sparse bipartite 1-to-1 assignment over shared candidate edges."""
        t0 = time.perf_counter()
        n = cgraph.n_snaps
        next_ptr = np.full(n, -1, dtype=np.int32)
        in_matched = np.zeros(n, dtype=bool)
        snap_time = cgraph.dataset["snap_time"]

        windows: List[Tuple[int, int, int]] = []
        if self.window_sec is not None and n > 0:
            t_cur = float(snap_time[0])
            t_max = float(snap_time[-1])
            s_idx = 0
            while s_idx < n:
                t_commit = t_cur + float(self.window_sec)
                commit_end = max(s_idx + 1, int(np.searchsorted(snap_time, t_commit, side="right")))
                u_end = min(n, int(np.searchsorted(snap_time, t_commit + self.overlap_sec, side="right")))
                windows.append((s_idx, min(n, commit_end), max(min(n, commit_end), u_end)))
                s_idx = min(n, commit_end)
                if s_idx < n:
                    t_cur = max(t_commit, float(snap_time[s_idx]))
                if t_cur > t_max + 1.0 and s_idx >= n:
                    break
        else:
            step = self.window_size
            for s_idx in range(0, n, step):
                commit_end = min(n, s_idx + step)
                u_end = min(n, commit_end + self.overlap)
                windows.append((s_idx, commit_end, u_end))

        offset = 10.0
        dummy_weight = offset + float(self.sim_threshold)
        out_nbrs = cgraph.out_nbrs
        out_fused = cgraph.out_fused

        for s_idx, commit_end, u_end in windows:
            u_nodes = [
                u for u in range(s_idx, u_end)
                if next_ptr[u] == -1 and (active_mask is None or active_mask[u]) and len(out_nbrs[u]) > 0
            ]
            if not u_nodes:
                continue

            # Collect all admissible candidate targets v without any artificial v < max_v_idx truncation
            v_chunks = []
            row_chunks = []
            sc_chunks = []
            for r_idx, u in enumerate(u_nodes):
                nbrs = out_nbrs[u]
                sc = out_fused[u]
                valid = (sc >= self.sim_threshold) & (~in_matched[nbrs])
                if active_mask is not None:
                    valid &= active_mask[nbrs]
                if np.any(valid):
                    v_sel = nbrs[valid]
                    s_sel = sc[valid]
                    v_chunks.append(v_sel)
                    row_chunks.append(np.full(len(v_sel), r_idx, dtype=np.int32))
                    sc_chunks.append(s_sel)

            if not v_chunks:
                continue

            all_v_raw = np.concatenate(v_chunks)
            all_r = np.concatenate(row_chunks)
            all_sc = np.concatenate(sc_chunks)
            v_list, all_c = np.unique(all_v_raw, return_inverse=True)
            n_rows = len(u_nodes)
            n_cols = len(v_list)

            ar = np.arange(n_rows, dtype=np.int32)
            rows = np.concatenate([all_r, ar])
            cols = np.concatenate([all_c.astype(np.int32), n_cols + ar])
            weights = np.concatenate([offset + all_sc.astype(np.float64), np.full(n_rows, dummy_weight, dtype=np.float64)])

            biadj = csr_matrix((weights, (rows, cols)), shape=(n_rows, n_cols + n_rows))
            r_ind, c_ind = min_weight_full_bipartite_matching(biadj, maximize=True)
            for r, c in zip(r_ind, c_ind):
                r_i = int(r)
                c_i = int(c)
                u = u_nodes[r_i]
                if u < commit_end and c_i < n_cols:
                    v = int(v_list[c_i])
                    if not in_matched[v]:
                        next_ptr[u] = v
                        in_matched[v] = True

        trajs = links_to_trajectories(next_ptr, active_mask=active_mask)
        elapsed = time.perf_counter() - t0
        return trajs, next_ptr, elapsed
