"""
Category-B Paradigm 3 — Online-Clust (StreamGreedyClustEngine):
Topology-free one-pass streaming incremental leader-follower feature clustering
(Focus OSDI 2018; Panorama VLDB 2019).
"""

from typing import List, Optional, Tuple
import time
import numpy as np

from portolan.candidate_graph import SpatioTemporalCandidateGraph
from portolan.query_eval import BaseDBEngine


class StreamGreedyClustEngine(BaseDBEngine):
    """
    3. Stream-GreedyClust (Online-Clust):
    Topology-free one-pass streaming incremental leader-follower feature clustering:
    assigns each arriving snapshot to the highest-similarity active cluster centroid within
    a bounded temporal window using CSLS-adjusted cosine similarity and recency EMA updates
    (using observable camera IDs to prevent trivial self-camera merges, with 0% road topology).
    """

    name = "Online-Clust"

    def __init__(
        self,
        sim_threshold: float = 0.34,
        max_clusters: int = 2400,
        window_sec: float = 160.0,
        ema_alpha: float = 0.35,
    ):
        self.sim_threshold = sim_threshold
        self.max_clusters = max_clusters
        self.window_sec = window_sec
        self.ema_alpha = ema_alpha

    def recover_trajectories(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], np.ndarray, float]:
        """Perform single-pass topology-free leader-follower centroid clustering."""
        t0 = time.perf_counter()
        n = cgraph.n_snaps
        emb_np = cgraph.dataset["embeddings"].numpy()
        snap_time = cgraph.dataset["snap_time"]
        snap_cam = cgraph.dataset["snap_cam"]
        loc_rk = cgraph.local_knn_mean_sim
        hub_susp = cgraph.hub_suspicion
        mean_rk = float(cgraph.mean_r_k)
        d = emb_np.shape[1]
        next_ptr = np.full(n, -1, dtype=np.int32)

        cap = 6000 if n > 60000 else self.max_clusters
        clust_vecs = np.zeros((cap, d), dtype=np.float32)
        clust_rk = np.zeros(cap, dtype=np.float32)
        clust_last_snap = np.full(cap, -1, dtype=np.int32)
        clust_last_cam = np.full(cap, -1, dtype=np.int32)
        clust_last_time = np.full(cap, -1e9, dtype=np.float64)
        clust_members: List[List[int]] = []
        clust_global_id = np.full(cap, -1, dtype=np.int32)
        active_cnt = 0

        for v in range(n):
            if active_mask is not None and not active_mask[v]:
                continue
            tv = float(snap_time[v])
            cv = int(snap_cam[v])
            ev = emb_np[v]
            rk_v = float(loc_rk[v])
            hub_pen_v = 0.008 * max(0.0, float(hub_susp[v]) - 6.0)

            best_slot = -1
            if active_cnt > 0:
                dt = tv - clust_last_time[:active_cnt]
                time_cam_ok = (dt >= 0.2) & (dt <= self.window_sec) & (clust_last_cam[:active_cnt] != cv)
                if np.any(time_cam_ok):
                    cand_slots = np.nonzero(time_cam_ok)[0]
                    raw_sims = clust_vecs[cand_slots] @ ev
                    avg_rk = 0.5 * (clust_rk[cand_slots] + rk_v)
                    csls_pen = 0.18 * np.maximum(0.0, avg_rk - mean_rk)
                    adj_sims = raw_sims - csls_pen - hub_pen_v
                    best_rel = int(np.argmax(adj_sims))
                    if float(adj_sims[best_rel]) >= self.sim_threshold:
                        best_slot = int(cand_slots[best_rel])

            if best_slot >= 0:
                gid = int(clust_global_id[best_slot])
                prev_s = int(clust_last_snap[best_slot])
                next_ptr[prev_s] = v
                clust_members[gid].append(v)
                a = self.ema_alpha
                new_c = (1.0 - a) * clust_vecs[best_slot] + a * ev
                norm_c = float(np.linalg.norm(new_c))
                if norm_c > 1e-8:
                    clust_vecs[best_slot] = new_c / norm_c
                clust_rk[best_slot] = (1.0 - a) * clust_rk[best_slot] + a * rk_v
                clust_last_snap[best_slot] = v
                clust_last_cam[best_slot] = cv
                clust_last_time[best_slot] = tv
            else:
                gid = len(clust_members)
                clust_members.append([v])
                if active_cnt < cap:
                    slot = active_cnt
                    active_cnt += 1
                else:
                    slot = int(np.argmin(clust_last_time))
                clust_vecs[slot] = ev
                clust_rk[slot] = rk_v
                clust_last_snap[slot] = v
                clust_last_cam[slot] = cv
                clust_last_time[slot] = tv
                clust_global_id[slot] = gid

        trajs = [m for m in clust_members if len(m) > 0]
        elapsed = time.perf_counter() - t0
        return trajs, next_ptr, elapsed
