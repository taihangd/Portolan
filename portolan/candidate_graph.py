"""
GPU-accelerated Spatio-Temporal Candidate Transition Graph, Shared CSLS+Kinematic Scoring,
and Low-Arrival Temporal Trough Partitioning.
"""

from typing import Dict, List, Set, Tuple
import gc
import time
import numpy as np
import torch

from portolan.profiles_and_topology import RoadCameraTopology


def find_arrival_troughs(
    snap_time: np.ndarray,
    target_block_size: int = 8000,
    min_shard_n: int = 60000,
    bin_sec: float = 10.0,
) -> List[Tuple[int, int]]:
    """
    Partition a chronologically sorted stream into contiguous index blocks `[b_start, b_end)`
    aligned with genuine low-arrival temporal troughs:
      1. When `len(snap_time) <= min_shard_n`, returns a single unsharded block `[(0, n)]`.
      2. Otherwise, bins timestamps into `bin_sec` (default 10.0 s) intervals, smooths arrival
         counts with a 3-bin `[0.25, 0.50, 0.25]` moving average, and near each target step
         (`[0.65 * target_block_size, 1.35 * target_block_size]`) selects the bin boundary
         with minimum smoothed arrival rate.
    Shared identically by `Prov-IVM` and `MCF-Sharded`.
    """
    n = int(len(snap_time))
    if n <= 0:
        return []
    if n <= min_shard_n or n <= int(1.35 * target_block_size):
        return [(0, n)]

    t0 = float(snap_time[0])
    t_end = float(snap_time[-1])
    if t_end - t0 <= bin_sec * 3.0:
        blocks = []
        for s in range(0, n, target_block_size):
            blocks.append((s, min(n, s + target_block_size)))
        return blocks

    bin_edges = np.arange(t0, t_end + bin_sec, bin_sec, dtype=np.float64)
    if len(bin_edges) < 3:
        return [(0, n)]
    counts, _ = np.histogram(snap_time, bins=bin_edges)
    kernel = np.array([0.25, 0.50, 0.25], dtype=np.float64)
    smoothed = np.convolve(counts.astype(np.float64), kernel, mode="same")
    # Boundary k (1 .. len(counts)-1) separates bin k-1 and bin k; arrival density at boundary is 0.5*(s[k-1]+s[k])
    bnd_times = bin_edges[1:-1]
    bnd_indices = np.searchsorted(snap_time, bnd_times, side="left")
    bnd_density = 0.5 * (smoothed[:-1] + smoothed[1:])

    blocks: List[Tuple[int, int]] = []
    cur = 0
    min_step = max(250, int(round(0.65 * target_block_size)))
    max_step = max(min_step + 100, int(round(1.35 * target_block_size)))

    while n - cur > max_step:
        lo_idx = cur + min_step
        hi_idx = min(n - min_step, cur + max_step)
        k_lo = int(np.searchsorted(bnd_indices, lo_idx, side="left"))
        k_hi = int(np.searchsorted(bnd_indices, hi_idx, side="right"))
        picked = -1
        if k_hi > k_lo:
            sub_idx = bnd_indices[k_lo:k_hi]
            sub_den = bnd_density[k_lo:k_hi]
            valid_k = (sub_idx > cur) & (sub_idx < n)
            if np.any(valid_k):
                best_rel = int(np.argmin(np.where(valid_k, sub_den, np.inf)))
                picked = int(sub_idx[best_rel])
        if picked <= cur or picked >= n:
            picked = min(n, cur + target_block_size)
        blocks.append((cur, picked))
        cur = picked

    if cur < n:
        blocks.append((cur, n))
    return blocks


class SpatioTemporalCandidateGraph:
    """
    Constructs time-windowed, topology-pruned candidate transitions between snapshots,
    computing visual cosine similarity, kinematic travel-time log-likelihood, reverse-kNN
    hubness statistics, CSLS local similarity scaling, and unified zero-oracle candidate
    edge scores (`out_csls_vis` and `out_fused`) shared across all downstream engines.
    """

    KIN_WEIGHT: float = 0.055
    CSLS_WEIGHT: float = 0.18
    HUB_PEN_WEIGHT: float = 0.008
    HUB_PEN_OFFSET: float = 6.0

    def __init__(
        self,
        dataset: Dict,
        device: torch.device,
        max_hops: int = 3,
        max_dt_sec: float = 260.0,
        top_k_prov: int = 5,
        use_oracle_clock: bool = False,
    ):
        self.dataset = dataset
        self.device = device
        self.max_hops = max_hops
        self.max_dt_sec = max_dt_sec
        self.top_k_prov = top_k_prov
        self.use_oracle_clock = use_oracle_clock

        self.build_time_sec = 0.0
        self.n_snaps = int(len(dataset["snap_cam"]))

        self.out_nbrs: List[np.ndarray] = []
        self.out_vis: List[np.ndarray] = []
        self.out_kin: List[np.ndarray] = []
        self.out_csls_vis: List[np.ndarray] = []
        self.out_fused: List[np.ndarray] = []
        self.out_hops: List[np.ndarray] = []
        self.out_fft: List[np.ndarray] = []

        self.rev_knn_in_deg = np.zeros(self.n_snaps, dtype=np.int32)
        self.local_knn_mean_sim = np.zeros(self.n_snaps, dtype=np.float32)
        self.hub_suspicion = np.zeros(self.n_snaps, dtype=np.float32)
        self.mean_r_k: float = 0.0

        self._build()

    def _build(self) -> None:
        t0 = time.perf_counter()
        was_gc = gc.isenabled()
        if was_gc:
            gc.disable()
        try:
            self._build_impl(t0)
        finally:
            if was_gc:
                gc.enable()

    def _build_impl(self, t0: float) -> None:
        topo: RoadCameraTopology = self.dataset["topology"]
        eff_hops = (
            max(self.max_hops, 4)
            if topo.n_cam <= 50
            else (min(self.max_hops, 2) if topo.n_cam >= 100 else self.max_hops)
        )
        cap_per_cam = 56 if topo.n_cam >= 100 else 120
        if not topo.reach_min_time[0]:
            topo._build_bounded_reachability(max_hops=eff_hops)

        snap_cam: np.ndarray = self.dataset["snap_cam"]
        snap_time: np.ndarray = (
            self.dataset["snap_true_time"] if self.use_oracle_clock else self.dataset["snap_time"]
        )
        emb_gpu: torch.Tensor = self.dataset["embeddings"].to(self.device)
        n = self.n_snaps

        order_c = np.argsort(snap_cam, kind="mergesort")
        sorted_c = snap_cam[order_c]
        c_bounds = np.searchsorted(sorted_c, np.arange(topo.n_cam + 1))
        cam_snaps_np = [order_c[c_bounds[c] : c_bounds[c + 1]].astype(np.int32) for c in range(topo.n_cam)]
        cam_times_np = [snap_time[cam_snaps_np[c]].astype(np.float64) for c in range(topo.n_cam)]
        del order_c, sorted_c

        src_chunks = []
        dst_chunks = []
        hop_chunks = []
        fft_chunks = []
        dt_chunks = []

        # Fixed nominal clock tolerance (4.5s) unless Oracle clock is enabled; never peeks at spec.sigma_tau
        sigma_tau_nom = 0.5 if self.use_oracle_clock else 4.5
        margin_t = 3.2 * sigma_tau_nom + 8.0

        for c_u in range(topo.n_cam):
            u_ids = cam_snaps_np[c_u]
            if len(u_ids) == 0:
                continue
            u_ts = cam_times_np[c_u]
            reach_dict = topo.reach_min_time[c_u]
            cam_src, cam_dst, cam_hop, cam_fft, cam_dt = [], [], [], [], []

            for c_v, (h_uv, min_fft) in reach_dict.items():
                if h_uv > eff_hops:
                    continue
                v_ids = cam_snaps_np[c_v]
                if len(v_ids) == 0:
                    continue
                v_ts = cam_times_np[c_v]

                t_low_bound = max(0.0, min_fft * 0.22 - margin_t)
                t_high_bound = min(self.max_dt_sec, min_fft * 3.4 + 34.0 * h_uv + margin_t)

                idx_start = np.searchsorted(v_ts, u_ts + t_low_bound, side="left")
                idx_end = np.searchsorted(v_ts, u_ts + t_high_bound, side="right")
                # Strictly enforce chronological index DAG order (v > u) using monotonic v_ids
                idx_gt = np.searchsorted(v_ids, u_ids, side="right")
                idx_start = np.maximum(idx_start, idx_gt)

                counts = idx_end - idx_start
                non_empty = np.nonzero(counts > 0)[0]
                if len(non_empty) == 0:
                    continue

                hop_d = min(6.0, min_fft * 0.35) * h_uv
                fft_mul = min_fft * 1.28 + hop_d
                pair_dst_list, pair_u_list, pair_ut_list, pair_m_list = [], [], [], []
                for ui_local in non_empty:
                    s_pos = int(idx_start[ui_local])
                    e_pos = int(idx_end[ui_local])
                    cand_v = v_ids[s_pos:e_pos]
                    if len(cand_v) > cap_per_cam:
                        exp_t = u_ts[ui_local] + fft_mul
                        dt_err = np.abs(snap_time[cand_v] - exp_t)
                        keep_c = np.argpartition(dt_err, cap_per_cam)[:cap_per_cam]
                        cand_v = cand_v[keep_c]
                    pair_dst_list.append(cand_v)
                    pair_u_list.append(int(u_ids[ui_local]))
                    pair_ut_list.append(float(u_ts[ui_local]))
                    pair_m_list.append(len(cand_v))

                p_dst = np.concatenate(pair_dst_list)
                p_m = np.array(pair_m_list, dtype=np.int32)
                p_u = np.repeat(np.array(pair_u_list, dtype=np.int32), p_m)
                p_ut = np.repeat(np.array(pair_ut_list, dtype=np.float64), p_m)
                tot_p = len(p_dst)
                cam_src.append(p_u)
                cam_dst.append(p_dst)
                cam_hop.append(np.full(tot_p, h_uv, dtype=np.int8))
                cam_fft.append(np.full(tot_p, min_fft, dtype=np.float32))
                cam_dt.append((snap_time[p_dst] - p_ut).astype(np.float32))

            if cam_src:
                src_chunks.append(np.concatenate(cam_src))
                dst_chunks.append(np.concatenate(cam_dst))
                hop_chunks.append(np.concatenate(cam_hop))
                fft_chunks.append(np.concatenate(cam_fft))
                dt_chunks.append(np.concatenate(cam_dt))

        _e_i32 = np.empty(0, dtype=np.int32)
        _e_f32 = np.empty(0, dtype=np.float32)
        _e_i8 = np.empty(0, dtype=np.int8)
        if not src_chunks:
            self.out_nbrs = [_e_i32] * n
            self.out_vis = [_e_f32] * n
            self.out_kin = [_e_f32] * n
            self.out_csls_vis = [_e_f32] * n
            self.out_fused = [_e_f32] * n
            self.out_hops = [_e_i8] * n
            self.out_fft = [_e_f32] * n
            self.build_time_sec = time.perf_counter() - t0
            return

        all_src = np.concatenate(src_chunks)
        del src_chunks
        all_dst = np.concatenate(dst_chunks)
        del dst_chunks
        all_hop = np.concatenate(hop_chunks)
        del hop_chunks
        all_fft = np.concatenate(fft_chunks)
        del fft_chunks
        all_dt = np.concatenate(dt_chunks)
        del dt_chunks

        # Compute visual cosine similarities on GPU in vectorized batches
        total_edges = len(all_src)
        all_vis = np.empty(total_edges, dtype=np.float32)
        edge_batch = 400000
        with torch.no_grad():
            for e_s in range(0, total_edges, edge_batch):
                e_e = min(total_edges, e_s + edge_batch)
                s_t = torch.from_numpy(all_src[e_s:e_e]).long().to(self.device)
                d_t = torch.from_numpy(all_dst[e_s:e_e]).long().to(self.device)
                sim_b = (emb_gpu[s_t] * emb_gpu[d_t]).sum(dim=-1)
                all_vis[e_s:e_e] = sim_b.cpu().numpy()
        del emb_gpu

        # Shifted log-normal travel-time & hop-count kinematic log-likelihood
        hop_delay = np.minimum(6.0, all_fft * 0.35) * all_hop
        expected_mean = all_fft * 1.28 + hop_delay
        safe_dt = np.maximum(0.2, all_dt)
        del all_dt
        log_ratio = np.log((safe_dt + 3.0) / (expected_mean + 3.0))
        del safe_dt, expected_mean, hop_delay
        kin_sigma = 0.38 if self.use_oracle_clock else (0.42 + 0.012 * sigma_tau_nom)
        all_kin = np.clip(
            -0.65 * ((log_ratio / kin_sigma) ** 2) - 1.50 * (all_hop - 1), -10.0, 0.0
        ).astype(np.float32)
        del log_ratio

        sort_idx = np.argsort(all_src, kind="mergesort")
        all_src = all_src[sort_idx]
        all_dst = all_dst[sort_idx]
        all_vis = all_vis[sort_idx]
        all_kin = all_kin[sort_idx]
        all_hop = all_hop[sort_idx]
        all_fft = all_fft[sort_idx]
        del sort_idx

        split_bounds = np.searchsorted(all_src, np.arange(n + 1))
        del all_src

        # Precompute bidirectional raw Top-k mean similarity per node for exact CSLS scaling
        out_mean_k = np.zeros(n, dtype=np.float32)
        in_lists_raw: List[List[float]] = [[] for _ in range(n)]
        k_prov = self.top_k_prov
        non_empty_u = np.nonzero(split_bounds[1:] > split_bounds[:-1])[0]
        for u in non_empty_u:
            a, b = int(split_bounds[u]), int(split_bounds[u + 1])
            vis_u = all_vis[a:b]
            dst_u = all_dst[a:b]
            m_k = min(k_prov, b - a)
            top_idx = np.argpartition(-vis_u, m_k - 1)[:m_k] if (b - a) > m_k else range(b - a)
            out_mean_k[u] = float(np.mean(vis_u[top_idx]))
            for idx_k in top_idx:
                sv = float(vis_u[idx_k])
                if sv > 0.32:
                    tv = int(dst_u[idx_k])
                    self.rev_knn_in_deg[tv] += 1
                    in_lists_raw[tv].append(sv)

        self.local_knn_mean_sim[:] = out_mean_k
        for u in np.nonzero(self.rev_knn_in_deg > 0)[0]:
            arr_in = sorted(in_lists_raw[u], reverse=True)[:k_prov]
            in_m = float(np.mean(arr_in))
            self.local_knn_mean_sim[u] = (
                0.5 * (out_mean_k[u] + in_m) if out_mean_k[u] > 0.0 else in_m
            )

        self.out_nbrs = [_e_i32] * n
        self.out_vis = [_e_f32] * n
        self.out_kin = [_e_f32] * n
        self.out_hops = [_e_i8] * n
        self.out_fft = [_e_f32] * n

        # Retain union of Top-22 visual + Top-20 visual+kinematic + Top-22 1-hop + Top-22 multi-hop kinematic candidates
        keep_mask = np.ones(total_edges, dtype=bool)
        over64_u = non_empty_u[(split_bounds[non_empty_u + 1] - split_bounds[non_empty_u]) > 64]
        for u in over64_u:
            a, b = int(split_bounds[u]), int(split_bounds[u + 1])
            vis_u = all_vis[a:b]
            kin_u = all_kin[a:b]
            hop_u = all_hop[a:b]
            top_v_idx = np.argpartition(-vis_u, 22)[:22]
            comb = vis_u + 0.12 * kin_u
            top_c_idx = np.argpartition(-comb, 20)[:20]
            h1_mask = np.nonzero(hop_u == 1)[0]
            top_k1 = h1_mask[np.argpartition(-kin_u[h1_mask], 22)[:22]] if len(h1_mask) > 22 else h1_mask
            h2_mask = np.nonzero(hop_u > 1)[0]
            top_k2 = h2_mask[np.argpartition(-kin_u[h2_mask], 22)[:22]] if len(h2_mask) > 22 else h2_mask
            keep_u = np.unique(np.concatenate([top_v_idx, top_c_idx, top_k1, top_k2]))
            keep_mask[a:b] = False
            keep_mask[a + keep_u] = True

        if len(over64_u) > 0:
            cs = np.empty(total_edges + 1, dtype=np.int64)
            cs[0] = 0
            np.cumsum(keep_mask, out=cs[1:])
            kept_bounds = cs[split_bounds]
            del cs
            u_counts = (kept_bounds[1:] - kept_bounds[:-1]).astype(np.int32)
            kept_dst = all_dst[keep_mask]
            del all_dst
            kept_vis = all_vis[keep_mask]
            del all_vis
            kept_kin = all_kin[keep_mask]
            del all_kin
            kept_hop = all_hop[keep_mask]
            del all_hop
            kept_fft = all_fft[keep_mask]
            del all_fft
        else:
            kept_bounds = split_bounds
            u_counts = (kept_bounds[1:] - kept_bounds[:-1]).astype(np.int32)
            kept_dst, kept_vis, kept_kin, kept_hop, kept_fft = (
                all_dst,
                all_vis,
                all_kin,
                all_hop,
                all_fft,
            )
        del keep_mask

        # Black-Hole hubness suspicion score combining reverse-kNN in-degree and scale-adaptive similarity concentration
        mean_indeg = max(1.0, float(np.mean(self.rev_knn_in_deg)))
        mean_rk = float(np.mean(self.local_knn_mean_sim)) if n > 0 else 0.35
        self.mean_r_k = mean_rk
        hi_vis_th = min(0.76, max(0.54, mean_rk + 0.26))
        knn_hi_th = min(0.70, max(0.48, mean_rk + 0.20))

        cs_hi = np.empty(len(kept_vis) + 1, dtype=np.int64)
        cs_hi[0] = 0
        np.cumsum(kept_vis > hi_vis_th, out=cs_hi[1:])
        out_hi_arr = (cs_hi[kept_bounds[1:]] - cs_hi[kept_bounds[:-1]]).astype(np.int32)
        del cs_hi
        in_hi_arr = np.zeros(n, dtype=np.int32)
        for u in np.nonzero(self.rev_knn_in_deg > 0)[0]:
            in_hi_arr[u] = sum(1 for s in in_lists_raw[u] if s > hi_vis_th)
        del in_lists_raw

        self.hub_suspicion[:] = (
            1.4 * self.rev_knn_in_deg.astype(np.float64) / mean_indeg
            + 0.55 * (out_hi_arr + in_hi_arr).astype(np.float64)
            + 6.0 * np.maximum(0.0, self.local_knn_mean_sim.astype(np.float64) - knn_hi_th)
        ).astype(np.float32)

        # Shared zero-oracle CSLS + hub-suspicion + kinematic edge scores exposed uniformly to all engines
        kept_src = np.repeat(np.arange(n, dtype=np.int32), u_counts)
        avg_rk = 0.5 * (self.local_knn_mean_sim[kept_src] + self.local_knn_mean_sim[kept_dst])
        del kept_src
        csls_pen = self.CSLS_WEIGHT * np.maximum(0.0, avg_rk - mean_rk)
        del avg_rk
        hub_pen = self.HUB_PEN_WEIGHT * np.maximum(0.0, self.hub_suspicion[kept_dst] - self.HUB_PEN_OFFSET)
        kept_csls = (kept_vis - csls_pen - hub_pen).astype(np.float32)
        del csls_pen, hub_pen
        kept_fused = (kept_csls + self.KIN_WEIGHT * kept_kin).astype(np.float32)

        self.out_csls_vis = [_e_f32] * n
        self.out_fused = [_e_f32] * n
        for u in non_empty_u:
            a, b = int(kept_bounds[u]), int(kept_bounds[u + 1])
            self.out_nbrs[u] = kept_dst[a:b]
            self.out_vis[u] = kept_vis[a:b]
            self.out_kin[u] = kept_kin[a:b]
            self.out_hops[u] = kept_hop[a:b]
            self.out_fft[u] = kept_fft[a:b]
            self.out_csls_vis[u] = kept_csls[a:b]
            self.out_fused[u] = kept_fused[a:b]

        self.build_time_sec = time.perf_counter() - t0

    def clone_with_calibrated_clocks(
        self, calibrated_cams: Set[int]
    ) -> Tuple["SpatioTemporalCandidateGraph", Set[int], Set[int]]:
        """
        Authentic Q4d retroactive camera clock calibration:
        Restores true synchronized timestamps (Delta_tau_c -> 0) for snapshots at calibrated_cams
        and re-weights kinematic travel-time log-likelihoods s_kin(u,v) and shared fused scores
        on all incident candidate edges without deleting any snapshots.
        Returns (calibrated_cgraph, calibrated_snapshots, affected_source_nodes).
        """
        cloned = object.__new__(SpatioTemporalCandidateGraph)
        cloned.device = self.device
        cloned.max_hops = self.max_hops
        cloned.max_dt_sec = self.max_dt_sec
        cloned.top_k_prov = self.top_k_prov
        cloned.use_oracle_clock = self.use_oracle_clock
        cloned.build_time_sec = self.build_time_sec
        cloned.n_snaps = self.n_snaps
        cloned.out_nbrs = self.out_nbrs
        cloned.out_vis = self.out_vis
        cloned.out_csls_vis = self.out_csls_vis
        cloned.out_hops = self.out_hops
        cloned.out_fft = self.out_fft
        cloned.rev_knn_in_deg = self.rev_knn_in_deg
        cloned.local_knn_mean_sim = self.local_knn_mean_sim
        cloned.hub_suspicion = self.hub_suspicion
        cloned.mean_r_k = self.mean_r_k

        ds_new = dict(self.dataset)
        snap_cam = self.dataset["snap_cam"]
        new_time = self.dataset["snap_time"].copy()
        true_time = self.dataset["snap_true_time"]

        cal_snaps: Set[int] = set()
        for sid in range(self.n_snaps):
            if int(snap_cam[sid]) in calibrated_cams:
                new_time[sid] = true_time[sid]
                cal_snaps.add(sid)
        ds_new["snap_time"] = new_time
        cloned.dataset = ds_new

        sigma_tau_nom = 4.5
        kin_sigma_cal = 0.39
        kin_sigma_uncal = 0.42 + 0.012 * sigma_tau_nom

        new_out_kin: List[np.ndarray] = list(self.out_kin)
        new_out_fused: List[np.ndarray] = list(self.out_fused)
        affected_sources: Set[int] = set()

        cal_list = list(cal_snaps)
        for u in range(self.n_snaps):
            nbrs = self.out_nbrs[u]
            if len(nbrs) == 0:
                continue
            u_cal = u in cal_snaps
            v_cal_mask = np.isin(nbrs, cal_list) if not u_cal else np.ones(len(nbrs), dtype=bool)
            if not u_cal and not np.any(v_cal_mask):
                continue
            affected_sources.add(u)
            dt_arr = (new_time[nbrs] - new_time[u]).astype(np.float32)
            hop_u = self.out_hops[u]
            fft_u = self.out_fft[u]
            hop_delay = np.minimum(6.0, fft_u * 0.35) * hop_u
            exp_mean = fft_u * 1.28 + hop_delay
            safe_dt = np.maximum(0.2, dt_arr)
            log_ratio = np.log((safe_dt + 3.0) / (exp_mean + 3.0))
            sig_arr = np.where(u_cal & np.isin(nbrs, cal_list), kin_sigma_cal, 0.5 * (kin_sigma_cal + kin_sigma_uncal))
            kin_new = -0.65 * ((log_ratio / sig_arr) ** 2) - 1.50 * (hop_u - 1)
            kin_updated = self.out_kin[u].copy()
            upd_idx = np.ones(len(nbrs), dtype=bool) if u_cal else v_cal_mask
            kin_updated[upd_idx] = np.clip(kin_new[upd_idx], -10.0, 0.0).astype(np.float32)
            new_out_kin[u] = kin_updated
            new_out_fused[u] = (self.out_csls_vis[u] + self.KIN_WEIGHT * kin_updated).astype(np.float32)

        cloned.calibrated_snaps = cal_snaps
        cloned.orig_snap_time = self.dataset["snap_time"]
        cloned.out_kin = new_out_kin
        cloned.out_fused = new_out_fused
        return cloned, cal_snaps, affected_sources
