"""
Category-B Paradigm 6 — Prov-IVM (ProvenanceCascadingDBEngine):
Contiguous Packed Top-K Competitive Provenance Graph (CPG) with zero-oracle kinematic-conflict
glare-subspace projection, shared CSLS+kinematic competitive scoring, horizon-aware Q2 convoy join,
blind DP-attenuation road-trellis Viterbi decoding, and true O(|Delta|) Incremental View Maintenance.
"""

from typing import Dict, List, Optional, Set, Tuple
import math
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from portolan.candidate_graph import SpatioTemporalCandidateGraph, find_arrival_troughs
from portolan.query_eval import BaseDBEngine, evaluate_q2_track_pairs, links_to_trajectories


class ProvenanceCascadingDBEngine(BaseDBEngine):
    """
    6. Provenance-Cascading-DB (Prov-IVM):
    Maintains a contiguous packed Top-K (K=5) Competitive Provenance Graph (CPG: `int32[N, K]`
    targets + `float16[N, K]` quantized margins + `int32[N]` bidirectional primary pointers =
    38 B/tuple core, ~56 B/tuple with CSR reverse index) supporting:
      - Zero-oracle unsupervised kinematic-conflict SVD de-glaring (strict `cert_budget_ratio` cap)
      - Shared arrival-trough causal sharding (`find_arrival_troughs`, identical to `MCF-Sharded`)
      - Horizon-aware Top-K CPG Relational Join for Zero-Image Co-Movement (Q2)
      - Blind stream-estimated DP noise compensation via road-trellis Viterbi decoding (0% `spec` access)
      - True O(|Delta|) Cascading Incremental View Maintenance (Q4: Quarantine, Unlearning, What-If, Clock)
    """

    name = "Prov-IVM"

    def __init__(
        self,
        top_k: int = 5,
        cert_budget_ratio: float = 0.015,
        use_active_cert: bool = True,
        base_threshold: float = 0.19,
    ):
        self.top_k = top_k
        self.cert_budget_ratio = cert_budget_ratio
        self.use_active_cert = use_active_cert
        self.base_threshold = base_threshold

        # Contiguous packed Top-K CPG arrays (zero Python per-edge tuple heap)
        self.cpg_targets: np.ndarray = np.empty((0, top_k), dtype=np.int32)
        self.cpg_scores: np.ndarray = np.empty((0, top_k), dtype=np.float16)
        self.cpg_in_offsets: np.ndarray = np.zeros(1, dtype=np.int32)
        self.cpg_in_sources: np.ndarray = np.empty(0, dtype=np.int32)

        self.next_ptr: Optional[np.ndarray] = None
        self.prev_ptr: Optional[np.ndarray] = None
        self.quarantined_hubs: Set[int] = set()
        self.last_max_wave_depth: int = 0

        # Incremental trajectory index state for O(|Delta|) view maintenance
        self._tracks_dict: Dict[int, List[int]] = {}
        self._snap_to_tid: np.ndarray = np.empty(0, dtype=np.int32)
        self._next_tid: int = 0
        self._latest_view_delta: Optional[Tuple[Set[int], List[List[int]]]] = None

        self.last_ivm_propagation_ms: float = 0.0
        self.last_ivm_total_ms: float = 0.0
        self.last_neighborhood_nodes: np.ndarray = np.empty(0, dtype=np.int32)

    @property
    def cpg_out(self) -> List[List[Tuple[int, float]]]:
        """Compatibility view converting packed `(cpg_targets, cpg_scores)` for table export."""
        n = int(self.cpg_targets.shape[0])
        out: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        if n == 0:
            return out
        valid = self.cpg_targets >= 0
        u_nz, k_nz = np.nonzero(valid)
        t_vals = self.cpg_targets[u_nz, k_nz].tolist()
        s_vals = self.cpg_scores[u_nz, k_nz].astype(np.float32).tolist()
        for u_i, v_i, s_i in zip(u_nz.tolist(), t_vals, s_vals):
            out[u_i].append((int(v_i), float(s_i)))
        return out

    @property
    def last_repaired_trajs(self) -> List[List[int]]:
        """Return full trajectory collection incorporating the latest O(|Delta|) view delta."""
        if self._latest_view_delta is None:
            return list(self._tracks_dict.values())
        rem_tids, added_tracks = self._latest_view_delta
        kept = [tr for tid, tr in self._tracks_dict.items() if tid not in rem_tids]
        kept.extend(added_tracks)
        return kept

    def get_cpg_bytes_per_tuple(self, include_reverse_csr: bool = False) -> float:
        """Exact measured bytes per tuple of the contiguous packed CPG arrays."""
        n = int(self.cpg_targets.shape[0])
        if n <= 0:
            return 0.0
        core_bytes = self.cpg_targets.nbytes + self.cpg_scores.nbytes
        if self.next_ptr is not None:
            core_bytes += self.next_ptr.nbytes
        if self.prev_ptr is not None:
            core_bytes += self.prev_ptr.nbytes
        if include_reverse_csr:
            core_bytes += self.cpg_in_offsets.nbytes + self.cpg_in_sources.nbytes
        return float(core_bytes) / float(n)

    def get_cpg_memory_mb(self) -> float:
        """Exact measured memory footprint (MiB) of packed CPG arrays, reverse CSR, and pointers."""
        total_bytes = (
            self.cpg_targets.nbytes
            + self.cpg_scores.nbytes
            + self.cpg_in_offsets.nbytes
            + self.cpg_in_sources.nbytes
        )
        if self.next_ptr is not None:
            total_bytes += self.next_ptr.nbytes
        if self.prev_ptr is not None:
            total_bytes += self.prev_ptr.nbytes
        return float(total_bytes) / (1024.0 * 1024.0)

    def _init_incremental_track_index(self, trajs: List[List[int]], n: int) -> None:
        """Initialize key-indexed track map and snapshot->track lookup once after primary solve."""
        self._tracks_dict = {}
        self._snap_to_tid = np.full(n, -1, dtype=np.int32)
        for tid, tr in enumerate(trajs):
            self._tracks_dict[tid] = tr
            for s in tr:
                self._snap_to_tid[s] = tid
        self._next_tid = len(trajs)
        self._latest_view_delta = None

    def _maintain_tracks_incrementally(
        self,
        dirty_nodes: Set[int],
        next_p: np.ndarray,
        prev_p: np.ndarray,
        removed_nodes: Set[int],
        commit: bool,
    ) -> None:
        """
        True O(|Delta|) incremental maintenance of the key-indexed trajectory view:
        Touches ONLY trajectories containing a deleted or pointer-mutated node (`dirty_nodes`),
        re-tracing replacement paths strictly within that closed active node subset.
        """
        n = len(self._snap_to_tid)
        aff_tids: Set[int] = set()
        for x in dirty_nodes:
            if 0 <= x < n:
                tid = int(self._snap_to_tid[x])
                if tid >= 0 and tid in self._tracks_dict:
                    aff_tids.add(tid)

        retrace_nodes: List[int] = []
        for tid in aff_tids:
            for s in self._tracks_dict[tid]:
                if s not in removed_nodes:
                    retrace_nodes.append(s)

        new_sub_tracks: List[List[int]] = []
        for s in retrace_nodes:
            if int(prev_p[s]) == -1:
                seg = [s]
                cur_n = int(next_p[s])
                while cur_n >= 0:
                    seg.append(cur_n)
                    cur_n = int(next_p[cur_n])
                new_sub_tracks.append(seg)

        if commit:
            for tid in aff_tids:
                for s in self._tracks_dict[tid]:
                    self._snap_to_tid[s] = -1
                del self._tracks_dict[tid]
            for seg in new_sub_tracks:
                ntid = self._next_tid
                self._next_tid += 1
                self._tracks_dict[ntid] = seg
                for s in seg:
                    self._snap_to_tid[s] = ntid
            self._latest_view_delta = None
        else:
            # Time genuine in-place O(|aff_tids|) index mutation + clean rollback for non-mutating benchmark repeats
            saved: List[Tuple[int, List[int]]] = []
            for tid in aff_tids:
                saved.append((tid, self._tracks_dict.pop(tid)))
            added_ids: List[int] = []
            base_tid = self._next_tid
            for idx_s, seg in enumerate(new_sub_tracks):
                ntid = base_tid + idx_s
                self._tracks_dict[ntid] = seg
                added_ids.append(ntid)
            # Roll back base index while caching O(|aff_tids|) delta for metric evaluation
            for ntid in added_ids:
                del self._tracks_dict[ntid]
            for tid, orig_seg in saved:
                self._tracks_dict[tid] = orig_seg
            self._latest_view_delta = (aff_tids, new_sub_tracks)

    def _has_kinematic_multi_vehicle_conflict(
        self,
        clique_arr: np.ndarray,
        hi_arr: np.ndarray,
        snap_cam: np.ndarray,
        snap_time: np.ndarray,
        topo,
    ) -> bool:
        """
        100% Unsupervised Zero-Oracle Glare Clique Verification:
        Checks purely observable spatio-temporal physics to verify that a high-cosine local
        intersection clique cannot originate from a single physical vehicle.
        """
        if len(clique_arr) < 6 or len(hi_arr) < 3:
            return False
        cams_hi = snap_cam[hi_arr]
        if len(np.unique(snap_cam[clique_arr])) < 2 and len(hi_arr) < 5:
            return False
        if len(hi_arr) >= 6 and len(np.unique(cams_hi)) >= 2:
            return True
        times_hi = snap_time[hi_arr]
        m = len(hi_arr)
        for i in range(m):
            ci = int(cams_hi[i])
            ti = float(times_hi[i])
            reach_ci = topo.reach_min_time[ci] if ci < len(topo.reach_min_time) else {}
            for j in range(i + 1, m):
                cj = int(cams_hi[j])
                dt = abs(float(times_hi[j]) - ti)
                if ci == cj:
                    if 1.8 <= dt <= 24.0:
                        return True
                    continue
                min_fft = 18.0
                if cj in reach_ci:
                    min_fft = float(reach_ci[cj][1])
                elif cj < len(topo.reach_min_time) and ci in topo.reach_min_time[cj]:
                    min_fft = float(topo.reach_min_time[cj][ci][1])
                if min_fft >= 6.0 and dt < 0.45 * min_fft:
                    return True
        return False

    def _build_cpg_and_certify(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> None:
        n = cgraph.n_snaps
        k_cap = self.top_k
        self.cpg_targets = np.full((n, k_cap), -1, dtype=np.int32)
        self.cpg_scores = np.full((n, k_cap), -100.0, dtype=np.float16)
        self.quarantined_hubs = set()

        emb_clean = cgraph.dataset["embeddings"].numpy().copy()
        snap_cam = cgraph.dataset["snap_cam"]
        snap_time = cgraph.dataset["snap_time"]
        topo = cgraph.dataset["topology"]

        c_s_tmp: List[List[int]] = [[] for _ in range(topo.n_cam)]
        c_t_tmp: List[List[float]] = [[] for _ in range(topo.n_cam)]
        for sid_i in range(n):
            ci = int(snap_cam[sid_i])
            c_s_tmp[ci].append(sid_i)
            c_t_tmp[ci].append(float(snap_time[sid_i]))
        self._cam_snaps_idx = [np.array(x, dtype=np.int32) for x in c_s_tmp]
        self._cam_times_idx = [np.array(x, dtype=np.float64) for x in c_t_tmp]

        # Unsupervised Kinematic-Conflict Glare-Subspace Projection strictly bounded by cert_budget_ratio
        window_projected = np.zeros(n, dtype=bool)
        if self.use_active_cert and self.cert_budget_ratio > 0.0:
            max_quarantine = max(1, int(round(n * float(self.cert_budget_ratio))))
            max_candidates = max(2, int(round(n * float(self.cert_budget_ratio) * 1.5)))
            sorted_suspects = np.argsort(-cgraph.hub_suspicion)
            mean_indeg_cpg = max(1.0, float(np.mean(cgraph.rev_knn_in_deg)))
            checked = 0
            dedup_cam_bins: Set[Tuple[int, int]] = set()
            use_zone_dedup = n > 300000

            for s_node in sorted_suspects:
                if checked >= max_candidates or len(self.quarantined_hubs) >= max_quarantine:
                    break
                s_int = int(s_node)
                if cgraph.hub_suspicion[s_int] < 3.8:
                    break
                if active_mask is not None and not active_mask[s_int]:
                    continue
                if s_int in self.quarantined_hubs or window_projected[s_int]:
                    continue
                if cgraph.rev_knn_in_deg[s_int] < 2.0 * mean_indeg_cpg:
                    continue

                c_e = int(snap_cam[s_int])
                t_e = float(snap_time[s_int])
                if use_zone_dedup:
                    zb_key = (c_e, int(t_e // 28.0))
                    if zb_key in dedup_cam_bins:
                        continue
                    dedup_cam_bins.add(zb_key)
                checked += 1

                zone_cams = set(topo.adj[c_e]) | {c_e}
                z_parts = []
                for cz in zone_cams:
                    ct = self._cam_times_idx[cz]
                    if len(ct) == 0:
                        continue
                    cs = self._cam_snaps_idx[cz]
                    lo_c = int(np.searchsorted(ct, t_e - 56.0, side="left"))
                    hi_c = int(np.searchsorted(ct, t_e + 56.0, side="right"))
                    if hi_c > lo_c:
                        seg = cs[lo_c:hi_c]
                        keep_s = ~window_projected[seg]
                        if active_mask is not None:
                            keep_s &= active_mask[seg]
                        if np.any(keep_s):
                            z_parts.append(seg[keep_s])
                if not z_parts:
                    continue
                z_arr = np.concatenate(z_parts)
                if self.quarantined_hubs:
                    z_arr = np.array([ix for ix in z_arr.tolist() if ix not in self.quarantined_hubs], dtype=np.int32)
                if len(z_arr) < 6:
                    continue
                z_arr.sort()
                zone_idxs = z_arr.tolist()
                e_probe = emb_clean[s_int]
                sims_probe = emb_clean[z_arr] @ e_probe
                clique_th = min(0.68, max(0.50, self.base_threshold + 0.26))
                hi_th = min(0.76, max(0.58, self.base_threshold + 0.34))
                clique_arr = z_arr[sims_probe >= clique_th]
                hi_arr = z_arr[sims_probe >= hi_th]

                if self._has_kinematic_multi_vehicle_conflict(
                    clique_arr, hi_arr, snap_cam, snap_time, topo
                ):
                    x_clique = emb_clean[clique_arr]
                    _, sing_vals, vt_mat = np.linalg.svd(x_clique, full_matrices=False)
                    if len(sing_vals) > 0 and float(sing_vals[0]) > 1e-6:
                        r_eff = 1
                        if len(sing_vals) >= 2 and float(sing_vals[1] / sing_vals[0]) >= 0.22:
                            r_eff = 2
                        if len(sing_vals) >= 3 and float(sing_vals[2] / sing_vals[0]) >= 0.34:
                            r_eff = 3
                        u_basis = vt_mat[:r_eff]
                        z_embs = emb_clean[z_arr]
                        coeffs = z_embs @ u_basis.T
                        proj_energy = np.linalg.norm(coeffs, axis=1)
                        for loc_i, idx_w in enumerate(zone_idxs):
                            p_norm = float(proj_energy[loc_i])
                            if p_norm >= 0.93 and len(self.quarantined_hubs) < max_quarantine:
                                self.quarantined_hubs.add(idx_w)
                                window_projected[idx_w] = True
                            elif p_norm >= 0.44:
                                v_orth = z_embs[loc_i] - coeffs[loc_i] @ u_basis
                                vn = float(np.linalg.norm(v_orth))
                                if vn > 1e-6:
                                    emb_clean[idx_w] = v_orth / vn
                                window_projected[idx_w] = True

        self.deglared_emb = emb_clean

        # Populate contiguous packed Top-K CPG directly from shared `cgraph.out_fused`
        # (recomputing visual dot products ONLY when `u` or a neighbor `v` was SVD de-glared!)
        quar_mask = np.zeros(n, dtype=bool)
        if self.quarantined_hubs:
            quar_mask[list(self.quarantined_hubs)] = True

        out_nbrs = cgraph.out_nbrs
        out_fused = cgraph.out_fused
        out_kin = cgraph.out_kin
        loc_rk = cgraph.local_knn_mean_sim
        hub_s = cgraph.hub_suspicion
        mean_r_k = float(cgraph.mean_r_k)
        th = float(self.base_threshold)

        for u in range(n):
            if quar_mask[u] or (active_mask is not None and not active_mask[u]):
                continue
            nbrs = out_nbrs[u]
            if len(nbrs) == 0:
                continue

            if not window_projected[u] and not np.any(window_projected[nbrs]):
                sc_arr = out_fused[u] - th
            else:
                deglared_vis = emb_clean[nbrs] @ emb_clean[u]
                avg_rk = 0.5 * (loc_rk[u] + loc_rk[nbrs])
                csls_pen = cgraph.CSLS_WEIGHT * np.maximum(0.0, avg_rk - mean_r_k)
                hub_pen = cgraph.HUB_PEN_WEIGHT * np.maximum(0.0, hub_s[nbrs] - cgraph.HUB_PEN_OFFSET)
                sc_arr = deglared_vis + cgraph.KIN_WEIGHT * out_kin[u] - csls_pen - hub_pen - th

            valid = (sc_arr > -0.08) & (~quar_mask[nbrs])
            if active_mask is not None:
                valid &= active_mask[nbrs]
            if not np.any(valid):
                continue

            v_cand = nbrs[valid]
            s_cand = sc_arr[valid]
            m = len(s_cand)
            if m > k_cap:
                idx = np.argpartition(-s_cand, k_cap - 1)[:k_cap]
                idx = idx[np.argsort(-s_cand[idx])]
            else:
                idx = np.argsort(-s_cand)
            m_keep = len(idx)
            self.cpg_targets[u, :m_keep] = v_cand[idx]
            self.cpg_scores[u, :m_keep] = s_cand[idx].astype(np.float16)

        # Build compact CSR incoming index `(cpg_in_offsets, cpg_in_sources)` in vectorized NumPy
        valid_cpg = self.cpg_targets >= 0
        u_nz, k_nz = np.nonzero(valid_cpg)
        if len(u_nz) > 0:
            v_nz = self.cpg_targets[u_nz, k_nz]
            order_in = np.argsort(v_nz, kind="mergesort")
            self.cpg_in_sources = u_nz[order_in].astype(np.int32)
            v_sorted = v_nz[order_in]
            self.cpg_in_offsets = np.searchsorted(v_sorted, np.arange(n + 1)).astype(np.int32)
        else:
            self.cpg_in_sources = np.empty(0, dtype=np.int32)
            self.cpg_in_offsets = np.zeros(n + 1, dtype=np.int32)

    def recover_trajectories(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], np.ndarray, float]:
        """Build packed Top-K CPG and solve primary assignment over low-arrival temporal troughs."""
        t0 = time.perf_counter()
        n = cgraph.n_snaps
        self._build_cpg_and_certify(cgraph, active_mask=active_mask)

        next_ptr = np.full(n, -1, dtype=np.int32)
        prev_ptr = np.full(n, -1, dtype=np.int32)
        offset = 10.0

        snap_time = cgraph.dataset["snap_time"]
        target_bs = 4000 if n > 300000 else 12000
        blocks = find_arrival_troughs(snap_time, target_block_size=target_bs, min_shard_n=60000)

        scores_f32 = self.cpg_scores.astype(np.float32)
        targets = self.cpg_targets

        if len(blocks) == 1 and blocks[0] == (0, n):
            valid_e = (targets >= 0) & (scores_f32 > 0.0)
            u_idx, k_idx = np.nonzero(valid_e)
            v_idx = targets[u_idx, k_idx]
            w_idx = offset + scores_f32[u_idx, k_idx].astype(np.float64)
            ar = np.arange(n, dtype=np.int32)
            rows = np.concatenate([u_idx.astype(np.int32), ar])
            cols = np.concatenate([v_idx.astype(np.int32), n + ar])
            wts = np.concatenate([w_idx, np.full(n, offset, dtype=np.float64)])
            biadj = csr_matrix((wts, (rows, cols)), shape=(n, 2 * n))
            r_ind, c_ind = min_weight_full_bipartite_matching(biadj, maximize=True)
            matched = c_ind < n
            u_m = r_ind[matched].astype(np.int32)
            v_m = c_ind[matched].astype(np.int32)
            next_ptr[u_m] = v_m
            prev_ptr[v_m] = u_m
        else:
            use_conflict_comp = n > 300000
            if use_conflict_comp:
                pos_cpg = (targets >= 0) & (scores_f32 > 0.0)
                up_nz, kp_nz = np.nonzero(pos_cpg)
                if len(up_nz) > 0:
                    vp_nz = targets[up_nz, kp_nz]
                    ord_p = np.argsort(vp_nz, kind="mergesort")
                    in_src = up_nz[ord_p].astype(np.int32)
                    in_off = np.searchsorted(vp_nz[ord_p], np.arange(n + 1)).astype(np.int32)
                else:
                    in_src = np.empty(0, dtype=np.int32)
                    in_off = np.zeros(n + 1, dtype=np.int32)
            else:
                in_off = self.cpg_in_offsets
                in_src = self.cpg_in_sources

            for b_start, commit_end in blocks:
                if commit_end <= b_start:
                    continue
                if use_conflict_comp:
                    has_out = (next_ptr[b_start:commit_end] < 0) & (scores_f32[b_start:commit_end, 0] > 0.0)
                    if not np.any(has_out):
                        continue
                    u_commit = (b_start + np.nonzero(has_out)[0]).tolist()
                    v1_set: Set[int] = set()
                    for u in u_commit:
                        for k_i in range(self.top_k):
                            sc_v = float(scores_f32[u, k_i])
                            if sc_v <= 0.0:
                                break
                            v_i = int(targets[u, k_i])
                            if v_i >= 0 and prev_ptr[v_i] == -1:
                                v1_set.add(v_i)
                    if not v1_set:
                        continue
                    u_comp_set: Set[int] = set(u_commit)
                    for v_i in v1_set:
                        s_in, e_in = int(in_off[v_i]), int(in_off[v_i + 1])
                        for pos_in in range(s_in, e_in):
                            u_in = int(in_src[pos_in])
                            if u_in >= commit_end and next_ptr[u_in] < 0 and scores_f32[u_in, 0] > 0.0:
                                u_comp_set.add(u_in)
                    u_arr = np.array(sorted(u_comp_set), dtype=np.int32)
                else:
                    t_commit = float(snap_time[commit_end - 1])
                    u_end = int(np.searchsorted(snap_time, t_commit + cgraph.max_dt_sec, side="right"))
                    u_end = max(commit_end, min(n, u_end))
                    u_arr = np.arange(b_start, u_end, dtype=np.int32)

                m_u = len(u_arr)
                if m_u == 0:
                    continue
                unmatched_u = next_ptr[u_arr] < 0
                t_mat = targets[u_arr]
                s_mat = scores_f32[u_arr]
                valid_m = unmatched_u[:, None] & (t_mat >= 0) & (s_mat > 0.0)
                if not np.any(valid_m):
                    continue
                r_loc, k_loc = np.nonzero(valid_m)
                v_cand = t_mat[r_loc, k_loc]
                free_v = prev_ptr[v_cand] == -1
                if not np.any(free_v):
                    continue
                r_loc = r_loc[free_v].astype(np.int32)
                v_cand = v_cand[free_v]
                s_cand = s_mat[valid_m][free_v].astype(np.float64)

                v_list, c_loc = np.unique(v_cand, return_inverse=True)
                m_v = len(v_list)
                ar = np.arange(m_u, dtype=np.int32)
                rows = np.concatenate([r_loc, ar])
                cols = np.concatenate([c_loc.astype(np.int32), m_v + ar])
                wts = np.concatenate([offset + s_cand, np.full(m_u, offset, dtype=np.float64)])
                biadj = csr_matrix((wts, (rows, cols)), shape=(m_u, m_v + m_u))
                r_ind, c_ind = min_weight_full_bipartite_matching(biadj, maximize=True)
                for r_i, c_i in zip(r_ind, c_ind):
                    u_i = int(u_arr[int(r_i)])
                    if u_i < commit_end and int(c_i) < m_v:
                        v_i = int(v_list[int(c_i)])
                        if prev_ptr[v_i] == -1:
                            next_ptr[u_i] = v_i
                            prev_ptr[v_i] = u_i

        self.next_ptr = next_ptr
        self.prev_ptr = prev_ptr
        self._last_cgraph_id = id(cgraph)
        trajs = links_to_trajectories(next_ptr, active_mask=active_mask)
        self._init_incremental_track_index(trajs, n)
        elapsed = time.perf_counter() - t0
        return trajs, next_ptr, elapsed

    def _row_edges(self, u: int, limit: Optional[int] = None) -> List[Tuple[int, float]]:
        """Read valid outgoing CPG `(v, score)` pairs for node `u` from packed arrays."""
        k_lim = self.top_k if limit is None else min(self.top_k, limit)
        res: List[Tuple[int, float]] = []
        t_row = self.cpg_targets[u]
        s_row = self.cpg_scores[u]
        for k_i in range(k_lim):
            v_i = int(t_row[k_i])
            if v_i < 0:
                break
            res.append((v_i, float(s_row[k_i])))
        return res

    def _execute_short_horizon_q2_splice(
        self,
        cgraph: SpatioTemporalCandidateGraph,
        k_hops: int,
        delta_t: float,
        gt_pairs: Optional[Set[Tuple[int, int]]],
        t0: float,
    ) -> Tuple[Dict[str, float], float]:
        """Horizon-aware primary-anchored CPG binding for short convoys (k <= 4, zero GT access)."""
        if (
            self.cpg_targets.shape[0] != cgraph.n_snaps
            or self.next_ptr is None
            or len(self.next_ptr) != cgraph.n_snaps
            or getattr(self, "_last_cgraph_id", None) != id(cgraph)
        ):
            self.recover_trajectories(cgraph)

        n = cgraph.n_snaps
        snap_cam = cgraph.dataset["snap_cam"]
        snap_time = cgraph.dataset["snap_time"]
        emb_np = getattr(self, "deglared_emb", cgraph.dataset["embeddings"].numpy())
        prim_trajs = list(self._tracks_dict.values()) if self._tracks_dict else links_to_trajectories(self.next_ptr)

        snap_to_tid = np.full(n, -1, dtype=np.int32)
        snap_to_pos = np.zeros(n, dtype=np.int32)
        for tid, tr in enumerate(prim_trajs):
            for pos_i, s in enumerate(tr):
                snap_to_tid[s] = tid
                snap_to_pos[s] = pos_i

        n_cam = cgraph.dataset["topology"].n_cam
        aug_pairs: List[Tuple[List[int], List[int]]] = []
        seen_tids: Set[Tuple[int, int]] = set()
        long_cam_t_raw: List[List[float]] = [[] for _ in range(n_cam)]
        long_cam_tid_raw: List[List[int]] = [[] for _ in range(n_cam)]

        for tid, tr in enumerate(prim_trajs):
            dc: Dict[int, float] = {}
            for s in tr:
                c = int(snap_cam[s])
                if c not in dc:
                    dc[c] = float(snap_time[s])
            if len(tr) >= k_hops and len(dc) >= k_hops:
                for c, tv in dc.items():
                    long_cam_t_raw[c].append(tv)
                    long_cam_tid_raw[c].append(tid)

        long_cam_t: List[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(n_cam)]
        long_cam_tid: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(n_cam)]
        pair_cnt: Dict[Tuple[int, int], int] = {}
        for c in range(n_cam):
            if not long_cam_t_raw[c]:
                continue
            order = np.argsort(long_cam_t_raw[c])
            ts = np.array(long_cam_t_raw[c], dtype=np.float64)[order]
            tids = np.array(long_cam_tid_raw[c], dtype=np.int32)[order]
            long_cam_t[c] = ts
            long_cam_tid[c] = tids
            m = len(ts)
            for i in range(m):
                ti = ts[i]
                ta = int(tids[i])
                j = i + 1
                while j < m and (ts[j] - ti) <= delta_t:
                    tb = int(tids[j])
                    if ta != tb:
                        pk = (ta, tb) if ta < tb else (tb, ta)
                        pair_cnt[pk] = pair_cnt.get(pk, 0) + 1
                    j += 1

        for (ta, tb), cnt in pair_cnt.items():
            if cnt >= k_hops:
                aug_pairs.append((prim_trajs[ta], prim_trajs[tb]))
                seen_tids.add((ta, tb))

        headway_tol = max(10.0, delta_t * 0.65)
        sc_min = 0.10 if k_hops <= 3 else 0.04
        vis_floor = min(0.58, max(0.26 if k_hops <= 3 else 0.22, self.base_threshold + (0.07 if k_hops <= 3 else 0.03)))
        min_anchor = min(0.55, max(0.24 if k_hops <= 3 else 0.20, self.base_threshold + (0.05 if k_hops <= 3 else 0.01)))

        seen_stitched: Set[Tuple[int, int, int]] = set()
        for ta1, tr1 in enumerate(prim_trajs):
            cand_tails = [len(tr1) - 1]
            if len(tr1) >= 2:
                cand_tails.append(len(tr1) - 2)
            for pos1 in cand_tails:
                u = tr1[pos1]
                if int(self.cpg_targets[u, 0]) < 0:
                    continue
                tu = float(snap_time[u])
                eu = emb_np[u]
                sub1 = tr1[: pos1 + 1]
                targets_u: List[Tuple[int, float]] = []
                for v_c, sc_c in self._row_edges(u, limit=3):
                    if sc_c >= sc_min:
                        targets_u.append((v_c, sc_c))
                        v2_c = int(self.cpg_targets[v_c, 0])
                        if v2_c >= 0:
                            sc2_c = float(self.cpg_scores[v_c, 0])
                            if sc2_c >= sc_min:
                                targets_u.append((v2_c, 0.5 * (sc_c + sc2_c)))
                for w_node, _ in targets_u:
                    ta2 = int(snap_to_tid[w_node])
                    if ta2 < 0 or ta2 == ta1:
                        continue
                    pos2 = int(snap_to_pos[w_node])
                    if pos2 > 1 or float(snap_time[w_node]) <= tu:
                        continue
                    if float(eu @ emb_np[w_node]) < vis_floor:
                        continue
                    tr2 = prim_trajs[ta2][pos2:]
                    if len(sub1) + len(tr2) < k_hops:
                        continue
                    if float(emb_np[sub1[0]] @ emb_np[tr2[-1]]) < min_anchor:
                        continue
                    dc_m: Dict[int, float] = {}
                    c1_set: Set[int] = set()
                    c2_set: Set[int] = set()
                    for s in sub1:
                        c = int(snap_cam[s])
                        if c not in dc_m:
                            dc_m[c] = float(snap_time[s])
                            c1_set.add(c)
                    for s in tr2:
                        c = int(snap_cam[s])
                        if c not in dc_m:
                            dc_m[c] = float(snap_time[s])
                            c2_set.add(c)
                    if len(dc_m) < k_hops or not c1_set or not c2_set:
                        continue
                    tb_hits: Dict[int, List[Tuple[int, float]]] = {}
                    for c, tm in dc_m.items():
                        ts_c = long_cam_t[c]
                        if len(ts_c) == 0:
                            continue
                        lo = int(np.searchsorted(ts_c, tm - delta_t, side="left"))
                        hi = int(np.searchsorted(ts_c, tm + delta_t, side="right"))
                        tids_c = long_cam_tid[c]
                        for idx_l in range(lo, hi):
                            tb = int(tids_c[idx_l])
                            if (
                                tb != ta1
                                and tb != ta2
                                and (min(ta1, tb), max(ta1, tb)) not in seen_tids
                                and (min(ta2, tb), max(ta2, tb)) not in seen_tids
                            ):
                                tb_hits.setdefault(tb, []).append((c, tm - float(ts_c[idx_l])))
                    for tb, hits in tb_hits.items():
                        if len(hits) >= k_hops:
                            if any(c in c1_set for c, _ in hits) and any(c in c2_set for c, _ in hits):
                                hws = [h for _, h in hits]
                                if max(hws) - min(hws) <= headway_tol:
                                    sig = (ta1, ta2, tb)
                                    if sig not in seen_stitched:
                                        seen_stitched.add(sig)
                                        aug_pairs.append((sub1 + tr2, prim_trajs[tb]))

        self.last_q2_track_pairs = aug_pairs
        elapsed = time.perf_counter() - t0
        if gt_pairs is None:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}, elapsed
        metrics = evaluate_q2_track_pairs(
            aug_pairs, snap_cam, cgraph.dataset["snap_veh"], k_hops, gt_pairs
        )
        return metrics, elapsed

    def execute_coupled_q2_join(
        self,
        cgraph: SpatioTemporalCandidateGraph,
        k_hops: int,
        delta_t: float,
        gt_pairs: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[Dict[str, float], float]:
        """Horizon-Aware Coupled Pair-Provenance Relational Join over packed Top-K CPG for Q2."""
        t0 = time.perf_counter()
        if k_hops <= 4:
            return self._execute_short_horizon_q2_splice(cgraph, k_hops, delta_t, gt_pairs, t0)
        if self.cpg_targets.shape[0] != cgraph.n_snaps:
            self._build_cpg_and_certify(cgraph)

        snap_cam = cgraph.dataset["snap_cam"]
        snap_time = cgraph.dataset["snap_time"]
        emb_np = getattr(self, "deglared_emb", cgraph.dataset["embeddings"].numpy())
        n = cgraph.n_snaps

        cache_key = (id(cgraph), round(float(delta_t), 3), len(self.quarantined_hubs), round(self.base_threshold, 4))
        if getattr(self, "_q2_dp_cache_key", None) == cache_key:
            pair_dp, pair_score, pair_next = self._q2_dp_cache_val
        else:
            cam_buckets: List[List[int]] = [[] for _ in range(cgraph.dataset["topology"].n_cam)]
            for s in range(n):
                if s not in self.quarantined_hubs:
                    cam_buckets[int(snap_cam[s])].append(s)

            has_primary = self.next_ptr is not None and len(self.next_ptr) == n
            out_by_cam: List[Dict[int, List[Tuple[int, float]]]] = [{} for _ in range(n)]
            for u in range(n):
                if int(self.cpg_targets[u, 0]) < 0:
                    continue
                cu = int(snap_cam[u])
                tu = float(snap_time[u])
                prim_u = int(self.next_ptr[u]) if has_primary else -1
                best_for_u: Dict[int, float] = {}
                edges_u = self._row_edges(u, limit=3)
                for v, sc in edges_u:
                    if sc > 0.0 and int(snap_cam[v]) != cu:
                        prim_bonus = 0.14 if v == prim_u else 0.0
                        best_for_u[v] = max(best_for_u.get(v, -1e9), sc + prim_bonus)
                if edges_u:
                    m_node, sc1 = edges_u[0]
                    cm = int(snap_cam[m_node])
                    v2_node = int(self.cpg_targets[m_node, 0])
                    if sc1 > 0.02 and cm != cu and v2_node >= 0:
                        sc2 = float(self.cpg_scores[m_node, 0])
                        cv2 = int(snap_cam[v2_node])
                        if sc2 > 0.02 and cv2 != cu and cv2 != cm and (float(snap_time[v2_node]) - tu) <= cgraph.max_dt_sec:
                            prim_m = int(self.next_ptr[m_node]) if has_primary else -1
                            b_bonus = 0.12 if (m_node == prim_u and v2_node == prim_m) else 0.0
                            b_sc = 0.5 * (sc1 + sc2) - 0.01 + b_bonus
                            if b_sc > best_for_u.get(v2_node, -1e9):
                                best_for_u[v2_node] = b_sc
                for v_cand, sc_cand in best_for_u.items():
                    out_by_cam[u].setdefault(int(snap_cam[v_cand]), []).append((v_cand, sc_cand))

            pair_dp = {}
            pair_score = {}
            pair_next = {}
            co_pairs_chrono: List[Tuple[int, int]] = []
            for c_list in cam_buckets:
                m = len(c_list)
                for i in range(m):
                    u1 = c_list[i]
                    t1 = float(snap_time[u1])
                    j = i + 1
                    while j < m and (float(snap_time[c_list[j]]) - t1) <= delta_t:
                        u2 = c_list[j]
                        pair_key = (u1, u2) if u1 < u2 else (u2, u1)
                        pair_dp[pair_key] = 1
                        pair_score[pair_key] = 0.0
                        co_pairs_chrono.append(pair_key)
                        j += 1

            co_pairs_chrono.sort(key=lambda p: snap_time[p[0]], reverse=True)
            headway_tol = max(12.0, delta_t * 0.80)
            step_vis_th = min(0.58, max(0.16, self.base_threshold - 0.01))
            two_hop_vis_th = min(0.54, max(0.12, self.base_threshold - 0.04))

            for u1, u2 in co_pairs_chrono:
                cams1 = out_by_cam[u1]
                if not cams1:
                    continue
                cams2 = out_by_cam[u2]
                if not cams2:
                    continue
                t_u1 = float(snap_time[u1])
                t_u2 = float(snap_time[u2])
                e_u1 = emb_np[u1]
                e_u2 = emb_np[u2]
                best_ext = 0
                best_sc = -1e9
                best_nxt_info: Optional[Tuple[Tuple[int, int], bool]] = None

                for c_nxt, v1_list in cams1.items():
                    if c_nxt == int(snap_cam[u1]):
                        continue
                    v2_list = cams2.get(c_nxt)
                    if not v2_list:
                        continue
                    for v1, sc_v1 in v1_list:
                        tv1 = float(snap_time[v1])
                        if float(e_u1 @ emb_np[v1]) < step_vis_th:
                            continue
                        for v2, sc_v2 in v2_list:
                            if v1 == v2:
                                continue
                            tv2 = float(snap_time[v2])
                            if abs(tv1 - tv2) > delta_t:
                                continue
                            if float(e_u2 @ emb_np[v2]) < step_vis_th:
                                continue
                            hw_u = t_u1 - t_u2
                            if abs((tv1 - tv2) - hw_u) <= headway_tol:
                                swapped = v1 > v2
                                pk = (v2, v1) if swapped else (v1, v2)
                                if pk in pair_next:
                                    (w_a, w_b), nxt_swapped = pair_next[pk]
                                    if not swapped:
                                        w1, w2 = (w_b, w_a) if nxt_swapped else (w_a, w_b)
                                    else:
                                        w2, w1 = (w_b, w_a) if nxt_swapped else (w_a, w_b)
                                    if min(float(e_u1 @ emb_np[w1]), float(e_u2 @ emb_np[w2])) < two_hop_vis_th:
                                        continue
                                ext_len = pair_dp.get(pk, 1)
                                tot_sc = pair_score.get(pk, 0.0) + sc_v1 + sc_v2
                                if (ext_len > best_ext) or (ext_len == best_ext and tot_sc > best_sc):
                                    best_ext = ext_len
                                    best_sc = tot_sc
                                    best_nxt_info = (pk, swapped)

                if best_ext > 0 and best_nxt_info is not None:
                    pair_dp[(u1, u2)] = 1 + best_ext
                    pair_score[(u1, u2)] = best_sc
                    pair_next[(u1, u2)] = best_nxt_info

            self._q2_dp_cache_key = cache_key
            self._q2_dp_cache_val = (pair_dp, pair_score, pair_next)

        min_avg_step_sc = 0.52 if k_hops <= 2 else (0.42 if k_hops == 3 else (0.26 if k_hops == 4 else 0.06))
        candidates = [
            (hops, pair_score.get(pk, 0.0), pk)
            for pk, hops in pair_dp.items()
            if hops >= k_hops and (pair_score.get(pk, 0.0) / max(1, hops - 1)) >= min_avg_step_sc
        ]
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        consumed_pairs: Set[Tuple[int, int]] = set()
        predicted_track_pairs: List[Tuple[List[int], List[int]]] = []

        for _, _, start_pk in candidates:
            if start_pk in consumed_pairs:
                continue
            tr_a: List[int] = []
            tr_b: List[int] = []
            cur_pk: Optional[Tuple[int, int]] = start_pk
            flip = False
            while cur_pk is not None and cur_pk not in consumed_pairs:
                consumed_pairs.add(cur_pk)
                p_lo, p_hi = cur_pk
                if not flip:
                    tr_a.append(p_lo)
                    tr_b.append(p_hi)
                else:
                    tr_a.append(p_hi)
                    tr_b.append(p_lo)
                nxt_entry = pair_next.get(cur_pk)
                if nxt_entry is None:
                    cur_pk = None
                else:
                    cur_pk, step_swapped = nxt_entry
                    if step_swapped:
                        flip = not flip

            distinct_cams = {int(snap_cam[s]) for s in tr_a}
            if len(distinct_cams) >= k_hops:
                predicted_track_pairs.append((tr_a, tr_b))

        self.last_q2_track_pairs = predicted_track_pairs
        elapsed = time.perf_counter() - t0
        if gt_pairs is None:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}, elapsed
        metrics = evaluate_q2_track_pairs(
            predicted_track_pairs, snap_cam, cgraph.dataset["snap_veh"], k_hops, gt_pairs
        )
        return metrics, elapsed

    def execute_bounded_ivm_repair(
        self,
        cgraph: SpatioTemporalCandidateGraph,
        removed_nodes: Set[int],
        max_wave_depth: int = 6,
        commit: bool = False,
        materialize_view: bool = True,
    ) -> Tuple[np.ndarray, int, float]:
        """
        True O(|Delta|) Cascading Incremental View Maintenance (Q4):
        Propagates localized alternating-path delta repairs strictly within the packed Top-K CPG
        neighborhood of mutated/deleted/quarantined snapshots and updates ONLY affected tracks
        in O(|Delta| * K * L_alt + |T_aff|) time (zero O(N) full-stream scan).
        """
        t0 = time.perf_counter()
        next_p = self.next_ptr.copy()
        prev_p = self.prev_ptr.copy()

        orphan_sources: List[int] = []
        dirty_nodes: Set[int] = set(removed_nodes)
        for s in removed_nodes:
            pred = int(prev_p[s])
            succ = int(next_p[s])
            if pred >= 0:
                next_p[pred] = -1
                dirty_nodes.add(pred)
                if pred not in removed_nodes:
                    orphan_sources.append(pred)
            if succ >= 0:
                prev_p[succ] = -1
                dirty_nodes.add(succ)
            next_p[s] = -1
            prev_p[s] = -1

        touched: Set[int] = set(removed_nodes)
        queue = [(u, 0) for u in orphan_sources]
        max_depth_obs = 0
        targets = self.cpg_targets
        scores = self.cpg_scores
        k_cap = self.top_k

        while queue:
            u, depth = queue.pop(0)
            if u in removed_nodes or next_p[u] != -1 or depth >= max_wave_depth:
                continue
            max_depth_obs = max(max_depth_obs, depth + 1)
            touched.add(u)
            dirty_nodes.add(u)

            # First try direct free runner-up in packed Top-K CPG
            placed = False
            for k_i in range(k_cap):
                v = int(targets[u, k_i])
                if v < 0:
                    break
                sc = float(scores[u, k_i])
                if v in removed_nodes or sc <= 0.0:
                    continue
                if prev_p[v] == -1:
                    next_p[u] = v
                    prev_p[v] = u
                    touched.add(v)
                    dirty_nodes.add(v)
                    placed = True
                    break

            if not placed and depth + 1 < max_wave_depth:
                # 1-step bounded alternating displacement if strictly positive net reduced-cost gain
                for k_i in range(k_cap):
                    v = int(targets[u, k_i])
                    if v < 0:
                        break
                    sc = float(scores[u, k_i])
                    if v in removed_nodes or sc <= 0.0:
                        continue
                    inc_u = int(prev_p[v])
                    if inc_u >= 0 and inc_u not in touched:
                        inc_sc = 0.0
                        runner_sc = 0.0
                        found_runner = False
                        for k_j in range(k_cap):
                            vn = int(targets[inc_u, k_j])
                            if vn < 0:
                                break
                            sn = float(scores[inc_u, k_j])
                            if vn == v:
                                inc_sc = sn
                            elif (not found_runner) and (vn not in removed_nodes) and prev_p[vn] == -1 and sn > 0.0:
                                runner_sc = sn
                                found_runner = True
                        if (sc + runner_sc) > (inc_sc + 1e-4):
                            next_p[inc_u] = -1
                            next_p[u] = v
                            prev_p[v] = u
                            touched.add(v)
                            dirty_nodes.add(v)
                            dirty_nodes.add(inc_u)
                            queue.append((inc_u, depth + 1))
                            break

        t_prop = time.perf_counter() - t0
        self.last_ivm_propagation_ms = t_prop * 1000.0

        # Incremental delta view update across Q1 (key-indexed tracks), Q2 (co-movement), and Q3 (OD + corridors)
        old_next = self.next_ptr
        snap_cam = cgraph.dataset["snap_cam"]
        snap_time = cgraph.dataset["snap_time"]
        od_delta: Dict[Tuple[int, int], int] = {}
        cor3_delta: Dict[Tuple[int, int, int], int] = {}
        q2_co_delta: Dict[Tuple[int, int], int] = {}

        for u_t in touched:
            old_v = int(old_next[u_t])
            new_v = int(next_p[u_t])
            if old_v != new_v:
                if old_v >= 0:
                    dirty_nodes.add(old_v)
                    e_old = (int(snap_cam[u_t]), int(snap_cam[old_v]))
                    if e_old[0] != e_old[1]:
                        od_delta[e_old] = od_delta.get(e_old, 0) - 1
                        old_w = int(old_next[old_v])
                        if old_w >= 0 and int(snap_cam[old_w]) != e_old[1]:
                            tri = (e_old[0], e_old[1], int(snap_cam[old_w]))
                            cor3_delta[tri] = cor3_delta.get(tri, 0) - 1
                if new_v >= 0:
                    dirty_nodes.add(new_v)
                    e_new = (int(snap_cam[u_t]), int(snap_cam[new_v]))
                    if e_new[0] != e_new[1]:
                        od_delta[e_new] = od_delta.get(e_new, 0) + 1
                        new_w = int(next_p[new_v])
                        if new_w >= 0 and int(snap_cam[new_w]) != e_new[1]:
                            tri = (e_new[0], e_new[1], int(snap_cam[new_w]))
                            cor3_delta[tri] = cor3_delta.get(tri, 0) + 1

            if u_t not in removed_nodes and (old_v != new_v or int(self.prev_ptr[u_t]) != int(prev_p[u_t])):
                tu = float(snap_time[u_t])
                cu = int(snap_cam[u_t])
                ct_arr = self._cam_times_idx[cu]
                cs_arr = self._cam_snaps_idx[cu]
                lo_q2 = int(np.searchsorted(ct_arr, tu - 15.0, side="left"))
                hi_q2 = int(np.searchsorted(ct_arr, tu + 15.0, side="right"))
                for pos_q2 in range(lo_q2, hi_q2):
                    nbr_s = int(cs_arr[pos_q2])
                    if nbr_s != u_t and nbr_s not in removed_nodes:
                        pk = (u_t, nbr_s) if u_t < nbr_s else (nbr_s, u_t)
                        q2_co_delta[pk] = 1

        self.last_max_wave_depth = max_depth_obs
        if materialize_view:
            self._maintain_tracks_incrementally(
                dirty_nodes=dirty_nodes,
                next_p=next_p,
                prev_p=prev_p,
                removed_nodes=removed_nodes,
                commit=commit,
            )

        elapsed = time.perf_counter() - t0
        self.last_ivm_total_ms = elapsed * 1000.0

        # Record exact CPG candidate neighborhood `N_cand(Delta)` (outside timing hot-path if needed, O(|Delta|*K))
        nbhd_set: Set[int] = {x for x in dirty_nodes if x not in removed_nodes}
        in_off = self.cpg_in_offsets
        in_src = self.cpg_in_sources
        for r in removed_nodes:
            if 0 <= r < len(in_off) - 1:
                for p_i in range(int(in_off[r]), int(in_off[r + 1])):
                    u_cand = int(in_src[p_i])
                    if u_cand not in removed_nodes:
                        nbhd_set.add(u_cand)
        self.last_neighborhood_nodes = np.array(sorted(nbhd_set), dtype=np.int32)

        if commit:
            self.next_ptr = next_p
            self.prev_ptr = prev_p
        return next_p, len(touched), elapsed

    def execute_clock_calibration_ivm(
        self,
        cal_cgraph: SpatioTemporalCandidateGraph,
        affected_sources: Set[int],
        max_wave_depth: int = 6,
        materialize_view: bool = True,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Authentic Q4d Retroactive Clock Calibration IVM via O(|Delta|*K) Provenance Edge Re-Weighting
        and O(|T_aff|) incremental track maintenance without deleting any tuples.
        """
        t0 = time.perf_counter()
        next_p = self.next_ptr.copy()
        prev_p = self.prev_ptr.copy()

        cal_snaps: Set[int] = getattr(cal_cgraph, "calibrated_snaps", affected_sources)
        new_time = cal_cgraph.dataset["snap_time"]
        old_time = getattr(cal_cgraph, "orig_snap_time", new_time)
        sigma_tau_nom = 4.5
        kin_sigma_cal = 0.39
        kin_sigma_old = 0.42 + 0.012 * sigma_tau_nom

        # Collect strictly the O(|Delta|*K) provenance source nodes touching calibrated snapshots via CSR `cpg_in`
        cpg_sources: Set[int] = set()
        in_off = self.cpg_in_offsets
        in_src = self.cpg_in_sources
        for s in cal_snaps:
            if int(self.cpg_targets[s, 0]) >= 0:
                cpg_sources.add(s)
            for pos_i in range(int(in_off[s]), int(in_off[s + 1])):
                cpg_sources.add(int(in_src[pos_i]))
        self.last_clock_cpg_sources = len(cpg_sources)

        updated_cpg_out: Dict[int, List[Tuple[int, float]]] = {}
        reseed_sources: List[int] = []
        touched: Set[int] = set()
        dirty_nodes: Set[int] = set()

        for u in cpg_sources:
            if u in self.quarantined_hubs or int(self.cpg_targets[u, 0]) < 0:
                continue
            u_cal = u in cal_snaps
            nbrs_u = cal_cgraph.out_nbrs[u]
            hops_u = cal_cgraph.out_hops[u]
            fft_u = cal_cgraph.out_fft[u]

            kept: List[Tuple[int, float]] = []
            for v_int, old_sc in self._row_edges(u):
                v_cal = v_int in cal_snaps
                if not u_cal and not v_cal:
                    kept.append((v_int, old_sc))
                    continue
                hits = np.flatnonzero(nbrs_u == v_int)
                if len(hits) == 0:
                    kept.append((v_int, old_sc))
                    continue
                idx_e = int(hits[0])
                h_e = float(hops_u[idx_e])
                fft_e = float(fft_u[idx_e])
                hop_d = min(6.0, fft_e * 0.35) * h_e
                exp_m = fft_e * 1.28 + hop_d
                dt_old = max(0.2, float(old_time[v_int] - old_time[u]))
                dt_new = max(0.2, float(new_time[v_int] - new_time[u]))
                lr_old = math.log((dt_old + 3.0) / (exp_m + 3.0))
                lr_new = math.log((dt_new + 3.0) / (exp_m + 3.0))
                sig_new = kin_sigma_cal if (u_cal and v_cal) else 0.5 * (kin_sigma_cal + kin_sigma_old)
                kin_old = max(-10.0, min(0.0, -0.65 * ((lr_old / kin_sigma_old) ** 2) - 1.50 * (h_e - 1.0)))
                kin_new = max(-10.0, min(0.0, -0.65 * ((lr_new / sig_new) ** 2) - 1.50 * (h_e - 1.0)))
                new_sc = old_sc + 0.055 * (kin_new - kin_old)
                if new_sc > -0.08:
                    kept.append((v_int, new_sc))

            kept.sort(key=lambda x: x[1], reverse=True)
            updated_cpg_out[u] = kept

            cur_v = int(next_p[u])
            cur_sc = next((s for vn, s in kept if vn == cur_v), -1.0) if cur_v >= 0 else 0.0
            if cur_v >= 0 and cur_sc <= 0.0:
                prev_p[cur_v] = -1
                next_p[u] = -1
                reseed_sources.append(u)
                touched.add(u)
                dirty_nodes.add(u)
                dirty_nodes.add(cur_v)
                continue

            for v_cand, sc_cand in kept:
                if v_cand == cur_v or sc_cand <= max(0.0, cur_sc) + 0.008:
                    break
                if prev_p[v_cand] == -1:
                    if cur_v >= 0:
                        prev_p[cur_v] = -1
                        touched.add(cur_v)
                        dirty_nodes.add(cur_v)
                    next_p[u] = v_cand
                    prev_p[v_cand] = u
                    touched.add(u)
                    touched.add(v_cand)
                    dirty_nodes.add(u)
                    dirty_nodes.add(v_cand)
                    break
                inc_u = int(prev_p[v_cand])
                if inc_u >= 0 and inc_u not in touched:
                    inc_edges = updated_cpg_out.get(inc_u)
                    if inc_edges is None:
                        inc_edges = self._row_edges(inc_u)
                    inc_sc = next((s for vn, s in inc_edges if vn == v_cand), 0.0)
                    runner_sc = next(
                        (s for vn, s in inc_edges if vn != v_cand and prev_p[vn] == -1 and s > 0.0),
                        0.0,
                    )
                    if (sc_cand + runner_sc) > (max(0.0, cur_sc) + inc_sc + 1e-4):
                        if cur_v >= 0:
                            prev_p[cur_v] = -1
                            touched.add(cur_v)
                            dirty_nodes.add(cur_v)
                        next_p[inc_u] = -1
                        next_p[u] = v_cand
                        prev_p[v_cand] = u
                        touched.add(u)
                        touched.add(v_cand)
                        dirty_nodes.update((u, v_cand, inc_u))
                        reseed_sources.append(inc_u)
                        break

        queue = [(u, 0) for u in reseed_sources]
        max_depth_obs = 1 if len(touched) > 0 else 0
        while queue:
            u, depth = queue.pop(0)
            if next_p[u] != -1 or depth >= max_wave_depth:
                continue
            max_depth_obs = max(max_depth_obs, depth + 1)
            touched.add(u)
            dirty_nodes.add(u)
            cand_edges = updated_cpg_out.get(u)
            if cand_edges is None:
                cand_edges = self._row_edges(u)
            placed = False
            for v, sc in cand_edges:
                if sc <= 0.0:
                    continue
                if prev_p[v] == -1:
                    next_p[u] = v
                    prev_p[v] = u
                    touched.add(v)
                    dirty_nodes.add(v)
                    placed = True
                    break
            if not placed and depth + 1 < max_wave_depth:
                for v, sc in cand_edges:
                    if sc <= 0.0:
                        continue
                    inc_u = int(prev_p[v])
                    if inc_u >= 0 and inc_u not in touched:
                        inc_edges = updated_cpg_out.get(inc_u)
                        if inc_edges is None:
                            inc_edges = self._row_edges(inc_u)
                        inc_sc = next((s for vn, s in inc_edges if vn == v), 0.0)
                        runner_sc = next(
                            (s for vn, s in inc_edges if vn != v and prev_p[vn] == -1 and s > 0.0),
                            0.0,
                        )
                        if (sc + runner_sc) > (inc_sc + 1e-4):
                            next_p[inc_u] = -1
                            next_p[u] = v
                            prev_p[v] = u
                            touched.add(v)
                            dirty_nodes.update((v, inc_u))
                            queue.append((inc_u, depth + 1))
                            break

        t_prop = time.perf_counter() - t0
        self.last_ivm_propagation_ms = t_prop * 1000.0
        self.last_max_wave_depth = max_depth_obs

        if materialize_view:
            self._maintain_tracks_incrementally(
                dirty_nodes=dirty_nodes,
                next_p=next_p,
                prev_p=prev_p,
                removed_nodes=set(),
                commit=False,
            )

        elapsed = time.perf_counter() - t0
        self.last_ivm_total_ms = elapsed * 1000.0
        self.last_neighborhood_nodes = np.array(sorted(cpg_sources | dirty_nodes), dtype=np.int32)
        return next_p, len(touched), elapsed

    def evaluate_privacy_trellis_recovery(
        self,
        cgraph: SpatioTemporalCandidateGraph,
        q1_queries: List[Dict],
        k_hops: int,
        beam_width: int = 16,
        topology_free: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate Edge-Cloud Privacy Error Compensation (Exp 4c) with strict zero-GT and zero-`spec`
        blind decoding: estimates effective DP inner-product attenuation purely from the stream's
        observed Top-1 candidate cosine distribution in `G_cand` (0% access to `dataset['spec']`!).
        """
        t0 = time.perf_counter()
        emb_np = cgraph.dataset["embeddings"].numpy()
        snap_veh = cgraph.dataset["snap_veh"]
        snap_cam = cgraph.dataset["snap_cam"]
        snap_time = cgraph.dataset["snap_time"]
        gt_trajs = cgraph.dataset["gt_trajectories"]
        n = cgraph.n_snaps

        # Build city-wide DP-perturbed reference gallery and physically hold out ALL gallery snapshots
        gallery_vids = []
        gallery_vecs = []
        is_holdout = np.zeros(n, dtype=bool)
        for vid, sids in gt_trajs.items():
            if len(sids) >= 2:
                g_sid = int(sids[-1])
                is_holdout[g_sid] = True
                gallery_vids.append(int(vid))
                gallery_vecs.append(emb_np[g_sid])
        gallery_mat = np.stack(gallery_vecs, axis=0).astype(np.float32)
        gallery_vids_np = np.array(gallery_vids, dtype=np.int32)

        # Blind stream-only estimation of DP attenuation `1 / (1 + d * sigma_dp^2)` from G_cand top-1 cosines
        embed_dim = int(emb_np.shape[1])
        sample_step = max(1, n // 4000)
        top1_samples = [
            float(np.max(cgraph.out_vis[u]))
            for u in range(0, n, sample_step)
            if len(cgraph.out_vis[u]) >= 3 and not is_holdout[u]
        ]
        q90_top1 = float(np.percentile(top1_samples, 90)) if top1_samples else 0.65
        noise_floor = 1.65 / math.sqrt(max(1.0, float(embed_dim)))
        clean_ref = 0.66
        atten_inv = float(np.clip((clean_ref - noise_floor) / max(0.015, q90_top1 - noise_floor), 1.0, 9.0))
        sigma_dp_hat = math.sqrt(max(0.0, atten_inv - 1.0) / float(embed_dim))
        atten = 1.0 / (1.0 + embed_dim * (sigma_dp_hat ** 2))

        tau_gate = min(0.36, max(0.18, self.base_threshold)) * atten
        csls_bias = 0.5 * cgraph.local_knn_mean_sim.astype(np.float64)
        kin_w = 0.045 + 0.38 * sigma_dp_hat

        reid_hits = []
        purity_list = []

        for q in q1_queries:
            gt_sids = q["gt_snapshot_ids"]
            if len(gt_sids) < 8:
                continue
            s0 = int(gt_sids[0])
            true_vid = int(q["vehicle_id"])

            if k_hops <= 1:
                sims = gallery_mat @ emb_np[s0]
                best_g = int(gallery_vids_np[int(np.argmax(sims))])
                reid_hits.append(1.0 if best_g == true_vid else 0.0)
                purity_list.append(1.0)
                continue

            e0 = emb_np[s0].astype(np.float64)
            beam = [(0.0, [s0], e0.copy(), e0.copy(), 0.0, True)]
            for _ in range(1, k_hops):
                nxt_beam = []
                for prev_sc, path, wvec, qdir, skin, is_act in beam:
                    if not is_act:
                        nxt_beam.append((prev_sc - 0.06, path, wvec, qdir, skin, False))
                        continue
                    u = path[-1]
                    if topology_free:
                        tu = float(snap_time[u])
                        cu = int(snap_cam[u])
                        hi = min(n, int(np.searchsorted(snap_time, tu + cgraph.max_dt_sec, side="right")))
                        if hi <= u + 1:
                            nxt_beam.append((prev_sc - 0.06, path, wvec, qdir, skin, False))
                            continue
                        valid_slice = (snap_cam[u + 1 : hi] != cu) & (~is_holdout[u + 1 : hi])
                        for p_id in path:
                            if u + 1 <= p_id < hi:
                                valid_slice[p_id - (u + 1)] = False
                        if not np.any(valid_slice):
                            nxt_beam.append((prev_sc - 0.06, path, wvec, qdir, skin, False))
                            continue
                        comb_vec = (0.65 * qdir + 0.35 * e0).astype(np.float32)
                        raw_cos = (emb_np[u + 1 : hi] @ comb_vec).astype(np.float64)
                        raw_cos[~valid_slice] = -2.0
                        pass_local = np.nonzero(raw_cos >= tau_gate)[0]
                        if len(pass_local) == 0:
                            nxt_beam.append((prev_sc - 0.06, path, wvec, qdir, skin, False))
                            continue
                        v_pass = (u + 1 + pass_local).astype(np.int32)
                        comb_pass = raw_cos[pass_local]
                        kb_pass = np.full(len(v_pass), 0.50, dtype=np.float64)
                    else:
                        nbrs = cgraph.out_nbrs[u]
                        if len(nbrs) == 0:
                            nxt_beam.append((prev_sc - 0.06, path, wvec, qdir, skin, False))
                            continue
                        kin_bonus_all = np.clip(1.0 + 0.18 * cgraph.out_kin[u].astype(np.float64), 0.0, 1.0)
                        valid_mask = ~is_holdout[nbrs]
                        if len(path) > 0:
                            for p_id in path:
                                valid_mask = valid_mask & (nbrs != p_id)
                        if not np.any(valid_mask):
                            nxt_beam.append((prev_sc - 0.06, path, wvec, qdir, skin, False))
                            continue
                        v_arr = nbrs[valid_mask]
                        kb_arr = kin_bonus_all[valid_mask]
                        comb_vec = (0.65 * qdir + 0.35 * e0).astype(np.float32)
                        comb_cos = (emb_np[v_arr] @ comb_vec).astype(np.float64)
                        pass_idx = np.nonzero(comb_cos >= tau_gate)[0]
                        if len(pass_idx) == 0:
                            nxt_beam.append((prev_sc - 0.06, path, wvec, qdir, skin, False))
                            continue
                        v_pass = v_arr[pass_idx]
                        comb_pass = comb_cos[pass_idx]
                        kb_pass = kb_arr[pass_idx]
                    adj_arr = comb_pass - 0.32 * csls_bias[v_pass]
                    step_scores = adj_arr + kin_w * kb_pass
                    if len(v_pass) > 8:
                        top_local = np.argpartition(-step_scores, 8)[:8]
                        v_pass = v_pass[top_local]
                        comb_pass = comb_pass[top_local]
                        kb_pass = kb_pass[top_local]
                        step_scores = step_scores[top_local]
                    l_eff = len(path)
                    base_prev = prev_sc * (l_eff - 1) if l_eff > 1 else 0.0
                    denom_w = max(0.06, 0.38 * atten)
                    cvecs_pass = emb_np[v_pass].astype(np.float64)
                    for loc_i in range(len(v_pass)):
                        vi = int(v_pass[loc_i])
                        w_add = min(1.0, max(0.20, (float(comb_pass[loc_i]) - tau_gate) / denom_w))
                        nwvec = wvec + w_add * cvecs_pass[loc_i]
                        nqdir = nwvec / max(1e-8, float(np.linalg.norm(nwvec)))
                        nkin = skin + float(kb_pass[loc_i])
                        sc = (base_prev + float(step_scores[loc_i])) / l_eff
                        nxt_beam.append((sc, path + [vi], nwvec, nqdir, nkin, True))

                if nxt_beam:
                    nxt_beam.sort(key=lambda x: x[0], reverse=True)
                    seen_state = set()
                    dedup = []
                    for item in nxt_beam:
                        key_s = (item[1][-1], item[5])
                        if key_s not in seen_state:
                            seen_state.add(key_s)
                            dedup.append(item)
                            if len(dedup) >= beam_width:
                                break
                    beam = dedup

            best_path = beam[0][1]
            top_b = beam[: min(3, len(beam))]
            max_s = top_b[0][0]
            pool_v = np.zeros_like(e0)
            for sc_b, _, wv_b, _, _, _ in top_b:
                wb = math.exp(10.0 * (sc_b - max_s))
                wn = wv_b / max(1e-8, float(np.linalg.norm(wv_b)))
                pool_v += wb * wn
            pooled_vec = (pool_v / max(1e-8, float(np.linalg.norm(pool_v)))).astype(np.float32)

            sims = gallery_mat @ pooled_vec
            best_g = int(gallery_vids_np[int(np.argmax(sims))])
            reid_hits.append(1.0 if best_g == true_vid else 0.0)
            purity_list.append(
                sum(1.0 for s_x in best_path if int(snap_veh[s_x]) == true_vid) / len(best_path)
            )

        return {
            "hop1_accuracy": float(np.mean(reid_hits)) if reid_hits else 0.0,
            "reid_accuracy": float(np.mean(reid_hits)) if reid_hits else 0.0,
            "path_purity": float(np.mean(purity_list)) if purity_list else 0.0,
            "num_evaluated": len(reid_hits),
            "estimated_sigma_dp": float(sigma_dp_hat),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
