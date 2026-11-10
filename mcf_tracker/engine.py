"""
Category-B Paradigm 2 — MCF-Tracker & MCF-Sharded (MinCostNetFlowEngine):
Minimum-cost node-disjoint DAG path cover over shared CSLS+kinematic candidate edges
(Zhang et al. CVPR 2008), solved via sparse bipartite Jonker-Volgenant optimization (`LAPJVsp`),
supporting both unsharded global solve (`MCF-Tracker`) and arrival-trough sharded Top-K solve
(`MCF-Sharded`) with affected-shard incremental recomputation.
"""

from typing import Dict, List, Optional, Set, Tuple
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from portolan.candidate_graph import SpatioTemporalCandidateGraph, find_arrival_troughs
from portolan.query_eval import BaseDBEngine, links_to_trajectories


class MinCostNetFlowEngine(BaseDBEngine):
    """
    2. MinCost-NetFlow (`MCF-Tracker` when `sharded=False`; `MCF-Sharded` when `sharded=True`):
      - `sharded=False` (`MCF-Tracker`): Vectorized sparse CSR construction + global single-shot
        Jonker-Volgenant (`LAPJVsp`) minimum-cost node-disjoint path cover over all candidate edges.
      - `sharded=True` (`MCF-Sharded`): Shares the exact same `find_arrival_troughs` low-arrival
        blocks, Top-K (`K=5`) edge pruning, and causal boundary conflict-component reconciliation
        as `Prov-IVM`, isolating `Prov-IVM`'s three provenance mechanisms (conflict-SVD de-glaring,
        CPG `Q2` horizon operator, and CPG `Q4` alternating-path IVM).
    """

    name = "MCF-Tracker"

    def __init__(
        self,
        match_reward: float = 0.18,
        sharded: bool = False,
        top_k: int = 5,
    ):
        self.match_reward = match_reward
        self.sharded = sharded
        self.top_k = top_k
        if sharded:
            self.name = "MCF-Sharded"

        self.shards_: List[Tuple[int, int]] = []
        self.next_ptr_: Optional[np.ndarray] = None
        self.prev_ptr_: Optional[np.ndarray] = None
        self.topk_targets_: Optional[np.ndarray] = None
        self.topk_scores_: Optional[np.ndarray] = None

    def _build_topk_arrays(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Top-K shared fused edges per node into contiguous arrays `(N, K)`."""
        n = cgraph.n_snaps
        k_cap = self.top_k
        targets = np.full((n, k_cap), -1, dtype=np.int32)
        scores = np.full((n, k_cap), -1e4, dtype=np.float32)
        out_nbrs = cgraph.out_nbrs
        out_fused = cgraph.out_fused
        th = float(self.match_reward)

        for u in range(n):
            if active_mask is not None and not active_mask[u]:
                continue
            nbrs = out_nbrs[u]
            if len(nbrs) == 0:
                continue
            sc = out_fused[u]
            valid = sc > th
            if active_mask is not None:
                valid &= active_mask[nbrs]
            if not np.any(valid):
                continue
            v_cand = nbrs[valid]
            s_cand = sc[valid] - th
            m = len(s_cand)
            if m > k_cap:
                idx = np.argpartition(-s_cand, k_cap - 1)[:k_cap]
                idx = idx[np.argsort(-s_cand[idx])]
                v_cand = v_cand[idx]
                s_cand = s_cand[idx]
            else:
                idx = np.argsort(-s_cand)
                v_cand = v_cand[idx]
                s_cand = s_cand[idx]
            targets[u, : len(v_cand)] = v_cand
            scores[u, : len(s_cand)] = s_cand

        return targets, scores

    @staticmethod
    def _solve_shard_range(
        blocks: List[Tuple[int, int]],
        targets: np.ndarray,
        scores: np.ndarray,
        in_sources: List[List[int]],
        snap_time: np.ndarray,
        max_dt_sec: float,
        next_ptr: np.ndarray,
        prev_ptr: np.ndarray,
        n: int,
    ) -> None:
        """Solve causal arrival-trough blocks with boundary conflict-component / horizon reconciliation."""
        offset = 10.0
        use_conflict_comp = n > 300000

        for b_start, commit_end in blocks:
            if commit_end <= b_start:
                continue
            if len(blocks) == 1 and b_start == 0 and commit_end == n:
                # Single-block exact Top-K vectorized CSR assignment
                valid_e = scores > 0.0
                u_idx, k_idx = np.nonzero(valid_e)
                v_idx = targets[u_idx, k_idx]
                w_idx = offset + scores[u_idx, k_idx].astype(np.float64)
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
                continue

            if use_conflict_comp:
                sub_t = targets[b_start:commit_end]
                sub_s = scores[b_start:commit_end]
                has_out = (next_ptr[b_start:commit_end] < 0) & (sub_s[:, 0] > 0.0)
                if not np.any(has_out):
                    continue
                u_commit = (b_start + np.nonzero(has_out)[0]).tolist()
                v1_set: Set[int] = set()
                for u in u_commit:
                    for k_i in range(targets.shape[1]):
                        sc_v = float(scores[u, k_i])
                        if sc_v <= 0.0:
                            break
                        v_i = int(targets[u, k_i])
                        if prev_ptr[v_i] == -1:
                            v1_set.add(v_i)
                if not v1_set:
                    continue
                u_comp_set: Set[int] = set(u_commit)
                for v_i in v1_set:
                    for u_in in in_sources[v_i]:
                        if u_in >= commit_end and next_ptr[u_in] < 0:
                            u_comp_set.add(u_in)
                u_list = sorted(u_comp_set)
                u_arr = np.array(u_list, dtype=np.int32)
            else:
                t_commit = float(snap_time[commit_end - 1])
                u_end = int(np.searchsorted(snap_time, t_commit + max_dt_sec, side="right"))
                u_end = max(commit_end, min(n, u_end))
                u_arr = np.arange(b_start, u_end, dtype=np.int32)

            m_u = len(u_arr)
            if m_u == 0:
                continue
            unmatched_u = next_ptr[u_arr] < 0
            t_mat = targets[u_arr]
            s_mat = scores[u_arr]
            valid_m = unmatched_u[:, None] & (s_mat > 0.0)
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

    def recover_trajectories(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], np.ndarray, float]:
        """Solve minimum-cost circulation via vectorized global CSR or arrival-trough Top-K shards."""
        t0 = time.perf_counter()
        n = cgraph.n_snaps
        next_ptr = np.full(n, -1, dtype=np.int32)

        if not self.sharded:
            # Vectorized NumPy sparse CSR construction (0 pure-Python inner edge loops!)
            offset = 10.0
            dummy_w = offset + float(self.match_reward)
            out_nbrs = cgraph.out_nbrs
            out_fused = cgraph.out_fused

            if active_mask is None:
                degs = np.array([len(nb) for nb in out_nbrs], dtype=np.int32)
                non_empty = np.nonzero(degs > 0)[0]
                if len(non_empty) > 0:
                    all_u = np.repeat(non_empty.astype(np.int32), degs[non_empty])
                    all_v = np.concatenate([out_nbrs[i] for i in non_empty])
                    all_sc = np.concatenate([out_fused[i] for i in non_empty])
                    keep = all_sc > self.match_reward
                    e_u = all_u[keep]
                    e_v = all_v[keep]
                    e_w = offset + all_sc[keep].astype(np.float64)
                else:
                    e_u = np.empty(0, dtype=np.int32)
                    e_v = np.empty(0, dtype=np.int32)
                    e_w = np.empty(0, dtype=np.float64)
            else:
                u_chunks, v_chunks, w_chunks = [], [], []
                for u in range(n):
                    if not active_mask[u] or len(out_nbrs[u]) == 0:
                        continue
                    nbrs = out_nbrs[u]
                    sc = out_fused[u]
                    keep = (sc > self.match_reward) & active_mask[nbrs]
                    if np.any(keep):
                        m_k = int(np.sum(keep))
                        u_chunks.append(np.full(m_k, u, dtype=np.int32))
                        v_chunks.append(nbrs[keep])
                        w_chunks.append(offset + sc[keep].astype(np.float64))
                if u_chunks:
                    e_u = np.concatenate(u_chunks)
                    e_v = np.concatenate(v_chunks)
                    e_w = np.concatenate(w_chunks)
                else:
                    e_u = np.empty(0, dtype=np.int32)
                    e_v = np.empty(0, dtype=np.int32)
                    e_w = np.empty(0, dtype=np.float64)

            ar = np.arange(n, dtype=np.int32)
            rows = np.concatenate([e_u, ar])
            cols = np.concatenate([e_v, n + ar])
            weights = np.concatenate([e_w, np.full(n, dummy_w, dtype=np.float64)])

            biadj = csr_matrix((weights, (rows, cols)), shape=(n, 2 * n))
            r_ind, c_ind = min_weight_full_bipartite_matching(biadj, maximize=True)
            matched = c_ind < n
            next_ptr[r_ind[matched].astype(np.int32)] = c_ind[matched].astype(np.int32)

            trajs = links_to_trajectories(next_ptr, active_mask=active_mask)
            elapsed = time.perf_counter() - t0
            return trajs, next_ptr, elapsed

        # Sharded Top-K arrival-trough solver (`MCF-Sharded`)
        targets, scores = self._build_topk_arrays(cgraph, active_mask=active_mask)
        snap_time = cgraph.dataset["snap_time"]
        target_bs = 4000 if n > 300000 else 12000
        blocks = find_arrival_troughs(snap_time, target_block_size=target_bs, min_shard_n=60000)

        in_sources: List[List[int]] = [[] for _ in range(n)]
        if n > 300000:
            u_nz, k_nz = np.nonzero(scores > 0.0)
            v_nz = targets[u_nz, k_nz]
            for u_i, v_i in zip(u_nz.tolist(), v_nz.tolist()):
                in_sources[v_i].append(u_i)

        prev_ptr = np.full(n, -1, dtype=np.int32)
        self._solve_shard_range(
            blocks=blocks,
            targets=targets,
            scores=scores,
            in_sources=in_sources,
            snap_time=snap_time,
            max_dt_sec=cgraph.max_dt_sec,
            next_ptr=next_ptr,
            prev_ptr=prev_ptr,
            n=n,
        )

        # Cache fine-grained arrival-trough shards (`<= 4,000` tuples) for affected-shard Q4 comparison
        self.shards_ = (
            blocks
            if len(blocks) > 1
            else find_arrival_troughs(snap_time, target_block_size=4000, min_shard_n=4000)
        )
        self.next_ptr_ = next_ptr.copy()
        self.prev_ptr_ = prev_ptr.copy()
        self.topk_targets_ = targets
        self.topk_scores_ = scores

        trajs = links_to_trajectories(next_ptr, active_mask=active_mask)
        elapsed = time.perf_counter() - t0
        return trajs, next_ptr, elapsed

    def recompute_affected_shards(
        self,
        cgraph: SpatioTemporalCandidateGraph,
        mutated_nodes: Set[int],
        active_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[int]], np.ndarray, float]:
        """
        Re-solve ONLY the arrival-trough shards impacted by `mutated_nodes` (deletions or
        re-weighted sources), providing a strong affected-shard `MCF-Sharded` baseline for Q4 IVM.
        """
        if self.next_ptr_ is None or len(self.next_ptr_) != cgraph.n_snaps:
            self.recover_trajectories(cgraph)

        t0 = time.perf_counter()
        n = cgraph.n_snaps
        snap_time = cgraph.dataset["snap_time"]
        shards = self.shards_
        shard_starts = np.array([b[0] for b in shards], dtype=np.int32)

        next_work = self.next_ptr_.copy()
        prev_work = self.prev_ptr_.copy()
        targets = self.topk_targets_.copy()
        scores = self.topk_scores_.copy()

        # Identify every node whose outgoing Top-K edges or matched links touch `mutated_nodes`
        dirty_nodes: Set[int] = set(mutated_nodes)
        for r in mutated_nodes:
            if 0 <= r < n:
                p_u = int(prev_work[r])
                if p_u >= 0:
                    dirty_nodes.add(p_u)
                n_v = int(next_work[r])
                if n_v >= 0:
                    dirty_nodes.add(n_v)

        if active_mask is not None:
            # Deletion / quarantine: invalidate deleted rows and edges pointing to deleted targets
            rem_arr = np.array(sorted(mutated_nodes), dtype=np.int32)
            if len(rem_arr) > 0:
                scores[rem_arr, :] = -1e4
                hit_mask = np.isin(targets, rem_arr) & (scores > 0.0)
                hit_u = np.nonzero(np.any(hit_mask, axis=1))[0]
                scores[hit_mask] = -1e4
                dirty_nodes.update(hit_u.tolist())
        else:
            # Clock calibration: refresh Top-K rows for `mutated_nodes` from updated `cgraph.out_fused`
            th = float(self.match_reward)
            k_cap = self.top_k
            for u in mutated_nodes:
                targets[u, :] = -1
                scores[u, :] = -1e4
                nbrs = cgraph.out_nbrs[u]
                if len(nbrs) == 0:
                    continue
                sc = cgraph.out_fused[u]
                valid = sc > th
                if np.any(valid):
                    v_c = nbrs[valid]
                    s_c = sc[valid] - th
                    idx = np.argsort(-s_c)[:k_cap]
                    targets[u, : len(idx)] = v_c[idx]
                    scores[u, : len(idx)] = s_c[idx]

        # Map dirty nodes to affected shards (+1 successor shard for cross-boundary lookahead consistency)
        dirty_shard_ids: Set[int] = set()
        for dn in dirty_nodes:
            s_id = int(np.searchsorted(shard_starts, dn, side="right")) - 1
            s_id = max(0, min(len(shards) - 1, s_id))
            dirty_shard_ids.add(s_id)
            if s_id + 1 < len(shards):
                dirty_shard_ids.add(s_id + 1)

        aff_blocks = [shards[i] for i in sorted(dirty_shard_ids)]
        # Clear existing assignments originating in affected shards before re-solving them
        for b_s, b_e in aff_blocks:
            old_targets = next_work[b_s:b_e]
            valid_old = old_targets >= 0
            if np.any(valid_old):
                prev_work[old_targets[valid_old]] = -1
            next_work[b_s:b_e] = -1
        if active_mask is not None:
            for r in mutated_nodes:
                if 0 <= r < n:
                    if prev_work[r] >= 0:
                        next_work[prev_work[r]] = -1
                        prev_work[r] = -1
                    if next_work[r] >= 0:
                        prev_work[next_work[r]] = -1
                        next_work[r] = -1

        in_sources: List[List[int]] = [[] for _ in range(n)]
        if n > 300000:
            for b_s, b_e in aff_blocks:
                hi_e = min(n, b_e + 4500)
                u_nz, k_nz = np.nonzero(scores[b_s:hi_e] > 0.0)
                u_abs = b_s + u_nz
                v_nz = targets[u_abs, k_nz]
                for u_i, v_i in zip(u_abs.tolist(), v_nz.tolist()):
                    in_sources[v_i].append(u_i)

        self._solve_shard_range(
            blocks=aff_blocks,
            targets=targets,
            scores=scores,
            in_sources=in_sources,
            snap_time=snap_time,
            max_dt_sec=cgraph.max_dt_sec,
            next_ptr=next_work,
            prev_ptr=prev_work,
            n=n,
        )
        t_solve = time.perf_counter() - t0
        trajs = links_to_trajectories(next_work, active_mask=active_mask)
        t_total = time.perf_counter() - t0
        self.last_affected_shard_info = {
            "total_shards": len(shards),
            "affected_shards": len(aff_blocks),
            "affected_shard_pct": 100.0 * len(aff_blocks) / max(1, len(shards)),
            "shard_solve_ms": t_solve * 1000.0,
            "total_with_view_ms": t_total * 1000.0,
        }
        return trajs, next_work, float(t_total)
