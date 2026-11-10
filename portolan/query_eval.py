"""
Shared Engine Interface & Declarative Multi-Modal Query Evaluators (Q1, Q2, Q3)
for the Portolan Cross-Camera Trajectory Database Benchmark Suite.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from portolan.candidate_graph import SpatioTemporalCandidateGraph


def links_to_trajectories(next_ptr: np.ndarray, active_mask: Optional[np.ndarray] = None) -> List[List[int]]:
    """Convert 1-to-1 or many-to-1 next_ptr array (size N, -1 for end) into trajectory chains."""
    n = len(next_ptr)
    has_pred = np.zeros(n, dtype=bool)
    for u in range(n):
        if active_mask is not None and not active_mask[u]:
            continue
        v = int(next_ptr[u])
        if v >= 0 and (active_mask is None or active_mask[v]):
            has_pred[v] = True

    trajs: List[List[int]] = []
    visited = np.zeros(n, dtype=bool)
    for u in range(n):
        if active_mask is not None and not active_mask[u]:
            continue
        if not has_pred[u] and not visited[u]:
            chain = []
            cur = u
            while cur >= 0 and not visited[cur]:
                if active_mask is not None and not active_mask[cur]:
                    break
                visited[cur] = True
                chain.append(cur)
                cur = int(next_ptr[cur])
            if chain:
                trajs.append(chain)
    return trajs


def evaluate_q1_from_trajs(trajs: List[List[int]], n_snaps: int, q1_queries: List[Dict]) -> float:
    """Evaluate Point Trajectory Recovery (Q1) average F1 from materialized trajectories."""
    if not q1_queries:
        return 0.0
    snap_to_traj: Dict[int, List[int]] = {}
    for tr in trajs:
        for s in tr:
            snap_to_traj[s] = tr

    f1_list = []
    for q in q1_queries:
        seed_s = q["seed_snapshot_id"]
        gt_set = set(q["gt_snapshot_ids"])
        pred_set = set(snap_to_traj.get(seed_s, [seed_s]))
        tp = len(gt_set & pred_set)
        prec = tp / max(1, len(pred_set))
        rec = tp / max(1, len(gt_set))
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    return float(np.mean(f1_list))


def evaluate_q2_track_pairs(
    track_pairs: List[Tuple[List[int], List[int]]],
    snap_cam: np.ndarray,
    snap_veh: np.ndarray,
    k_hops: int,
    gt_pairs: Set[Tuple[int, int]],
) -> Dict[str, float]:
    """
    Unified, strict entity-level scorer for Zero-Image Co-Movement Relational Join (Q2).
    Used identically by BOTH eager trajectory engines and Prov-IVM's coupled pair-provenance join:
      - Assigns each predicted track side to its majority ground-truth vehicle ID (mode).
      - Counts a true positive iff the two majority vehicles are distinct, belong to gt_pairs,
        and both majority vehicles are genuinely recovered across >= k_hops shared cameras.
      - Counts self-split tracks (vi == vj) and drifted chains (< k_hops true shared cameras)
        strictly as false positives.
      - Returns 0.0 (never 1.0) if gt_pairs is empty.
    """
    if len(gt_pairs) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_veh_pairs: Set[Tuple[int, int]] = set()
    for pair_idx, (tr_i, tr_j) in enumerate(track_pairs):
        if len(tr_i) < k_hops or len(tr_j) < k_hops:
            continue
        vehs_i = [int(snap_veh[s]) for s in tr_i]
        vals_i, cnts_i = np.unique(vehs_i, return_counts=True)
        vi = int(vals_i[int(np.argmax(cnts_i))])

        vehs_j = [int(snap_veh[s]) for s in tr_j]
        vals_j, cnts_j = np.unique(vehs_j, return_counts=True)
        vj = int(vals_j[int(np.argmax(cnts_j))])

        if vi == vj:
            pred_veh_pairs.add((-pair_idx - 1, -pair_idx - 1))
            continue

        cams_vi = {int(snap_cam[s]) for s in tr_i if int(snap_veh[s]) == vi}
        cams_vj = {int(snap_cam[s]) for s in tr_j if int(snap_veh[s]) == vj}
        p_ord = (vi, vj) if vi < vj else (vj, vi)
        if len(cams_vi & cams_vj) >= k_hops and p_ord in gt_pairs:
            pred_veh_pairs.add(p_ord)
        else:
            pred_veh_pairs.add((-pair_idx - 1, -pair_idx - 1))

    tp = len(pred_veh_pairs & gt_pairs)
    prec = tp / max(1, len(pred_veh_pairs))
    rec = tp / max(1, len(gt_pairs))
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def evaluate_q2_eager_from_trajs(
    trajs: List[List[int]],
    snap_cam: np.ndarray,
    snap_time: np.ndarray,
    snap_veh: np.ndarray,
    k_hops: int,
    delta_t: float,
    gt_pairs: Set[Tuple[int, int]],
) -> Dict[str, float]:
    """
    Execute Zero-Image Co-Movement Relational Join (Q2) over eagerly materialized trajectories
    and evaluate Precision, Recall, and F1 via the unified majority-entity scorer.
    """
    cam_visits: Dict[int, List[Tuple[float, int]]] = {}
    for tid, tr in enumerate(trajs):
        if len(tr) < k_hops:
            continue
        seen_c = set()
        for s in tr:
            c = int(snap_cam[s])
            if c not in seen_c:
                seen_c.add(c)
                cam_visits.setdefault(c, []).append((float(snap_time[s]), tid))

    pair_counts: Dict[Tuple[int, int], int] = {}
    for c, visits in cam_visits.items():
        visits.sort(key=lambda x: x[0])
        m = len(visits)
        for i in range(m):
            ti, tid_i = visits[i]
            j = i + 1
            while j < m and (visits[j][0] - ti) <= delta_t:
                tid_j = visits[j][1]
                if tid_i != tid_j:
                    pair = (tid_i, tid_j) if tid_i < tid_j else (tid_j, tid_i)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                j += 1

    matched_track_pairs = [
        (trajs[ti], trajs[tj]) for (ti, tj), cnt in pair_counts.items() if cnt >= k_hops
    ]
    return evaluate_q2_track_pairs(matched_track_pairs, snap_cam, snap_veh, k_hops, gt_pairs)


def evaluate_q3_from_trajs(
    trajs: List[List[int]], snap_cam: np.ndarray, q3_workload: Dict
) -> Dict[str, float]:
    """Evaluate Corridor OD Flow & Bottleneck Aggregation Query (Q3) accuracy."""
    gt_od: Dict[Tuple[int, int], int] = q3_workload["gt_od_counts"]
    gt_top_bn: List[Tuple[int, int, int]] = q3_workload["gt_top_bottlenecks"]

    pred_od: Dict[Tuple[int, int], int] = {}
    pred_cor3: Dict[Tuple[int, int, int], int] = {}
    for tr in trajs:
        cams = [int(snap_cam[s]) for s in tr]
        for i in range(len(cams) - 1):
            if cams[i] != cams[i + 1]:
                e = (cams[i], cams[i + 1])
                pred_od[e] = pred_od.get(e, 0) + 1
        for i in range(len(cams) - 2):
            if cams[i] != cams[i + 1] and cams[i + 1] != cams[i + 2]:
                tri = (cams[i], cams[i + 1], cams[i + 2])
                pred_cor3[tri] = pred_cor3.get(tri, 0) + 1

    all_keys = set(gt_od.keys()) | set(pred_od.keys())
    abs_err = sum(abs(gt_od.get(k, 0) - pred_od.get(k, 0)) for k in all_keys)
    tot_gt = max(1, sum(gt_od.values()))
    flow_acc = max(0.0, 1.0 - abs_err / (1.5 * tot_gt))

    pred_top_bn = [
        c for c, _ in sorted(pred_cor3.items(), key=lambda x: x[1], reverse=True)[: max(1, len(gt_top_bn))]
    ]
    bn_recall = len(set(pred_top_bn) & set(gt_top_bn)) / max(1, len(gt_top_bn))
    return {
        "flow_accuracy": float(flow_acc),
        "bottleneck_recall": float(bn_recall),
        "q3_score": float(0.6 * flow_acc + 0.4 * bn_recall),
    }


class BaseDBEngine:
    """Abstract base interface for cross-camera trajectory and vector database engines."""

    name: str = "BaseEngine"

    def recover_trajectories(
        self, cgraph: SpatioTemporalCandidateGraph, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], np.ndarray, float]:
        """Recover ordered entity trajectories, successor pointers, and wall-clock latency (seconds)."""
        raise NotImplementedError
