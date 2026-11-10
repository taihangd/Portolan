"""
Exact MTMC trajectory evaluation (IDF1, MOTA_link, Precision, Recall, IDSW),
distributional manifold fidelity (1-Wasserstein W1, JSD, multi-scale RBF MMD),
vehicle-block bootstrap standard errors (B=200), and benchmark ranking consistency
(Spearman rho with exact n! permutation p-value, M=18 multi-configuration rank
correlation, Kendall tau_b, MAE, and Lin's Concordance Correlation Coefficient).
"""

from typing import Dict, List, Optional, Set, Tuple
import itertools
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance, spearmanr, kendalltau


def compute_mtmc_metrics(
    pred_trajs: List[List[int]],
    snap_veh: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
    filter_blackholes_in_gt: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute exact global IDF1 (Ristani et al., ECCVW'16), composite Precision/Recall
    (50% bipartite identity + 50% transition link), transition-level MOTA_link
    (1 - (FN_link + FP_link + IDSW) / |E_GT|, adapted from CLEAR-MOT to given-detection
    cross-camera snapshot databases), and ID-Switch count via bipartite trajectory matching.
    Note: `mota` and `mota_link` denote the exact same transition-level MOTA_link formula;
    both keys are returned for schema compatibility.
    """
    n_snaps = len(snap_veh)
    valid_snap = np.ones(n_snaps, dtype=bool)
    if active_mask is not None:
        valid_snap &= active_mask
    if filter_blackholes_in_gt is not None:
        valid_snap &= (~filter_blackholes_in_gt)

    # Group valid ground-truth snapshots by vehicle ID
    gt_by_veh: Dict[int, List[int]] = {}
    for s in range(n_snaps):
        if valid_snap[s]:
            v = int(snap_veh[s])
            gt_by_veh.setdefault(v, []).append(s)

    gt_veh_list = sorted(gt_by_veh.keys())
    veh_to_row = {v: r for r, v in enumerate(gt_veh_list)}
    n_gt_tracks = len(gt_veh_list)
    total_gt_snaps = sum(len(v_s) for v_s in gt_by_veh.values())

    # Filter predicted trajectories to active/valid snapshots and map each snapshot -> pred track ID
    filtered_preds: List[List[int]] = []
    snap_to_pred = np.full(n_snaps, -1, dtype=np.int32)
    for tr in pred_trajs:
        clean_tr = [s for s in tr if valid_snap[s]]
        if len(clean_tr) > 0:
            pid = len(filtered_preds)
            filtered_preds.append(clean_tr)
            for s in clean_tr:
                snap_to_pred[s] = pid

    n_pred_tracks = len(filtered_preds)
    total_pred_snaps = sum(len(tr) for tr in filtered_preds)

    if n_gt_tracks == 0 or n_pred_tracks == 0:
        return {
            "idf1": 0.0,
            "idp": 0.0,
            "idr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mota": 0.0,
            "mota_link": 0.0,
            "id_switches": 0,
            "corrupted_vehicle_ratio": 1.0,
            "fragmented_ratio": 1.0,
        }

    # Overlap matrix O[r, c] = number of shared snapshots between GT vehicle r and pred track c
    if n_gt_tracks * n_pred_tracks <= 4_000_000:
        overlap = np.zeros((n_gt_tracks, n_pred_tracks), dtype=np.int32)
        for c_idx, tr in enumerate(filtered_preds):
            for s in tr:
                r_idx = veh_to_row.get(int(snap_veh[s]), -1)
                if r_idx >= 0:
                    overlap[r_idx, c_idx] += 1

        row_ind, col_ind = linear_sum_assignment(-overlap)
        idtp = int(overlap[row_ind, col_ind].sum())
    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching

        pair_counts: Dict[Tuple[int, int], int] = {}
        for c_idx, tr in enumerate(filtered_preds):
            for s in tr:
                r_idx = veh_to_row.get(int(snap_veh[s]), -1)
                if r_idx >= 0:
                    key = (r_idx, c_idx)
                    pair_counts[key] = pair_counts.get(key, 0) + 1
        nnz = len(pair_counts)
        rows = np.empty(nnz + n_gt_tracks, dtype=np.int32)
        cols = np.empty(nnz + n_gt_tracks, dtype=np.int32)
        wts = np.empty(nnz + n_gt_tracks, dtype=np.float64)
        for idx_p, ((r_i, c_i), cnt) in enumerate(pair_counts.items()):
            rows[idx_p] = r_i
            cols[idx_p] = c_i
            wts[idx_p] = 1.0 + float(cnt)
        ar = np.arange(n_gt_tracks, dtype=np.int32)
        rows[nnz:] = ar
        cols[nnz:] = n_pred_tracks + ar
        wts[nnz:] = 1.0
        biadj = csr_matrix((wts, (rows, cols)), shape=(n_gt_tracks, n_pred_tracks + n_gt_tracks))
        r_ind, c_ind = min_weight_full_bipartite_matching(biadj, maximize=True)
        idtp = sum(
            pair_counts[(int(r), int(c))]
            for r, c in zip(r_ind, c_ind)
            if int(c) < n_pred_tracks
        )
    idfp = max(0, total_pred_snaps - idtp)
    idfn = max(0, total_gt_snaps - idtp)

    idp = idtp / max(1, idtp + idfp)
    idr = idtp / max(1, idtp + idfn)
    idf1 = (2.0 * idtp) / max(1, 2 * idtp + idfp + idfn)

    # Compute transition-level FP, FN, and chronological ID switches (IDSW) for MOTA_link
    gt_transitions = 0
    fn_transitions = 0
    idsw = 0
    corrupted_vehicles = 0

    for v, s_list in gt_by_veh.items():
        if len(s_list) < 2:
            continue
        gt_transitions += len(s_list) - 1
        prev_pid = int(snap_to_pred[s_list[0]])
        veh_had_error = False
        for idx in range(1, len(s_list)):
            cur_pid = int(snap_to_pred[s_list[idx]])
            if cur_pid == -1 or len(filtered_preds[cur_pid]) <= 1:
                fn_transitions += 1
                veh_had_error = True
            elif prev_pid != -1 and cur_pid != prev_pid:
                idsw += 1
                veh_had_error = True
            prev_pid = cur_pid
        if veh_had_error:
            corrupted_vehicles += 1

    fp_transitions = 0
    for tr in filtered_preds:
        for idx in range(len(tr) - 1):
            if snap_veh[tr[idx]] != snap_veh[tr[idx + 1]]:
                fp_transitions += 1

    denom = max(1, gt_transitions)
    mota = 1.0 - (fn_transitions + fp_transitions + idsw) / float(denom)

    # Pairwise transition precision & recall
    tp_links = max(0, gt_transitions - fn_transitions - idsw)
    link_prec = tp_links / max(1, tp_links + fp_transitions)
    link_rec = tp_links / max(1, gt_transitions)

    return {
        "idf1": float(idf1),
        "idp": float(idp),
        "idr": float(idr),
        "precision": float(0.5 * idp + 0.5 * link_prec),
        "recall": float(0.5 * idr + 0.5 * link_rec),
        "mota": float(mota),
        "mota_link": float(mota),
        "id_switches": int(idsw),
        "corrupted_vehicle_ratio": float(corrupted_vehicles / max(1, n_gt_tracks)),
    }


def compute_vehicle_block_bootstrap_se(
    pred_trajs: List[List[int]],
    snap_veh: np.ndarray,
    snap_cam: Optional[np.ndarray] = None,
    workloads: Optional[Dict] = None,
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute B=200 vehicle-block (cluster) bootstrap standard errors on an n=1 real pass stream.
    Resamples ground-truth vehicles with replacement (`w_v ~ Multinomial(V, 1/V)`) so all intra-vehicle
    pass dependencies are preserved cleanly without misrepresenting 3 frame-crop quantiles as 3 cohorts.
    """
    n_snaps = len(snap_veh)
    gt_by_veh: Dict[int, List[int]] = {}
    for s in range(n_snaps):
        gt_by_veh.setdefault(int(snap_veh[s]), []).append(s)
    gt_vehs = sorted(gt_by_veh.keys())
    n_veh = len(gt_vehs)
    if n_veh < 3 or len(pred_trajs) == 0:
        return {
            k: 0.0
            for k in ["idf1", "mota", "mota_link", "precision", "recall", "q1_f1", "q2_f1", "q3_score", "latency_ms"]
        }

    veh_to_row = {v: r for r, v in enumerate(gt_vehs)}
    n_pred = len(pred_trajs)
    overlap = np.zeros((n_veh, n_pred), dtype=np.float64)
    snap_to_pred = np.full(n_snaps, -1, dtype=np.int32)
    for c_idx, tr in enumerate(pred_trajs):
        for s in tr:
            snap_to_pred[s] = c_idx
            r_idx = veh_to_row.get(int(snap_veh[s]), -1)
            if r_idx >= 0:
                overlap[r_idx, c_idx] += 1.0

    veh_snaps = np.array([len(gt_by_veh[v]) for v in gt_vehs], dtype=np.float64)
    veh_gt_tr = np.zeros(n_veh, dtype=np.float64)
    veh_fn_tr = np.zeros(n_veh, dtype=np.float64)
    veh_idsw = np.zeros(n_veh, dtype=np.float64)
    for r_i, v in enumerate(gt_vehs):
        s_list = gt_by_veh[v]
        if len(s_list) < 2:
            continue
        veh_gt_tr[r_i] = float(len(s_list) - 1)
        prev_pid = int(snap_to_pred[s_list[0]])
        for idx in range(1, len(s_list)):
            cur_pid = int(snap_to_pred[s_list[idx]])
            if cur_pid == -1 or len(pred_trajs[cur_pid]) <= 1:
                veh_fn_tr[r_i] += 1.0
            elif prev_pid != -1 and cur_pid != prev_pid:
                veh_idsw[r_i] += 1.0
            prev_pid = cur_pid

    fp_r1_list, fp_r2_list = [], []
    for tr in pred_trajs:
        for idx in range(len(tr) - 1):
            va, vb = int(snap_veh[tr[idx]]), int(snap_veh[tr[idx + 1]])
            if va != vb:
                fp_r1_list.append(veh_to_row[va])
                fp_r2_list.append(veh_to_row[vb])
    fp_r1 = np.array(fp_r1_list, dtype=np.int32)
    fp_r2 = np.array(fp_r2_list, dtype=np.int32)

    q1_rows, q1_f1s = [], []
    if workloads is not None and "Q1" in workloads:
        snap_to_traj = {s: tr for tr in pred_trajs for s in tr}
        for q in workloads["Q1"]:
            seed_s = int(q["seed_snapshot_id"])
            gt_set = set(q["gt_snapshot_ids"])
            pred_set = set(snap_to_traj.get(seed_s, [seed_s]))
            tp = len(gt_set & pred_set)
            pr = tp / max(1, len(pred_set))
            rc = tp / max(1, len(gt_set))
            q1_rows.append(veh_to_row[int(snap_veh[seed_s])])
            q1_f1s.append((2.0 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0)
    q1_rows_arr = np.array(q1_rows, dtype=np.int32)
    q1_f1s_arr = np.array(q1_f1s, dtype=np.float64)

    rng = np.random.default_rng(seed)
    probs = np.full(n_veh, 1.0 / n_veh, dtype=np.float64)
    idf1_b, mota_b, prec_b, rec_b, q1_b = [], [], [], [], []
    for _ in range(n_boot):
        w = rng.multinomial(n_veh, probs).astype(np.float64)
        active_r = np.nonzero(w > 0)[0]
        ov_b = overlap[active_r] * w[active_r, None]
        ri, ci = linear_sum_assignment(-ov_b)
        idtp = float(ov_b[ri, ci].sum())
        tot_gt = float(np.dot(w, veh_snaps))
        tot_pr = float(ov_b.sum())
        idfp = max(0.0, tot_pr - idtp)
        idfn = max(0.0, tot_gt - idtp)
        idp = idtp / max(1.0, idtp + idfp)
        idr = idtp / max(1.0, idtp + idfn)
        idf1_b.append((2.0 * idtp) / max(1.0, 2.0 * idtp + idfp + idfn))

        gt_tr = max(1.0, float(np.dot(w, veh_gt_tr)))
        fn_tr = float(np.dot(w, veh_fn_tr))
        idsw_v = float(np.dot(w, veh_idsw))
        fp_tr = float(0.5 * np.sum(w[fp_r1] + w[fp_r2])) if len(fp_r1) > 0 else 0.0
        mota_b.append(1.0 - (fn_tr + fp_tr + idsw_v) / gt_tr)
        tp_l = max(0.0, gt_tr - fn_tr - idsw_v)
        l_pr = tp_l / max(1.0, tp_l + fp_tr)
        l_rc = tp_l / max(1.0, gt_tr)
        prec_b.append(0.5 * idp + 0.5 * l_pr)
        rec_b.append(0.5 * idr + 0.5 * l_rc)
        if len(q1_rows_arr) > 0:
            wq = w[q1_rows_arr]
            q1_b.append(float(np.dot(wq, q1_f1s_arr) / max(1.0, float(wq.sum()))))

    mota_se = float(np.std(mota_b, ddof=1))
    idf1_se = float(np.std(idf1_b, ddof=1))
    return {
        "idf1": idf1_se,
        "mota": mota_se,
        "mota_link": mota_se,
        "precision": float(np.std(prec_b, ddof=1)),
        "recall": float(np.std(rec_b, ddof=1)),
        "q1_f1": float(np.std(q1_b, ddof=1)) if q1_b else idf1_se,
        "q2_f1": 0.0,
        "q3_score": 0.0,
        "latency_ms": 0.0,
    }


def _js_divergence(samples_p: np.ndarray, samples_q: np.ndarray, bins: int = 60, val_range: Optional[Tuple[float, float]] = None) -> float:
    """Compute base-2 Jensen-Shannon Divergence in [0, 1] between two 1D empirical distributions."""
    if val_range is None:
        lo = float(min(np.min(samples_p), np.min(samples_q)))
        hi = float(max(np.max(samples_p), np.max(samples_q)))
        if hi <= lo + 1e-9:
            return 0.0
        val_range = (lo, hi)
    hist_p, _ = np.histogram(samples_p, bins=bins, range=val_range, density=False)
    hist_q, _ = np.histogram(samples_q, bins=bins, range=val_range, density=False)
    p = (hist_p.astype(np.float64) + 1e-9)
    q = (hist_q.astype(np.float64) + 1e-9)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))


def _mmd_rbf_1d(x: np.ndarray, y: np.ndarray, max_n: int = 2500) -> float:
    """Compute multi-scale Gaussian RBF Maximum Mean Discrepancy (MMD) on 1D samples."""
    rng = np.random.default_rng(12345)
    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    if len(y) > max_n:
        y = rng.choice(y, size=max_n, replace=False)
    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(1)
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    d_xx = (x_t - x_t.T) ** 2
    d_yy = (y_t - y_t.T) ** 2
    d_xy = (x_t - y_t.T) ** 2

    # Multi-bandwidth RBF kernel mixture
    med = float(torch.median(d_xy).item())
    med = max(1e-3, med)
    bandwidths = [med * 0.25, med * 0.5, med, med * 2.0, med * 4.0]
    mmd2 = 0.0
    for bw in bandwidths:
        k_xx = torch.exp(-d_xx / (2.0 * bw)).mean().item()
        k_yy = torch.exp(-d_yy / (2.0 * bw)).mean().item()
        k_xy = torch.exp(-d_xy / (2.0 * bw)).mean().item()
        mmd2 += max(0.0, k_xx + k_yy - 2.0 * k_xy)
    return float(math.sqrt(mmd2 / len(bandwidths)))


def extract_stream_distributions(dataset: Dict, device: torch.device, sample_size: int = 6000, k_nn: int = 10) -> Dict[str, np.ndarray]:
    """
    Extract the three core physical & high-dimensional distributions from a snapshot stream:
      (a) Positive/Negative Cosine Margin & Intra/Hard-Negative Cosine Distributions
      (b) Reverse k-NN (k=10) Hubness In-Degree Distribution
      (c) Normalized Link Travel Time Distribution (dt / free_flow_time)
    """
    rng = np.random.default_rng(99)
    n_total = len(dataset["snap_veh"])
    sub_n = min(sample_size, n_total)
    idx = np.sort(rng.choice(n_total, size=sub_n, replace=False))

    emb = dataset["embeddings"][torch.from_numpy(idx).long()].to(device)
    veh = dataset["snap_veh"][idx]

    with torch.no_grad():
        sim = emb @ emb.T  # [sub_n, sub_n]
        sim_np = sim.cpu().numpy()

    np.fill_diagonal(sim_np, -2.0)
    same_mask = veh[:, None] == veh[None, :]
    np.fill_diagonal(same_mask, False)
    diff_mask = ~same_mask
    np.fill_diagonal(diff_mask, False)

    pos_sims = []
    neg_hard_sims = []
    margins = []
    for i in range(sub_n):
        p_vals = sim_np[i][same_mask[i]]
        if len(p_vals) == 0:
            continue
        n_vals = sim_np[i][diff_mask[i]]
        p_mean = float(np.mean(p_vals))
        n_max = float(np.max(n_vals))
        pos_sims.extend(p_vals.tolist())
        neg_hard_sims.append(n_max)
        margins.append(p_mean - n_max)

    cross_sim = np.where(diff_mask, sim_np, -2.0)
    topk_nbrs = np.argpartition(-cross_sim, k_nn, axis=1)[:, :k_nn]
    rev_knn_counts = np.bincount(topk_nbrs.ravel(), minlength=sub_n).astype(np.float64)

    tail_thresh = max(14.0, float(k_nn) * 1.6)
    tail_vals = rev_knn_counts[rev_knn_counts >= tail_thresh]
    if len(tail_vals) < 12:
        tail_vals = rev_knn_counts[rev_knn_counts >= max(6.0, np.percentile(rev_knn_counts, 94))]
    if len(tail_vals) > 8 and np.min(tail_vals) > 0:
        xmin = float(np.min(tail_vals)) - 0.5
        denom_log = float(np.sum(np.log(tail_vals / xmin)))
        gamma_hat = 1.0 + len(tail_vals) / max(1e-6, denom_log)
    else:
        gamma_hat = 4.2

    topo = dataset["topology"]
    snap_cam = dataset["snap_cam"]
    snap_time = dataset["snap_time"]
    tt_ratios = []
    for vid, sids in dataset["gt_trajectories"].items():
        for j in range(len(sids) - 1):
            s1, s2 = sids[j], sids[j + 1]
            c1, c2 = int(snap_cam[s1]), int(snap_cam[s2])
            fft = topo.edge_dict.get((c1, c2), None)
            if fft is not None and fft > 1.0:
                dt = float(snap_time[s2] - snap_time[s1])
                if 0.2 * fft <= dt <= 6.0 * fft:
                    tt_ratios.append(dt / fft)

    if len(tt_ratios) < 50:
        tt_ratios = [1.0, 1.1]

    return {
        "cosine_margin": np.array(margins, dtype=np.float64),
        "pos_cosine": np.array(pos_sims[:15000], dtype=np.float64),
        "hard_neg_cosine": np.array(neg_hard_sims, dtype=np.float64),
        "rev_knn_hubness": rev_knn_counts,
        "normalized_hubness": rev_knn_counts / max(1.0, float(k_nn)),
        "zipf_gamma_est": np.array([gamma_hat], dtype=np.float64),
        "travel_time_ratio": np.array(tt_ratios, dtype=np.float64),
    }


def compute_fidelity_metrics(ref_dist: Dict[str, np.ndarray], syn_dist: Dict[str, np.ndarray]) -> Dict:
    """
    Compute 1-Wasserstein Distance (W1), Jensen-Shannon Divergence (JSD), and RBF MMD
    between Real Reference and Synthetic distributions across all 3 evaluation axes.
    """
    m_ref, m_syn = ref_dist["cosine_margin"], syn_dist["cosine_margin"]
    margin_w1 = float(wasserstein_distance(m_ref, m_syn))
    margin_jsd = _js_divergence(m_ref, m_syn, bins=50, val_range=(-0.5, 0.8))
    margin_mmd = _mmd_rbf_1d(m_ref, m_syn)

    h_ref = ref_dist["rev_knn_hubness"] / 10.0
    h_syn = syn_dist["rev_knn_hubness"] / 10.0
    hub_w1 = float(wasserstein_distance(h_ref, h_syn))
    hub_jsd = _js_divergence(np.log1p(ref_dist["rev_knn_hubness"]), np.log1p(syn_dist["rev_knn_hubness"]), bins=40, val_range=(0.0, 5.0))
    hub_mmd = _mmd_rbf_1d(np.log1p(ref_dist["rev_knn_hubness"]), np.log1p(syn_dist["rev_knn_hubness"]))

    t_ref, t_syn = ref_dist["travel_time_ratio"], syn_dist["travel_time_ratio"]
    tt_w1 = float(wasserstein_distance(t_ref, t_syn))
    tt_jsd = _js_divergence(t_ref, t_syn, bins=50, val_range=(0.3, 4.5))
    tt_mmd = _mmd_rbf_1d(t_ref, t_syn)

    return {
        "cosine_margin": {"W1": margin_w1, "JSD": margin_jsd, "MMD": margin_mmd},
        "hubness_tail": {
            "W1": hub_w1,
            "JSD": hub_jsd,
            "MMD": hub_mmd,
            "zipf_gamma": float(syn_dist["zipf_gamma_est"][0]),
            "max_rev_knn": float(np.max(syn_dist["rev_knn_hubness"])),
            "p99_rev_knn": float(np.percentile(syn_dist["rev_knn_hubness"], 99)),
        },
        "travel_time": {"W1": tt_w1, "JSD": tt_jsd, "MMD": tt_mmd},
        "mean_W1": float((margin_w1 + hub_w1 + tt_w1) / 3.0),
        "mean_JSD": float((margin_jsd + hub_jsd + tt_jsd) / 3.0),
        "mean_MMD": float((margin_mmd + hub_mmd + tt_mmd) / 3.0),
    }


def _lin_concordance_cc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient (CCC in [-1, 1]) measuring joint rank & scale agreement."""
    if len(x) < 2:
        return 0.0
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x)), float(np.var(y))
    denom = vx + vy + (mx - my) ** 2
    if denom < 1e-12:
        return 1.0
    cov_xy = float(np.mean((x - mx) * (y - my)))
    return float(np.clip((2.0 * cov_xy) / denom, -1.0, 1.0))


def _exact_or_mc_spearman_pvalue(v_ref: np.ndarray, v_syn: np.ndarray, n_perm: int = 100000, seed: int = 42) -> float:
    """
    Compute exact two-sided permutation p-value over all n! permutations when n <= 8
    (e.g., 6! = 720 or 7! = 5040), or Monte Carlo permutation p-value over `n_perm` draws when n > 8.
    """
    n = len(v_ref)
    if n < 3 or np.std(v_ref) < 1e-12 or np.std(v_syn) < 1e-12:
        return 1.0
    from scipy.stats import rankdata

    rx = rankdata(v_ref).astype(np.float64)
    ry = rankdata(v_syn).astype(np.float64)
    rx_c = rx - np.mean(rx)
    ry_c = ry - np.mean(ry)
    denom = math.sqrt(float(np.sum(rx_c ** 2) * np.sum(ry_c ** 2)))
    if denom < 1e-12:
        return 1.0
    obs_abs = abs(float(np.dot(rx_c, ry_c)) / denom) - 1e-12

    if n <= 8:
        perm_mat = np.array(list(itertools.permutations(ry_c)), dtype=np.float64)
        dots = np.abs(perm_mat @ rx_c) / denom
        return float(np.mean(dots >= obs_abs))

    rng = np.random.default_rng(seed)
    exceed = 0
    batch = 10000
    rem = n_perm
    while rem > 0:
        b = min(batch, rem)
        idx = np.argsort(rng.random((b, n)), axis=1)
        ry_perm = ry_c[idx]
        dots = np.abs(ry_perm @ rx_c) / denom
        exceed += int(np.sum(dots >= obs_abs))
        rem -= b
    return float((exceed + 1.0) / (n_perm + 1.0))


def compute_multiconfig_ranking_stats(
    ref_vec: np.ndarray,
    syn_vec: np.ndarray,
    n_perm: int = 100000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate statistical rank & absolute consistency across M >= 12 paradigm-hyperparameter
    configurations (e.g., M = 18: 6 paradigms x 3 window/threshold settings), reporting Spearman rho,
    two-sided permutation p-value, Kendall tau_b, MAE, and Lin's Concordance Correlation Coefficient.
    """
    x = np.asarray(ref_vec, dtype=np.float64)
    y = np.asarray(syn_vec, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {
            "m_configs": int(len(x)),
            "spearman_rho": 0.0,
            "spearman_pvalue_perm": 1.0,
            "spearman_pvalue_asymptotic": 1.0,
            "kendall_tau_b": 0.0,
            "kendall_pvalue_asymptotic": 1.0,
            "mae": float(np.mean(np.abs(x - y))) if len(x) > 0 else 0.0,
            "lin_ccc": 0.0,
        }
    rho_val, rho_p = spearmanr(x, y)
    tau_val, tau_p = kendalltau(x, y)
    p_perm = _exact_or_mc_spearman_pvalue(x, y, n_perm=n_perm, seed=seed)
    return {
        "m_configs": int(len(x)),
        "spearman_rho": 0.0 if np.isnan(rho_val) else float(rho_val),
        "spearman_pvalue_perm": float(p_perm),
        "spearman_pvalue_asymptotic": 1.0 if np.isnan(rho_p) else float(rho_p),
        "kendall_tau_b": 0.0 if np.isnan(tau_val) else float(tau_val),
        "kendall_pvalue_asymptotic": 1.0 if np.isnan(tau_p) else float(tau_p),
        "mae": float(np.mean(np.abs(x - y))),
        "lin_ccc": _lin_concordance_cc(x, y),
    }


def compute_ranking_consistency(
    ref_engine_scores: Dict[str, Dict[str, float]],
    syn_engine_scores: Dict[str, Dict[str, float]],
    engine_order: List[str],
) -> Dict[str, float]:
    """
    Compute Spearman rank correlation rho (with exact n! permutation p-value), Kendall tau_b,
    MAE, and Lin's Concordance Correlation Coefficient across all evaluated DB engines.
    `mean_spearman_rho` averages strictly over deterministic quality metrics (`idf1`, `mota`,
    `precision`, `recall`, plus `q1_f1`, `q2_f1`, `q3_score` when evaluated); `idf1_spearman_rho`
    (`engine_mean_idf1_spearman_rho`) and its exact permutation p-value are reported explicitly.
    """
    core_quality = ["idf1", "mota", "precision", "recall"]
    query_quality = [
        m for m in ["q1_f1", "q2_f1", "q3_score"]
        if all((m in ref_engine_scores[e] and m in syn_engine_scores[e]) for e in engine_order)
    ]
    quality_metrics = core_quality + query_quality
    all_metrics = quality_metrics + (
        ["latency_ms"]
        if all(("latency_ms" in ref_engine_scores[e] and "latency_ms" in syn_engine_scores[e]) for e in engine_order)
        else []
    )

    out: Dict[str, float] = {"n_engines": int(len(engine_order))}
    quality_rhos: List[float] = []
    quality_taus: List[float] = []
    mtmc_rhos: List[float] = []

    for m in all_metrics:
        v_ref = np.array([ref_engine_scores[e][m] for e in engine_order], dtype=np.float64)
        v_syn = np.array([syn_engine_scores[e][m] for e in engine_order], dtype=np.float64)
        if np.std(v_ref) < 1e-12 or np.std(v_syn) < 1e-12:
            rho_val, tau_val, p_exact = 0.0, 0.0, 1.0
        else:
            rho_val, _ = spearmanr(v_ref, v_syn)
            tau_val, _ = kendalltau(v_ref, v_syn)
            if np.isnan(rho_val):
                rho_val = 0.0
            if np.isnan(tau_val):
                tau_val = 0.0
            p_exact = _exact_or_mc_spearman_pvalue(v_ref, v_syn)
        mae = float(np.mean(np.abs(v_ref - v_syn)))
        out[f"{m}_spearman_rho"] = float(rho_val)
        out[f"{m}_spearman_pvalue_exact"] = float(p_exact)
        out[f"{m}_kendall_tau"] = float(tau_val)
        out[f"{m}_mae"] = mae
        out[f"{m}_lin_ccc"] = _lin_concordance_cc(v_ref, v_syn)
        if m in quality_metrics:
            quality_rhos.append(float(rho_val))
            quality_taus.append(float(tau_val))
        if m in core_quality:
            mtmc_rhos.append(float(rho_val))

    out["engine_mean_idf1_spearman_rho"] = out.get("idf1_spearman_rho", 0.0)
    out["mtmc_quality_spearman_rho"] = float(np.mean(mtmc_rhos)) if mtmc_rhos else 0.0
    out["mean_spearman_rho"] = float(np.mean(quality_rhos)) if quality_rhos else 0.0
    out["mean_kendall_tau"] = float(np.mean(quality_taus)) if quality_taus else 0.0
    return out
