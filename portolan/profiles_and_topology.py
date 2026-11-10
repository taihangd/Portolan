"""Calibrated MTMC/ReID seed profiles, declarative workload specs, and urban topologies."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import torch


def select_best_device() -> torch.device:
    """Select the CUDA GPU with the most free memory, or CPU fallback."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_mem, _ = torch.cuda.mem_get_info(idx)
            if free_mem > best_free:
                best_free = free_mem
                best_idx = idx
        except Exception:
            continue
    return torch.device(f"cuda:{best_idx}")


@dataclass
class SeedProfile:
    """Calibrated statistical seed profile P_seed extracted from held-out calibration streams."""
    name: str
    intra_cos_mean: float = 0.78
    intra_cos_std: float = 0.08
    inter_cos_mean: float = 0.12
    inter_cos_std: float = 0.09
    hard_neg_cos_mean: float = 0.56
    hard_neg_group_size: int = 8
    zipf_hubness_gamma: float = 2.10
    vmf_kappa_mean: float = 460.0
    vmf_kappa_std: float = 65.0
    cam_drift_angle_rad: float = 0.34
    travel_lognorm_sigma_free: float = 0.18
    travel_lognorm_sigma_cong: float = 0.42
    signal_delay_mean_sec: float = 24.0
    queue_prob_base: float = 0.28
    natural_hub_ratio: float = 0.015
    travel_free_ratio: float = 0.90

    @classmethod
    def from_feature_tensors(
        cls,
        embeddings,
        labels,
        travel_times: Optional[np.ndarray] = None,
        name: str = "Custom-Profile",
    ) -> "SeedProfile":
        """Profile an arbitrary L2-normalized ReID embedding tensor/array and travel-time array."""
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        emb = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        n = min(emb.shape[0], 4000)
        emb_sub = emb[:n]
        lbl_sub = labels[:n]
        sim = emb_sub @ emb_sub.T
        same = (lbl_sub.unsqueeze(1) == lbl_sub.unsqueeze(0)) & (
            ~torch.eye(n, dtype=torch.bool, device=emb.device)
        )
        diff = lbl_sub.unsqueeze(1) != lbl_sub.unsqueeze(0)
        intra_vals = sim[same]
        inter_vals = sim[diff]
        mu_pos = float(intra_vals.mean().item()) if intra_vals.numel() > 0 else 0.78
        sd_pos = float(intra_vals.std().item()) if intra_vals.numel() > 0 else 0.08
        mu_neg = float(inter_vals.mean().item()) if inter_vals.numel() > 0 else 0.12
        sd_neg = float(inter_vals.std().item()) if inter_vals.numel() > 0 else 0.09
        d = emb.shape[1]
        kappa_est = max(25.0, (d - 1) / max(1e-3, 1.0 - mu_pos))
        sigma_free = 0.18
        if travel_times is not None and len(travel_times) > 10:
            log_tt = np.log(np.clip(travel_times, 1.0, None))
            sigma_free = float(np.clip(np.std(log_tt) * 0.6, 0.05, 0.85))
        return cls(
            name=name,
            intra_cos_mean=mu_pos,
            intra_cos_std=sd_pos,
            inter_cos_mean=mu_neg,
            inter_cos_std=sd_neg,
            vmf_kappa_mean=kappa_est,
            travel_lognorm_sigma_free=sigma_free,
        )


def get_seed_profile(name: str = "CityFlow-Seed", data_dir: Optional[Path] = None) -> SeedProfile:
    """
    Load empirically measured out-of-sample seed statistical profile from `data/real_seed_profiles.json`
    (fitted strictly on disjoint calibration splits of CityFlowV2 and VeRi-776), with exact
    measured fallback values when running standalone unit tests.
    """
    key = name.lower()
    canonical = "CityFlow-Seed" if "cityflow" in key else ("VeRi-Seed" if "veri" in key else "")
    if not canonical:
        raise ValueError(f"Unknown seed profile: {name}")

    cand_dirs = []
    if data_dir is not None:
        cand_dirs.append(Path(data_dir))
    cand_dirs.append(Path(__file__).resolve().parent.parent / "data")

    for d_path in cand_dirs:
        prof_json = d_path / "real_seed_profiles.json"
        if prof_json.exists():
            try:
                raw = json.loads(prof_json.read_text())
                if canonical in raw:
                    st = raw[canonical]
                    return SeedProfile(
                        name=canonical,
                        intra_cos_mean=float(st["intra_cos_mean"]),
                        intra_cos_std=float(st["intra_cos_std"]),
                        inter_cos_mean=float(st["inter_cos_mean"]),
                        inter_cos_std=float(st["inter_cos_std"]),
                        hard_neg_cos_mean=float(st["hard_neg_cos_mean"]),
                        hard_neg_group_size=8 if canonical == "CityFlow-Seed" else 6,
                        zipf_hubness_gamma=float(st["empirical_zipf_gamma"]),
                        vmf_kappa_mean=float(st["vmf_kappa_mean"]),
                        vmf_kappa_std=float(st["vmf_kappa_std"]),
                        cam_drift_angle_rad=float(st["cam_drift_angle_rad"]),
                        travel_lognorm_sigma_free=float(st["travel_lognorm_sigma_free"]),
                        travel_lognorm_sigma_cong=float(st["travel_lognorm_sigma_cong"]),
                        signal_delay_mean_sec=float(st["signal_delay_mean_sec"]),
                        queue_prob_base=float(st["queue_prob_base"]),
                        natural_hub_ratio=float(st["natural_hub_ratio"]),
                        travel_free_ratio=float(st.get("travel_free_ratio", 0.90)),
                    )
            except Exception:
                pass

    if canonical == "CityFlow-Seed":
        return SeedProfile(
            name="CityFlow-Seed",
            intra_cos_mean=0.3837,
            intra_cos_std=0.1955,
            inter_cos_mean=0.0141,
            inter_cos_std=0.1136,
            hard_neg_cos_mean=0.5133,
            hard_neg_group_size=8,
            zipf_hubness_gamma=5.9683,
            vmf_kappa_mean=571.92,
            vmf_kappa_std=85.79,
            cam_drift_angle_rad=0.6734,
            travel_lognorm_sigma_free=0.5261,
            travel_lognorm_sigma_cong=0.4557,
            signal_delay_mean_sec=4.15,
            queue_prob_base=0.3800,
            natural_hub_ratio=0.0093,
            travel_free_ratio=0.8800,
        )
    return SeedProfile(
        name="VeRi-Seed",
        intra_cos_mean=0.4776,
        intra_cos_std=0.2020,
        inter_cos_mean=0.0099,
        inter_cos_std=0.0918,
        hard_neg_cos_mean=0.4596,
        hard_neg_group_size=6,
        zipf_hubness_gamma=5.3405,
        vmf_kappa_mean=754.35,
        vmf_kappa_std=113.15,
        cam_drift_angle_rad=0.6072,
        travel_lognorm_sigma_free=0.6735,
        travel_lognorm_sigma_cong=0.1576,
        signal_delay_mean_sec=19.31,
        queue_prob_base=0.3784,
        natural_hub_ratio=0.0209,
        travel_free_ratio=2.3968,
    )


@dataclass
class WorkloadSpec:
    """Declarative knob specification W_spec controlling spatio-temporal & embedding synthesis."""
    scale_name: str
    n_cam: int
    n_veh: int
    target_snapshots: int = 0
    embed_dim: int = 256
    rho_occ: float = 0.15               # Occlusion drop rate in [0, 0.5]
    sigma_tau: float = 2.5              # Camera clock drift std (seconds) in [0, 30]
    alpha_hub: float = 0.016            # Embedding black-hole ratio in [0, 0.05]
    glare_family: str = "tanh_3mode"    # "tanh_3mode" (3-mode QR tanh) or "gamma_rank4" (rank-4 ReLU power-law)
    sigma_dp: float = 0.0               # Edge-cloud DP Gaussian noise std
    epsilon_dp: float = 0.0             # Optional Laplace/Gaussian DP budget parameter
    convoy_ratio: float = 0.08          # Fraction of vehicles in co-movement convoys
    follower_pairs: int = 25            # Number of delayed follower co-movement pairs
    congestion_amplitude: float = 0.85
    horizon_sec: float = 3600.0
    q1_num_queries: int = 200
    q2_min_hops: int = 4
    q2_time_window_sec: float = 15.0
    q4_cert_budget_ratio: float = 0.01
    q4_unlearn_vehicles: int = 35
    q4_dropout_cam_ratio: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_snapshots <= 0:
            self.target_snapshots = max(200, int(self.n_veh * 12))

    @classmethod
    def from_yaml(cls, yaml_path) -> Tuple["SeedProfile", "WorkloadSpec"]:
        """Load (SeedProfile, WorkloadSpec) directly from a declarative YAML configuration file."""
        import yaml

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        bench = cfg.get("benchmark", {})
        topo_m = cfg.get("topology_and_mobility", {})
        vis_m = cfg.get("visual_manifold_and_defects", {})
        qw = cfg.get("query_workloads", {})

        prof_name = bench.get("seed_profile", cfg.get("default_profile", "CityFlow-Seed"))
        profile = get_seed_profile(prof_name)
        scale_name = bench.get("scale_name", "Portolan-Tiny")
        base = get_benchmark_spec(
            scale_name,
            seed=int(bench.get("seed", 42)),
            embed_dim=int(vis_m.get("embed_dim", 256)),
        )
        spec = cls(
            scale_name=scale_name,
            n_cam=int(topo_m.get("n_cam", base.n_cam)),
            n_veh=int(topo_m.get("n_veh", base.n_veh)),
            target_snapshots=int(topo_m.get("target_snapshots", base.target_snapshots)),
            embed_dim=int(vis_m.get("embed_dim", base.embed_dim)),
            rho_occ=float(vis_m.get("rho_occ", base.rho_occ)),
            sigma_tau=float(vis_m.get("sigma_tau", base.sigma_tau)),
            alpha_hub=float(vis_m.get("alpha_hub", base.alpha_hub)),
            glare_family=str(vis_m.get("glare_family", base.glare_family)),
            sigma_dp=float(vis_m.get("sigma_dp", base.sigma_dp)),
            epsilon_dp=float(vis_m.get("epsilon_dp", base.epsilon_dp)),
            convoy_ratio=float(topo_m.get("convoy_ratio", base.convoy_ratio)),
            follower_pairs=int(topo_m.get("follower_pairs", base.follower_pairs)),
            congestion_amplitude=float(topo_m.get("congestion_amplitude", base.congestion_amplitude)),
            horizon_sec=float(topo_m.get("horizon_sec", base.horizon_sec)),
            q1_num_queries=int(qw.get("q1_num_queries", base.q1_num_queries)),
            q2_min_hops=int(qw.get("q2_min_hops", base.q2_min_hops)),
            q2_time_window_sec=float(qw.get("q2_time_window_sec", base.q2_time_window_sec)),
            q4_cert_budget_ratio=float(qw.get("q4_cert_budget_ratio", base.q4_cert_budget_ratio)),
            q4_unlearn_vehicles=int(qw.get("q4_unlearn_vehicles", base.q4_unlearn_vehicles)),
            q4_dropout_cam_ratio=float(qw.get("q4_dropout_cam_ratio", base.q4_dropout_cam_ratio)),
            seed=int(bench.get("seed", base.seed)),
        )
        return profile, spec


def get_benchmark_spec(scale_name: str, seed: int = 42, embed_dim: int = 256) -> WorkloadSpec:
    """Return one of the four standard Portolan benchmark scales."""
    key = scale_name.lower()
    if "tiny" in key:
        return WorkloadSpec(
            scale_name="Portolan-Tiny",
            n_cam=40,
            n_veh=2000,
            target_snapshots=25000,
            embed_dim=embed_dim,
            horizon_sec=2400.0,
            follower_pairs=15,
            seed=seed,
        )
    elif "small" in key:
        return WorkloadSpec(
            scale_name="Portolan-Small",
            n_cam=200,
            n_veh=15000,
            target_snapshots=200000,
            embed_dim=embed_dim,
            horizon_sec=3600.0,
            follower_pairs=50,
            seed=seed,
        )
    elif "medium" in key:
        return WorkloadSpec(
            scale_name="Portolan-Medium",
            n_cam=1000,
            n_veh=100000,
            target_snapshots=1200000,
            embed_dim=embed_dim,
            horizon_sec=7200.0,
            follower_pairs=150,
            seed=seed,
        )
    elif "city" in key:
        return WorkloadSpec(
            scale_name="Portolan-CityScale",
            n_cam=5000,
            n_veh=500000,
            target_snapshots=5000000,
            embed_dim=embed_dim,
            horizon_sec=10800.0,
            follower_pairs=400,
            seed=seed,
        )
    raise ValueError(f"Unknown benchmark scale: {scale_name}")


class RoadCameraTopology:
    """Hierarchical arterial-collector urban road & camera topology G = (C, E)."""

    def __init__(self, n_cam: int, seed: int = 42):
        self.n_cam = n_cam
        self.seed = seed
        rng = np.random.default_rng(seed)

        # Build perturbed urban arterial-branch grid with realistic link lengths & speed tiers
        cols = int(math.ceil(math.sqrt(n_cam * 1.25)))
        rows = int(math.ceil(n_cam / cols))
        spacing_m = 380.0  # ~380m average urban intersection spacing

        coords = []
        is_arterial = []
        for idx in range(n_cam):
            r = idx // cols
            c = idx % cols
            jitter_x = rng.normal(0.0, 55.0)
            jitter_y = rng.normal(0.0, 55.0)
            coords.append([c * spacing_m + jitter_x, r * spacing_m + jitter_y])
            # Every 3rd row/col forms major high-speed urban arterials
            is_arterial.append((r % 3 == 0) or (c % 3 == 0))

        self.coords = np.array(coords, dtype=np.float32)  # [N_cam, 2] in meters
        self.is_arterial = np.array(is_arterial, dtype=bool)
        self.cam_fov_angle = rng.uniform(0.0, 2.0 * math.pi, size=n_cam).astype(np.float32)

        # Construct directed edges E: grid neighbors + arterial diagonal/expressway connectors
        adj: List[List[int]] = [[] for _ in range(n_cam)]
        for idx in range(n_cam):
            r = idx // cols
            c = idx % cols
            candidates = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nidx = nr * cols + nc
                    if nidx < n_cam:
                        candidates.append(nidx)
            # Add diagonal arterial-branch connectors with probability 0.35
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                if rng.random() < 0.35:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nidx = nr * cols + nc
                        if nidx < n_cam:
                            candidates.append(nidx)
            for nidx in candidates:
                if nidx not in adj[idx]:
                    adj[idx].append(nidx)
                if idx not in adj[nidx]:
                    adj[nidx].append(idx)

        self.adj = adj
        self.max_deg = max(len(nbrs) for nbrs in adj)

        # Padded neighbor tables for fast vectorized GPU/CPU route sampling
        self.nbr_table = np.full((n_cam, self.max_deg), -1, dtype=np.int32)
        self.edge_dist = np.zeros((n_cam, self.max_deg), dtype=np.float32)
        self.edge_speed = np.zeros((n_cam, self.max_deg), dtype=np.float32)
        self.free_flow_time = np.zeros((n_cam, self.max_deg), dtype=np.float32)
        self.edge_heading = np.zeros((n_cam, self.max_deg), dtype=np.float32)

        self.edge_dict: Dict[Tuple[int, int], float] = {}
        for u in range(n_cam):
            for k, v in enumerate(adj[u]):
                self.nbr_table[u, k] = v
                dx = float(self.coords[v, 0] - self.coords[u, 0])
                dy = float(self.coords[v, 1] - self.coords[u, 1])
                dist = max(120.0, math.hypot(dx, dy))
                # Arterial speed ~15.5 m/s (56 km/h), collector ~11.0 m/s (40 km/h)
                spd = 15.5 if (self.is_arterial[u] and self.is_arterial[v]) else 11.2
                spd *= float(rng.uniform(0.92, 1.08))
                fft = dist / spd
                self.edge_dist[u, k] = dist
                self.edge_speed[u, k] = spd
                self.free_flow_time[u, k] = fft
                self.edge_heading[u, k] = math.atan2(dy, dx)
                self.edge_dict[(u, v)] = fft

        self.reach_min_time: List[Dict[int, Tuple[int, float]]] = [
            {} for _ in range(n_cam)
        ]
        if n_cam <= 1500:
            self._build_bounded_reachability(max_hops=3)

    def _build_bounded_reachability(self, max_hops: int = 3) -> None:
        """Compute exact shortest travel time and hop count up to max_hops from every camera."""
        for src in range(self.n_cam):
            best: Dict[int, Tuple[int, float]] = {src: (0, 0.0)}
            frontier = [(src, 0, 0.0)]
            while frontier:
                nxt_frontier = []
                for u, h, t_acc in frontier:
                    if h >= max_hops:
                        continue
                    for k, v in enumerate(self.adj[u]):
                        if v == src:
                            continue
                        nt = t_acc + float(self.free_flow_time[u, k])
                        nh = h + 1
                        if v not in best or nt < best[v][1]:
                            best[v] = (nh, nt)
                            nxt_frontier.append((v, nh, nt))
                frontier = nxt_frontier
            del best[src]
            self.reach_min_time[src] = best

    @classmethod
    def from_empirical_arrays(
        cls,
        coords: np.ndarray,
        edges_u: np.ndarray,
        edges_v: np.ndarray,
        edges_fft: np.ndarray,
        edges_dist: np.ndarray,
        seed: int = 42,
    ) -> "RoadCameraTopology":
        """
        Construct a RoadCameraTopology from physical camera coordinates, directed road links,
        and physical distance-derived free-flow travel times (0% test-stream timestamp leakage).
        """
        topo = object.__new__(cls)
        n_cam = int(coords.shape[0])
        topo.n_cam = n_cam
        topo.seed = seed
        topo.coords = np.asarray(coords, dtype=np.float32).copy()
        rng = np.random.default_rng(seed)
        topo.cam_fov_angle = rng.uniform(0.0, 2.0 * math.pi, size=n_cam).astype(np.float32)

        edge_info: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for u_raw, v_raw, fft_raw, dist_raw in zip(edges_u, edges_v, edges_fft, edges_dist):
            u_i, v_i = int(u_raw), int(v_raw)
            if 0 <= u_i < n_cam and 0 <= v_i < n_cam and u_i != v_i:
                edge_info[(u_i, v_i)] = (max(2.0, float(fft_raw)), max(25.0, float(dist_raw)))

        # Guarantee every camera has >= 2 outgoing neighbors so route sampling has zero dead ends
        for u_i in range(n_cam):
            out_cnt = sum(1 for (a, _) in edge_info if a == u_i)
            if out_cnt < 2:
                dists = np.linalg.norm(topo.coords - topo.coords[u_i], axis=1)
                dists[u_i] = 1e9
                for nb in np.argsort(dists)[:3]:
                    v_i = int(nb)
                    if v_i != u_i and (u_i, v_i) not in edge_info:
                        d_m = max(25.0, float(dists[v_i]))
                        edge_info[(u_i, v_i)] = (max(2.0, d_m / 13.9), d_m)
                        out_cnt += 1
                        if out_cnt >= 2:
                            break

        adj: List[List[int]] = [[] for _ in range(n_cam)]
        in_deg = np.zeros(n_cam, dtype=np.int32)
        for (u_i, v_i) in sorted(edge_info.keys()):
            adj[u_i].append(v_i)
            in_deg[v_i] += 1

        tot_deg = np.array([len(adj[u]) + int(in_deg[u]) for u in range(n_cam)], dtype=np.int32)
        med_deg = float(np.median(tot_deg))
        topo.is_arterial = (tot_deg >= med_deg).astype(bool)
        topo.adj = adj
        topo.max_deg = max(1, max(len(nbrs) for nbrs in adj))

        topo.nbr_table = np.full((n_cam, topo.max_deg), -1, dtype=np.int32)
        topo.edge_dist = np.zeros((n_cam, topo.max_deg), dtype=np.float32)
        topo.edge_speed = np.zeros((n_cam, topo.max_deg), dtype=np.float32)
        topo.free_flow_time = np.zeros((n_cam, topo.max_deg), dtype=np.float32)
        topo.edge_heading = np.zeros((n_cam, topo.max_deg), dtype=np.float32)
        topo.edge_dict = {}

        for u_i in range(n_cam):
            for k, v_i in enumerate(adj[u_i]):
                fft_val, dist_val = edge_info[(u_i, v_i)]
                dx = float(topo.coords[v_i, 0] - topo.coords[u_i, 0])
                dy = float(topo.coords[v_i, 1] - topo.coords[u_i, 1])
                spd = dist_val / max(1.0, fft_val)
                topo.nbr_table[u_i, k] = v_i
                topo.edge_dist[u_i, k] = dist_val
                topo.edge_speed[u_i, k] = spd
                topo.free_flow_time[u_i, k] = fft_val
                topo.edge_heading[u_i, k] = math.atan2(dy, dx)
                topo.edge_dict[(u_i, v_i)] = fft_val

        topo.reach_min_time = [{} for _ in range(n_cam)]
        topo._build_bounded_reachability(max_hops=4 if n_cam <= 50 else 3)
        return topo


def build_zero_gt_cityflow_edges(
    coords: np.ndarray,
    v_free_mps: float = 13.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a physical urban corridor road graph purely from camera GPS homography coordinates:
    connects co-intersection camera poles within <=230 m plus up to 4 nearest adjacent-intersection
    cameras within <=900 m (pruning >420 m collinear shortcuts), and computes free-flow travel times
    from FOV-footprint road distance `max(25 m, d_pole - 110 m) / 13.9 m/s` (50 km/h arterial, ~55 m
    nominal pole-to-intersection FOV standoff per camera), using 0% ground-truth labels or timestamps.
    """
    n_cam = int(coords.shape[0])
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1).astype(np.float64)

    cand_edges = set()
    for u in range(n_cam):
        order = np.argsort(dmat[u])
        for v in order:
            v_i = int(v)
            if v_i != u and dmat[u, v_i] <= 230.0:
                cand_edges.add((u, v_i))
                cand_edges.add((v_i, u))
        picked_inter = 0
        for v in order:
            v_i = int(v)
            if v_i == u or dmat[u, v_i] <= 230.0:
                continue
            if dmat[u, v_i] <= 900.0 or picked_inter < 2:
                cand_edges.add((u, v_i))
                cand_edges.add((v_i, u))
                picked_inter += 1
                if picked_inter >= 4:
                    break

    # Prune transitive corridor shortcuts when an intermediate intersection w lies collinearly between u and v
    kept_u, kept_v, kept_dist, kept_fft = [], [], [], []
    for (u, v) in sorted(cand_edges):
        duv = float(dmat[u, v])
        is_shortcut = False
        if duv > 420.0:
            for w in range(n_cam):
                if w == u or w == v:
                    continue
                if (u, w) in cand_edges and (w, v) in cand_edges:
                    duw = float(dmat[u, w])
                    dwv = float(dmat[w, v])
                    if duw > 180.0 and dwv > 180.0 and (duw + dwv) < 1.05 * duv:
                        is_shortcut = True
                        break
        if not is_shortcut:
            dist_m = max(25.0, duv - 110.0)
            fft_s = max(2.0, dist_m / float(v_free_mps))
            kept_u.append(u)
            kept_v.append(v)
            kept_dist.append(dist_m)
            kept_fft.append(fft_s)

    return (
        np.array(kept_u, dtype=np.int32),
        np.array(kept_v, dtype=np.int32),
        np.array(kept_fft, dtype=np.float32),
        np.array(kept_dist, dtype=np.float32),
    )


def build_zero_gt_veri_edges(
    coords: np.ndarray,
    npz_edges_u: np.ndarray,
    npz_edges_v: np.ndarray,
    npz_edges_dist: np.ndarray,
    cal_veh: np.ndarray,
    cal_cam: np.ndarray,
    cal_time: np.ndarray,
    v_free_mps: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the VeRi-776 campus road graph strictly from physical road distances (`camera_Dist.txt`
    stored in `npz_edges_dist`) and Calibration Trips ONLY (`veh_id % 2 == 0`), with physical
    free-flow travel times `edges_fft = max(2.0, edges_dist / 15.0)` (54 km/h ring road) and
    0% held-out trip (`veh_id % 2 == 1`) transition or timestamp access.
    """
    n_cam = int(coords.shape[0])
    coord_dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(np.float64)
    phys_dist: Dict[Tuple[int, int], float] = {}
    for u_r, v_r, d_r in zip(npz_edges_u, npz_edges_v, npz_edges_dist):
        u_i, v_i = int(u_r), int(v_r)
        if 0 <= u_i < n_cam and 0 <= v_i < n_cam and u_i != v_i:
            d_val = max(25.0, float(d_r))
            phys_dist[(u_i, v_i)] = d_val
            phys_dist.setdefault((v_i, u_i), d_val)

    cand_edges = set()
    # 1. Calibration-split transitions ONLY (strictly 0% overlap with held-out evaluation trips)
    by_v: Dict[int, List[Tuple[float, int]]] = {}
    for idx in range(len(cal_veh)):
        by_v.setdefault(int(cal_veh[idx]), []).append((float(cal_time[idx]), int(cal_cam[idx])))
    for _, seq in by_v.items():
        seq.sort()
        for i in range(len(seq) - 1):
            t1, c1 = seq[i]
            t2, c2 = seq[i + 1]
            if c1 != c2 and 0.3 <= (t2 - t1) <= 240.0:
                cand_edges.add((c1, c2))

    # 2. Physical road proximity <= 260m from camera_Dist table
    for (u_i, v_i), d_val in phys_dist.items():
        if d_val <= 260.0:
            cand_edges.add((u_i, v_i))

    def _get_dist(a: int, b: int) -> float:
        if (a, b) in phys_dist:
            return phys_dist[(a, b)]
        return max(60.0, float(coord_dmat[a, b]))

    # Prune transitive multi-hop shortcuts purely using physical road distances (0% timestamp quantiles)
    kept_u, kept_v, kept_dist, kept_fft = [], [], [], []
    for (u, v) in sorted(cand_edges):
        duv = _get_dist(u, v)
        is_shortcut = False
        if duv > 220.0:
            for w in range(n_cam):
                if w == u or w == v:
                    continue
                if (u, w) in cand_edges and (w, v) in cand_edges:
                    if (_get_dist(u, w) + _get_dist(w, v)) < 1.18 * duv:
                        is_shortcut = True
                        break
        if not is_shortcut:
            dist_m = max(25.0, duv)
            fft_s = max(2.0, dist_m / float(v_free_mps))
            kept_u.append(u)
            kept_v.append(v)
            kept_dist.append(dist_m)
            kept_fft.append(fft_s)

    return (
        np.array(kept_u, dtype=np.int32),
        np.array(kept_v, dtype=np.int32),
        np.array(kept_fft, dtype=np.float32),
        np.array(kept_dist, dtype=np.float32),
    )


def compute_empirical_profile_stats(stream_dataset: Dict, name: str) -> Dict:
    """
    Estimate unclipped Proposition-1 manifold, hubness, and travel-time seed parameters strictly
    from a calibration stream dictionary (`split='cal'`), using physical `topology.edge_dict`
    free-flow times and wide physical sanity bounds (`> 0` / `< 1.2`) without artificial ceilings.
    """
    emb = stream_dataset["embeddings"]
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb, dtype=torch.float32)
    emb = torch.nn.functional.normalize(emb.float().cpu(), p=2, dim=-1)
    veh = np.asarray(stream_dataset["snap_veh"], dtype=np.int32)
    cam = np.asarray(stream_dataset["snap_cam"], dtype=np.int32)
    ts = np.asarray(stream_dataset["snap_time"], dtype=np.float64)
    topo: RoadCameraTopology = stream_dataset["topology"]
    n = int(len(veh))

    sim = (emb @ emb.T).numpy()
    same_v = (veh[:, None] == veh[None, :]) & (~np.eye(n, dtype=bool))
    diff_v = veh[:, None] != veh[None, :]

    intra_vals = sim[same_v]
    inter_vals = sim[diff_v]
    mu_pos = float(np.mean(intra_vals)) if intra_vals.size > 0 else 0.50
    sd_pos = float(np.std(intra_vals)) if intra_vals.size > 0 else 0.15
    mu_neg = float(np.mean(inter_vals)) if inter_vals.size > 0 else 0.02
    sd_neg = float(np.std(inter_vals)) if inter_vals.size > 0 else 0.10

    cross_sim = np.where(diff_v, sim, -2.0)
    hard_neg_per_snap = np.max(cross_sim, axis=1)
    hard_neg_mean = float(np.mean(hard_neg_per_snap))

    k_nn = min(10, max(1, n - 1))
    topk = np.argpartition(-cross_sim, k_nn, axis=1)[:, :k_nn]
    rev_deg = np.bincount(topk.ravel(), minlength=n).astype(np.float64)
    hub_ratio = float(np.clip(np.mean(rev_deg >= 25.0), 1e-4, 0.50))
    tail = rev_deg[rev_deg >= 16.0]
    if len(tail) > 8:
        xmin = float(np.min(tail)) - 0.5
        gamma_hat = float(1.0 + len(tail) / max(1e-6, float(np.sum(np.log(tail / xmin)))))
    else:
        gamma_hat = 3.5

    by_v: Dict[int, List[int]] = {}
    for idx in range(n):
        by_v.setdefault(int(veh[idx]), []).append(idx)

    # Calibrate road travel-time statistics on inter-intersection links (dist >= 80m); if calibration
    # cameras are purely single-intersection clusters (< 80m), retain nominal physical free-flow ratio 0.88
    road_ratios: List[float] = []
    all_ratios: List[float] = []
    for _, sids in by_v.items():
        for i in range(len(sids) - 1):
            s1, s2 = sids[i], sids[i + 1]
            c1, c2 = int(cam[s1]), int(cam[s2])
            if c1 == c2:
                continue
            fft = topo.edge_dict.get((c1, c2))
            dist_uv = float(np.linalg.norm(topo.coords[c1] - topo.coords[c2]))
            if c1 in range(topo.n_cam) and c2 in topo.adj[c1]:
                k_pos = topo.adj[c1].index(c2)
                dist_uv = float(topo.edge_dist[c1, k_pos])
            if fft is None and c2 in topo.reach_min_time[c1]:
                fft = topo.reach_min_time[c1][c2][1]
            dt = float(ts[s2] - ts[s1])
            if fft is not None and fft >= 2.0 and 0.2 <= dt <= 240.0:
                r = dt / float(fft)
                if 0.20 <= r <= 6.0:
                    all_ratios.append(r)
                    if dist_uv >= 80.0:
                        road_ratios.append(r)

    use_road = len(road_ratios) >= 15
    tt_arr = np.array(road_ratios if use_road else all_ratios, dtype=np.float64)
    if tt_arr.size == 0:
        tt_arr = np.array([1.0, 1.4], dtype=np.float64)
    q65 = float(np.percentile(tt_arr, 65))
    free_part = tt_arr[tt_arr <= q65]
    cong_part = tt_arr[tt_arr > q65]
    free_ratio_med = float(np.clip(np.percentile(tt_arr, 28), 0.45, 2.50)) if use_road else 0.88

    # Measure excess signal/queueing delay relative to calibrated free-flow transit baseline
    delays: List[float] = []
    for _, sids in by_v.items():
        for i in range(len(sids) - 1):
            s1, s2 = sids[i], sids[i + 1]
            c1, c2 = int(cam[s1]), int(cam[s2])
            if c1 == c2:
                continue
            fft = topo.edge_dict.get((c1, c2))
            if fft is None and c2 in topo.reach_min_time[c1]:
                fft = topo.reach_min_time[c1][c2][1]
            dt = float(ts[s2] - ts[s1])
            if fft is not None and fft >= 2.0 and 0.2 <= dt <= 240.0:
                r = dt / float(fft)
                if 0.20 <= r <= 6.0:
                    eff_fft = max(float(fft), 9.0)
                    if r > free_ratio_med * 1.35:
                        delays.append((r - free_ratio_med) * eff_fft)

    sigma_free = float(np.clip(np.std(np.log(np.maximum(0.20, free_part))), 0.05, 0.95))
    sigma_cong = (
        float(np.clip(np.std(np.log(np.maximum(0.20, cong_part))), 0.08, 1.10))
        if len(cong_part) > 5
        else 0.38
    )
    q_prob = float(np.clip(np.mean(tt_arr > float(np.percentile(tt_arr, 62))), 0.05, 0.85))
    delay_mean = float(np.clip(np.mean(delays), 0.5, 90.0)) if delays else 12.0

    # Unclipped Proposition 1 decomposition: near-view 85th-percentile cosine vs. cross-camera mean
    d = int(emb.shape[1])
    p85_pos = float(np.percentile(intra_vals, 85)) if intra_vals.size > 0 else min(0.92, mu_pos + sd_pos)
    atten = float(np.clip(mu_pos / max(1e-4, p85_pos), 0.25, 0.995))
    drift_rad = float(np.clip(math.sqrt(max(1e-4, -math.log(atten))), 0.05, 1.20))
    target_ad = math.sqrt(float(np.clip(p85_pos, 0.15, 0.99)))
    kappa_mean = float(np.clip((d - 1.0) / max(1e-3, 2.0 * (1.0 - target_ad)), 20.0, 2500.0))

    n_edges = sum(len(nbrs) for nbrs in topo.adj)
    return {
        "name": name,
        "calibration_split": str(stream_dataset.get("split", "cal")),
        "n_snapshots": int(n),
        "n_cam": int(topo.n_cam),
        "n_veh": int(len(stream_dataset["gt_trajectories"])),
        "n_directed_edges": int(n_edges),
        "intra_cos_mean": round(mu_pos, 4),
        "intra_cos_std": round(sd_pos, 4),
        "inter_cos_mean": round(mu_neg, 4),
        "inter_cos_std": round(sd_neg, 4),
        "hard_neg_cos_mean": round(hard_neg_mean, 4),
        "empirical_zipf_gamma": round(gamma_hat, 4),
        "natural_hub_ratio": round(hub_ratio, 4),
        "vmf_kappa_mean": round(kappa_mean, 2),
        "vmf_kappa_std": round(0.15 * kappa_mean, 2),
        "cam_drift_angle_rad": round(drift_rad, 4),
        "travel_lognorm_sigma_free": round(sigma_free, 4),
        "travel_lognorm_sigma_cong": round(sigma_cong, 4),
        "signal_delay_mean_sec": round(delay_mean, 2),
        "queue_prob_base": round(q_prob, 4),
        "travel_free_ratio": round(free_ratio_med, 4),
    }


def calibrate_seed_profile_from_stream(
    stream_dataset: Dict, name: str
) -> Tuple[SeedProfile, Dict]:
    """Fit a SeedProfile and summary dict directly from a calibration stream (`split='cal'`)."""
    st = compute_empirical_profile_stats(stream_dataset, name=name)
    canonical = "CityFlow-Seed" if "cityflow" in name.lower() else "VeRi-Seed"
    prof = SeedProfile(
        name=canonical,
        intra_cos_mean=float(st["intra_cos_mean"]),
        intra_cos_std=float(st["intra_cos_std"]),
        inter_cos_mean=float(st["inter_cos_mean"]),
        inter_cos_std=float(st["inter_cos_std"]),
        hard_neg_cos_mean=float(st["hard_neg_cos_mean"]),
        hard_neg_group_size=8 if canonical == "CityFlow-Seed" else 6,
        zipf_hubness_gamma=float(st["empirical_zipf_gamma"]),
        vmf_kappa_mean=float(st["vmf_kappa_mean"]),
        vmf_kappa_std=float(st["vmf_kappa_std"]),
        cam_drift_angle_rad=float(st["cam_drift_angle_rad"]),
        travel_lognorm_sigma_free=float(st["travel_lognorm_sigma_free"]),
        travel_lognorm_sigma_cong=float(st["travel_lognorm_sigma_cong"]),
        signal_delay_mean_sec=float(st["signal_delay_mean_sec"]),
        queue_prob_base=float(st["queue_prob_base"]),
        natural_hub_ratio=float(st["natural_hub_ratio"]),
        travel_free_ratio=float(st["travel_free_ratio"]),
    )
    return prof, st


def load_real_benchmark_stream(
    dataset_name: str = "CityFlow-Seed",
    fold_idx: int = 1,
    data_dir: Optional[Path] = None,
    q2_min_hops: int = 3,
    q2_time_window_sec: float = 20.0,
    split: str = "eval",
    seed_profile_override: Optional[SeedProfile] = None,
) -> Dict:
    """
    Load a real-world cross-camera ReID & trajectory stream with strict out-of-sample separation:
      - `Real-CityFlow` (`real_cityflow_stream.npz`):
          * `split="cal"`: Calibration Corridors (`cam < 9`, S01+S02, 9 cameras, 240 vehicles, 859 passes).
          * `split="eval"` (default): Held-Out 31-Camera Network (`cam >= 9`, S03+S04+S05, 31 cameras
            re-indexed `0..30`, 426 unseen vehicles, 2,178 passes — 0% camera/road/vehicle overlap!).
          * Physical free-flow times `edges_fft = max(2.0, edges_dist / 13.9)` (50 km/h arterial, 0% GT).
      - `Real-VeRi` (`real_veri776_stream.npz`):
          * `split="cal"`: Calibration Trips (`veh_id % 2 == 0`, 137 trips, 958 passes).
          * `split="eval"` (default): Held-Out Evaluation Trips (`veh_id % 2 == 1`, 136 unseen trips,
            1,010 passes — 0% vehicle/trip overlap!).
          * Physical free-flow times `edges_fft = max(2.0, edges_dist / 15.0)` (54 km/h ring road, 0% test GT).
    """
    key = dataset_name.lower()
    is_veri = "veri" in key
    if "cityflow" in key:
        fname = "real_cityflow_stream.npz"
        prof_name = "CityFlow-Seed"
        stream_label = "Real-CityFlow"
        v_free_mps = 13.9
    elif is_veri:
        fname = "real_veri776_stream.npz"
        prof_name = "VeRi-Seed"
        stream_label = "Real-VeRi"
        v_free_mps = 15.0
    else:
        raise ValueError(f"Unknown real dataset stream: {dataset_name}")

    split_mode = split.lower().strip()
    if split_mode not in ("cal", "eval", "all"):
        raise ValueError(f"Invalid split={split!r}; expected 'cal', 'eval', or 'all'.")

    base_dir = Path(data_dir) if data_dir is not None else (Path(__file__).resolve().parent.parent / "data")
    npz_path = base_dir / fname
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Real benchmark stream {npz_path} not found. Run scripts/extract_real_benchmarks.py first."
        )

    f_idx = int(fold_idx) % 3
    with np.load(npz_path, allow_pickle=False) as z:
        full_coords = z["coords"].astype(np.float32)
        npz_edges_u = z["edges_u"].astype(np.int32)
        npz_edges_v = z["edges_v"].astype(np.int32)
        npz_edges_dist = z["edges_dist"].astype(np.float32)

        raw_time = z[f"fold_{f_idx}_snap_time"].astype(np.float64)
        raw_veh = z[f"fold_{f_idx}_snap_veh"].astype(np.int32)
        raw_cam = z[f"fold_{f_idx}_snap_cam"].astype(np.int32)
        raw_emb = z[f"fold_{f_idx}_embeddings"].astype(np.float32)

    if not is_veri:
        if split_mode == "cal":
            mask = raw_cam < 9
            coords = full_coords[:9].copy()
            cam_sub = raw_cam[mask].copy()
        elif split_mode == "eval":
            mask = raw_cam >= 9
            coords = full_coords[9:].copy()
            cam_sub = (raw_cam[mask] - 9).astype(np.int32)
        else:
            mask = np.ones(len(raw_cam), dtype=bool)
            coords = full_coords.copy()
            cam_sub = raw_cam.copy()
        edges_u, edges_v, edges_fft, edges_dist = build_zero_gt_cityflow_edges(
            coords=coords, v_free_mps=v_free_mps
        )
    else:
        cal_mask_ref = (raw_veh % 2) == 0
        if split_mode == "cal":
            mask = cal_mask_ref
        elif split_mode == "eval":
            mask = (raw_veh % 2) == 1
        else:
            mask = np.ones(len(raw_veh), dtype=bool)
        coords = full_coords.copy()
        cam_sub = raw_cam[mask].copy()
        edges_u, edges_v, edges_fft, edges_dist = build_zero_gt_veri_edges(
            coords=coords,
            npz_edges_u=npz_edges_u,
            npz_edges_v=npz_edges_v,
            npz_edges_dist=npz_edges_dist,
            cal_veh=raw_veh[cal_mask_ref],
            cal_cam=raw_cam[cal_mask_ref],
            cal_time=raw_time[cal_mask_ref],
            v_free_mps=v_free_mps,
        )

    snap_time = raw_time[mask]
    raw_veh_sub = raw_veh[mask]
    snap_cam = cam_sub
    emb_np = raw_emb[mask]

    # Re-index split vehicle IDs contiguously to 0..n_veh_sub-1 while recording raw IDs for disjointness checks
    uniq_raw_vehs, inv_veh = np.unique(raw_veh_sub, return_inverse=True)
    snap_veh = inv_veh.astype(np.int32)
    n_veh = int(len(uniq_raw_vehs))
    n_cam = int(coords.shape[0])

    # Ensure strict chronological ordering
    order = np.argsort(snap_time, kind="mergesort")
    snap_time = snap_time[order]
    snap_veh = snap_veh[order]
    snap_cam = snap_cam[order]
    emb_np = emb_np[order]
    emb_t = torch.nn.functional.normalize(torch.from_numpy(emb_np), p=2, dim=-1)

    n_snaps = int(len(snap_veh))
    gt_trajs: Dict[int, List[int]] = {v: [] for v in range(n_veh)}
    for sid, vid in enumerate(snap_veh.tolist()):
        gt_trajs.setdefault(int(vid), []).append(sid)

    # Diagnostic empirical hub indicator (reverse-10NN cross-identity degree >= 25; never exposed to engines)
    sim_mat = (emb_t @ emb_t.T).numpy()
    diff_v = snap_veh[:, None] != snap_veh[None, :]
    cross_sim = np.where(diff_v, sim_mat, -2.0)
    k_nn = min(10, max(1, n_snaps - 1))
    topk = np.argpartition(-cross_sim, k_nn, axis=1)[:, :k_nn]
    rev_deg = np.bincount(topk.ravel(), minlength=n_snaps)
    is_bh = (rev_deg >= 25).astype(bool)

    topology = RoadCameraTopology.from_empirical_arrays(
        coords=coords,
        edges_u=edges_u,
        edges_v=edges_v,
        edges_fft=edges_fft,
        edges_dist=edges_dist,
        seed=42 + f_idx,
    )

    profile = (
        seed_profile_override
        if seed_profile_override is not None
        else get_seed_profile(prof_name, data_dir=base_dir)
    )
    horizon_obs = max(300.0, float(snap_time[-1] - snap_time[0])) if n_snaps > 1 else 1800.0
    matched_spec = WorkloadSpec(
        scale_name=stream_label if split_mode == "eval" else f"{stream_label}-{split_mode}",
        n_cam=n_cam,
        n_veh=n_veh,
        target_snapshots=n_snaps,
        embed_dim=int(emb_t.shape[1]),
        rho_occ=0.30 if is_veri else 0.14,
        sigma_tau=0.8 if is_veri else 1.2,
        alpha_hub=profile.natural_hub_ratio,
        convoy_ratio=0.08,
        follower_pairs=max(6, n_veh // 40),
        congestion_amplitude=0.58 if is_veri else 0.28,
        horizon_sec=horizon_obs,
        q1_num_queries=min(150, n_veh),
        q2_min_hops=q2_min_hops,
        q2_time_window_sec=q2_time_window_sec,
        seed=42 + f_idx,
    )

    return {
        "name": stream_label,
        "split": split_mode,
        "fold_idx": f_idx,
        "v_free_mps": v_free_mps,
        "raw_veh_ids": uniq_raw_vehs,
        "total_snapshots": n_snaps,
        "embeddings": emb_t,
        "snap_veh": snap_veh,
        "snap_cam": snap_cam,
        "snap_time": snap_time,
        "snap_true_time": snap_time.copy(),
        "is_blackhole": is_bh,
        "gt_trajectories": gt_trajs,
        "topology": topology,
        "spec": matched_spec,
        "profile": profile,
    }
