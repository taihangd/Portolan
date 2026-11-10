"""
Portolan Three-Engine Simulation-Free Cross-Camera Trajectory & Workload Synthesizer.

Engine 1: Macro-Micro Coupled Mobility Synthesizer (3D-Free Spatio-Temporal Engine)
Engine 2: Manifold-Preserving Visual Embedding & Defect Synthesizer (Latent Snapshot Engine)
Engine 3: Multi-Modal Declarative Query Workload Synthesizer (Q1, Q2, Q3, Q4 + GT)
"""

from typing import Dict, List, Optional, Tuple, Union
import copy
import gc
import math
import sys
import time
try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore[assignment]
import numpy as np
import torch


def _get_peak_rss_mb() -> float:
    """Return peak process RSS in MiB across Linux (KiB) and macOS/Darwin (bytes)."""
    if resource is None:
        return 0.0
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0

from portolan.profiles_and_topology import (
    RoadCameraTopology,
    SeedProfile,
    WorkloadSpec,
)


def sample_vmf_radial_w(
    kappa: torch.Tensor, dim: int, device: torch.device, n_reject_iters: int = 5
) -> torch.Tensor:
    """
    Exact vectorized Wood (1994) / Ulrich (1984) sampler for the radial cosine w in [-1, 1]
    of the von Mises-Fisher distribution vMF(mu, kappa) on S^{d-1}.
    """
    n = kappa.shape[0]
    m = float(dim - 1)
    # Wood's parameters: b, x0, c
    b = (-2.0 * kappa + torch.sqrt(4.0 * (kappa ** 2) + m * m)) / m
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + m * torch.log(torch.clamp(1.0 - x0 * x0, min=1e-12))

    half_m = torch.full((n,), m / 2.0, device=device, dtype=torch.float32)
    beta_dist = torch.distributions.Beta(half_m, half_m)

    w_out = torch.empty(n, device=device, dtype=torch.float32)
    accepted = torch.zeros(n, device=device, dtype=torch.bool)

    for _ in range(n_reject_iters):
        if accepted.all():
            break
        z = beta_dist.sample()
        w_prop = (1.0 - (1.0 + b) * z) / torch.clamp(1.0 - (1.0 - b) * z, min=1e-7)
        u = torch.clamp(torch.rand(n, device=device), min=1e-12)
        lhs = kappa * w_prop + m * torch.log(torch.clamp(1.0 - x0 * w_prop, min=1e-12)) - c
        acc_now = (~accepted) & (lhs >= torch.log(u))
        w_out = torch.where(acc_now, w_prop, w_out)
        accepted = accepted | acc_now

    # Fallback saddlepoint mean for the <0.01% unaccepted tail
    if not accepted.all():
        w_mean = 1.0 - m / torch.clamp(2.0 * kappa, min=m + 1.0)
        w_out = torch.where(accepted, w_out, w_mean)

    return torch.clamp(w_out, -0.9999, 0.9999)


class PortolanSynthesizer:
    """Full manifold-preserving, congestion-coupled, black-hole-aware synthesizer."""

    def __init__(
        self,
        profile: SeedProfile,
        spec: WorkloadSpec,
        topology: RoadCameraTopology,
        device: Union[torch.device, str],
        use_vmf: bool = True,
        use_blackhole: bool = True,
        use_congestion_wave: bool = True,
        use_hard_negatives: bool = True,
        glare_family: Optional[str] = None,
    ):
        self.profile = profile
        self.spec = spec
        self.topology = topology
        self.device = torch.device(device)
        self.use_vmf = use_vmf
        self.use_blackhole = use_blackhole
        self.use_congestion_wave = use_congestion_wave
        self.use_hard_negatives = use_hard_negatives
        self.glare_family = str(glare_family if glare_family is not None else getattr(spec, "glare_family", "tanh_3mode"))

    def _prepare_manifold_parameters(self, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        """Precompute global SO(d) camera viewpoint rotors, vehicle prototypes, and hub attractors."""
        d = self.spec.embed_dim
        n_cam = self.topology.n_cam
        n_veh = self.spec.n_veh

        # Orthonormal basis W0 in O(d) via QR decomposition
        q_mat, _ = torch.linalg.qr(torch.randn(d, d, device=self.device))

        # Camera-specific Givens plane rotation angles phi_c in R^{d/2} so R_c = W0 G(phi_c) W0^T in SO(d).
        # When cross-camera cosine spread exceeds single-view vMF dispersion (sigma_pos * sqrt(kappa / d) > 0.25),
        # adjacent corridor cameras exhibit bimodal front/rear viewing-aspect polarity (0% GT access).
        half_d = d // 2
        sigma_angle = self.profile.cam_drift_angle_rad
        aspect_excess = self.profile.intra_cos_std * math.sqrt(max(1.0, self.profile.vmf_kappa_mean) / 256.0) - 0.25
        if self.use_vmf and aspect_excess > 0.0:
            polarity = np.zeros(n_cam, dtype=np.float32)
            for c in range(n_cam):
                nbr_pols = [polarity[nb] for nb in self.topology.adj[c] if nb < c and polarity[nb] != 0.0]
                if nbr_pols:
                    polarity[c] = -1.0 if sum(nbr_pols) > 0 else 1.0
                else:
                    polarity[c] = 1.0 if rng.random() < 0.5 else -1.0
            cam_psi = np.where(polarity > 0, 0.0, math.pi) + rng.normal(0.0, 0.38, size=n_cam)
            asp_strength = float(np.clip(aspect_excess * 9.5, 0.0, 0.65))
            m_asp = half_d // 2
            psi_t = torch.from_numpy(cam_psi.astype(np.float32)).to(self.device).unsqueeze(1)
            freq = torch.where(torch.arange(m_asp, device=self.device) % 2 == 0, 1.0, -1.0).unsqueeze(0)
            phi_asp = (
                asp_strength * torch.cos(psi_t) * freq
                + (sigma_angle * 0.55) * torch.randn(n_cam, m_asp, device=self.device)
            )
            phi_res = sigma_angle * torch.randn(n_cam, half_d - m_asp, device=self.device)
            phi_cam = torch.cat([phi_asp, phi_res], dim=1)
        else:
            phi_cam = torch.randn(n_cam, half_d, device=self.device) * sigma_angle
        cos_cam = torch.cos(phi_cam)
        sin_cam = torch.sin(phi_cam)

        # Camera-specific vMF concentrations kappa_c calibrated to intra_cos_mean, vmf_kappa_mean & cam drift
        # E[e_i^T e_j] = A_d(kappa)^2 * E[cos(phi_ci - phi_cj)] => A_d(kappa) = sqrt(mu_pos / E_rot)
        if self.use_vmf and aspect_excess > 0.0:
            att_asp = 0.5 * (1.0 + math.cos(2.0 * asp_strength)) * math.exp(-((sigma_angle * 0.55) ** 2))
            f_asp = m_asp / max(1.0, float(half_d))
            drift_attenuation = f_asp * att_asp + (1.0 - f_asp) * math.exp(-(sigma_angle ** 2))
        else:
            drift_attenuation = math.exp(-(sigma_angle ** 2))
        target_ad = math.sqrt(float(np.clip(self.profile.intra_cos_mean / max(0.32, drift_attenuation), 0.25, 0.95)))
        kappa_prop1 = (d - 1.0) / max(0.025, 2.0 * (1.0 - target_ad))
        dim_scale = d / 256.0
        kappa_base = 0.65 * kappa_prop1 + 0.35 * (self.profile.vmf_kappa_mean * dim_scale)
        cam_kappa = torch.clamp(
            kappa_base + (self.profile.vmf_kappa_std * dim_scale) * torch.randn(n_cam, device=self.device),
            min=60.0 * dim_scale,
            max=2400.0 * dim_scale,
        )

        # Data-calibrated background cone bias + 18 color-body cones + hard-negative trim twins from SeedProfile
        bg_vec = torch.nn.functional.normalize(torch.randn(1, d, device=self.device), dim=-1)
        bg_var = float(np.clip(max(0.0, self.profile.inter_cos_mean), 0.002, 0.25))
        bg_w = math.sqrt(bg_var)
        cluster_var = float(np.clip(self.profile.inter_cos_std * 1.35, 0.04, 0.26))

        n_clusters = 18
        cluster_ranks = np.arange(1, n_clusters + 1, dtype=np.float64)
        cluster_probs = cluster_ranks ** (-0.85)
        cluster_probs /= cluster_probs.sum()

        cluster_centers = torch.nn.functional.normalize(
            bg_w * bg_vec + math.sqrt(1.0 - bg_w ** 2) * torch.randn(n_clusters, d, device=self.device),
            dim=-1,
        )

        if self.use_hard_negatives:
            grp_sz = max(3, int(self.profile.hard_neg_group_size))
            n_trims = max(24, n_veh // grp_sz)
            trim_to_cluster = rng.choice(n_clusters, size=n_trims, p=cluster_probs)
            trim_centers = torch.nn.functional.normalize(
                math.sqrt(cluster_var) * cluster_centers[torch.from_numpy(trim_to_cluster).long().to(self.device)]
                + math.sqrt(1.0 - cluster_var) * torch.nn.functional.normalize(torch.randn(n_trims, d, device=self.device), dim=-1),
                dim=-1,
            )
            veh_trim_np = np.arange(n_veh) % n_trims
            rng.shuffle(veh_trim_np)
            veh_cluster_np = trim_to_cluster[veh_trim_np]
            veh_cluster_id = torch.from_numpy(veh_cluster_np).long().to(self.device)
            veh_trim_id = torch.from_numpy(veh_trim_np).long().to(self.device)

            eff_pair_atten = max(0.38, (target_ad ** 2) * math.sqrt(max(0.25, drift_attenuation)))
            twin_target = float(
                np.clip((self.profile.hard_neg_cos_mean - 0.08) / eff_pair_atten, 0.36, 0.90)
            )
            twin_w = math.sqrt(twin_target)
            indiv_noise = torch.nn.functional.normalize(torch.randn(n_veh, d, device=self.device), dim=-1)
            veh_proto = torch.nn.functional.normalize(
                twin_w * trim_centers[veh_trim_id] + math.sqrt(1.0 - twin_w ** 2) * indiv_noise,
                dim=-1,
            )
        else:
            veh_cluster_np = rng.choice(n_clusters, size=n_veh, p=cluster_probs)
            veh_cluster_id = torch.from_numpy(veh_cluster_np).long().to(self.device)
            indiv_noise = torch.nn.functional.normalize(torch.randn(n_veh, d, device=self.device), dim=-1)
            veh_proto = torch.nn.functional.normalize(
                math.sqrt(max(0.01, self.profile.inter_cos_mean)) * bg_vec
                + math.sqrt(1.0 - max(0.01, self.profile.inter_cos_mean)) * indiv_noise,
                dim=-1,
            )

        # Pre-rotate vehicle prototypes into W0^T coordinate frame for O(N*d) camera SO(d) application!
        veh_proto_w = veh_proto @ q_mat  # [N_veh, d]

        bh_centers = cluster_centers
        bh_centers_w = bh_centers @ q_mat

        # Optional privacy-preserving random orthogonal projection matrix P_priv in O(d)
        p_priv = None
        if self.spec.sigma_dp > 0.0 or self.spec.epsilon_dp > 0.0:
            p_priv, _ = torch.linalg.qr(torch.randn(d, d, device=self.device))

        return {
            "q_mat": q_mat,
            "cos_cam": cos_cam,
            "sin_cam": sin_cam,
            "cam_kappa": cam_kappa,
            "veh_proto_w": veh_proto_w,
            "bh_centers": bh_centers,
            "bh_centers_w": bh_centers_w,
            "veh_cluster_id": veh_cluster_id,
            "p_priv": p_priv,
        }

    def _synthesize_batch_mobility(
        self,
        veh_start: int,
        veh_end: int,
        steps_h: int,
        rng: np.random.Generator,
        cam_clock_offset: np.ndarray,
        rush_center: np.ndarray,
        max_radius: float,
        return_unoccluded_tape: bool = False,
    ):
        """
        Engine 1: Vectorized Macro-Micro Coupled Mobility synthesis for a batch of vehicles.
        Returns (snap_veh, snap_cam, snap_obs_time, snap_true_time, convoy_pairs_in_batch).
        """
        b_size = veh_end - veh_start
        n_cam = self.topology.n_cam
        max_deg = self.topology.max_deg

        cams_mat = np.empty((b_size, steps_h), dtype=np.int32)
        times_mat = np.empty((b_size, steps_h), dtype=np.float64)

        cur_cam = rng.integers(0, n_cam, size=b_size, dtype=np.int32)
        prev_cam = np.full(b_size, -1, dtype=np.int32)
        dest_angle = rng.uniform(0.0, 2.0 * math.pi, size=b_size).astype(np.float32)
        cur_t = rng.uniform(0.0, self.spec.horizon_sec * 0.75, size=b_size)

        cams_mat[:, 0] = cur_cam
        times_mat[:, 0] = cur_t

        wave_vel = np.array([0.85, 0.65], dtype=np.float32)  # m/s urban congestion wave propagation

        for h in range(1, steps_h):
            nbrs = self.topology.nbr_table[cur_cam]         # [B, D_max]
            valid = nbrs >= 0
            hdg = self.topology.edge_heading[cur_cam]       # [B, D_max]
            fft = self.topology.free_flow_time[cur_cam]     # [B, D_max]

            ang_diff = np.abs(np.arctan2(
                np.sin(hdg - dest_angle[:, None]),
                np.cos(hdg - dest_angle[:, None]),
            ))
            is_back = (nbrs == prev_cam[:, None])
            safe_nbrs = np.where(valid, nbrs, 0)
            art_bonus = np.where(self.topology.is_arterial[safe_nbrs], -0.35, 0.0)

            logits = -2.2 * (ang_diff + 3.5 * is_back + art_bonus)
            logits = np.where(valid, logits, -1e9)
            # Vectorized Gumbel-Max categorical transition
            gumbel = -np.log(-np.log(np.clip(rng.random((b_size, max_deg)), 1e-12, 1.0 - 1e-12)))
            choice_k = np.argmax(logits + gumbel, axis=1)

            row_idx = np.arange(b_size)
            nxt_cam = nbrs[row_idx, choice_k]
            step_fft = fft[row_idx, choice_k].astype(np.float64)

            if self.use_congestion_wave:
                c_xy = self.topology.coords[cur_cam]
                wave_front = rush_center[None, :] + (cur_t[:, None] * 0.25) * wave_vel[None, :]
                dist_norm = np.linalg.norm(c_xy - wave_front, axis=1) / max_radius
                temporal_sin = np.sin(np.pi * np.clip(cur_t / self.spec.horizon_sec, 0.0, 1.0)) ** 2
                phi_wave = 1.0 + self.spec.congestion_amplitude * np.exp(-2.2 * (dist_norm ** 2)) * temporal_sin
                p_queue = np.clip(self.profile.queue_prob_base * phi_wave, 0.05, 0.72)
                is_queued = rng.random(b_size) < p_queue

                z_norm = rng.standard_normal(b_size)
                f_ratio = float(getattr(self.profile, "travel_free_ratio", 0.90))
                free_base = f_ratio * np.where(step_fft < 10.0, 0.92, 1.04)
                sig_free = float(np.clip(self.profile.travel_lognorm_sigma_free, 0.10, 0.55))
                dt_free = free_base * step_fft * (phi_wave ** 0.65) * np.exp(
                    sig_free * z_norm
                )
                # Intra-intersection overlapping cameras (step_fft < 8s) do not incur an independent full red-light cycle
                sig_scale = np.clip((step_fft - 2.0) / 12.0, 0.12, 1.0)
                sig_cong = float(np.clip(self.profile.travel_lognorm_sigma_cong, 0.12, 0.60))
                dt_cong = (free_base * step_fft * phi_wave + self.profile.signal_delay_mean_sec * 0.85 * sig_scale) * np.exp(
                    sig_cong * z_norm - 0.06
                )
                dt = np.where(is_queued, dt_cong, dt_free)
            else:
                # Stationary unimodal free-flow walk without macroscopic congestion wave or signal queue mixture
                f_ratio = float(getattr(self.profile, "travel_free_ratio", 0.90))
                dt = (0.88 * f_ratio) * step_fft * np.exp(0.14 * rng.standard_normal(b_size))

            min_step_dt = np.minimum(2.0, np.maximum(0.25, step_fft * 0.30))
            cur_t = cur_t + np.maximum(min_step_dt, dt)
            cams_mat[:, h] = nxt_cam
            times_mat[:, h] = cur_t
            prev_cam = cur_cam
            cur_cam = nxt_cam

        # Embed explicit Co-Movement Convoy Groups & Delayed Follower Pairs in batch
        convoy_pairs: List[Tuple[int, int]] = []
        n_conv = int(b_size * self.spec.convoy_ratio)
        ptr = 0
        while ptr + 1 < n_conv:
            g_sz = 2 if (ptr + 2 >= n_conv or rng.random() < 0.7) else 3
            leader = ptr
            for m_idx in range(1, g_sz):
                follower = ptr + m_idx
                cams_mat[follower, :] = cams_mat[leader, :]
                headway = float(rng.uniform(2.0, 6.5)) * m_idx
                times_mat[follower, :] = times_mat[leader, :] + headway + rng.normal(0.0, 0.75, size=steps_h)
            for a_i in range(g_sz):
                for b_i in range(a_i + 1, g_sz):
                    convoy_pairs.append((veh_start + ptr + a_i, veh_start + ptr + b_i))
            ptr += g_sz

        # Embed delayed follower co-movement trajectories (trailing by 9-13s with optional 1-camera branch detour)
        n_follow = min(self.spec.follower_pairs, max(0, (b_size - n_conv) // 4))
        follow_base = n_conv
        for s_i in range(n_follow):
            target_local = follow_base + 2 * s_i
            follower_local = follow_base + 2 * s_i + 1
            if follower_local >= b_size:
                break
            cams_mat[follower_local, :] = cams_mat[target_local, :]
            lag = float(rng.uniform(8.5, 13.0))
            times_mat[follower_local, :] = times_mat[target_local, :] + lag + rng.normal(0.0, 1.0, size=steps_h)
            # Optional 1-hop parallel detour in the middle if steps_h >= 7
            if steps_h >= 7:
                mid = steps_h // 2
                u_prev = cams_mat[target_local, mid - 1]
                u_next = cams_mat[target_local, mid + 1]
                common_nbrs = [
                    nb for nb in self.topology.adj[u_prev]
                    if nb != cams_mat[target_local, mid] and u_next in self.topology.adj[nb]
                ]
                if common_nbrs:
                    cams_mat[follower_local, mid] = common_nbrs[0]
            convoy_pairs.append((veh_start + target_local, veh_start + follower_local))

        # Bimodal urban OD trip length: 52% arterial through-commuters, 48% local errands
        commuter_mask = rng.random(b_size) < 0.52
        if steps_h < 10:
            long_lo = max(3, steps_h - 3)
            short_lo = 2
            short_hi = max(3, int(round(steps_h * 0.45)))
        else:
            long_lo = max(6, steps_h - 4)
            short_lo = 4
            short_hi = max(6, int(round(steps_h * 0.40)))
        long_lens = rng.integers(long_lo, steps_h + 1, size=b_size)
        short_lens = rng.integers(short_lo, short_hi + 1, size=b_size)
        veh_lengths = np.where(commuter_mask, long_lens, short_lens)
        # Ensure explicit convoy & follower corridors traverse full steps_h hops for long-range Q2 evaluation
        paired_end = min(b_size, n_conv + 2 * n_follow)
        if paired_end > 0:
            veh_lengths[:paired_end] = steps_h
        step_idx_grid = np.arange(steps_h)[None, :]
        in_trip_mask = step_idx_grid < veh_lengths[:, None]

        occ_draw = rng.random((b_size, steps_h))
        # Keep endpoints more reliably when rho_occ < 0.45 so trajectories remain observable
        keep_occ = occ_draw >= self.spec.rho_occ
        if self.spec.rho_occ < 0.48:
            keep_occ[:, 0] = True
            last_indices = np.clip(veh_lengths - 1, 0, steps_h - 1)
            keep_occ[np.arange(b_size), last_indices] = True

        final_mask = in_trip_mask & keep_occ
        v_ids_grid = np.broadcast_to((np.arange(veh_start, veh_end, dtype=np.int32))[:, None], (b_size, steps_h))

        if return_unoccluded_tape:
            full_veh = v_ids_grid[in_trip_mask]
            full_cam = cams_mat[in_trip_mask]
            full_true_t = times_mat[in_trip_mask]
            full_keep_occ = keep_occ[in_trip_mask]
            full_jitter = rng.normal(0.0, 0.45, size=full_true_t.shape[0])
            full_obs_t = full_true_t + cam_clock_offset[full_cam] + full_jitter
            return full_veh, full_cam, full_obs_t, full_true_t, full_keep_occ, full_jitter, convoy_pairs

        flat_veh = v_ids_grid[final_mask]
        flat_cam = cams_mat[final_mask]
        flat_true_t = times_mat[final_mask]
        flat_obs_t = flat_true_t + cam_clock_offset[flat_cam] + rng.normal(0.0, 0.45, size=flat_true_t.shape[0])

        return flat_veh, flat_cam, flat_obs_t, flat_true_t, convoy_pairs

    def _synthesize_batch_embeddings(
        self,
        flat_veh: np.ndarray,
        flat_cam: np.ndarray,
        flat_obs_t: np.ndarray,
        manifold: Dict[str, torch.Tensor],
        rng: np.random.Generator,
        return_deglared: bool = False,
    ):
        """
        Engine 2: GPU-accelerated Manifold-Preserving Visual Embedding & Black-Hole Defect Synthesizer.
        """
        n_snaps = flat_veh.shape[0]
        d = self.spec.embed_dim

        v_t = torch.from_numpy(flat_veh).long().to(self.device)
        c_t = torch.from_numpy(flat_cam).long().to(self.device)

        # 1. Apply camera-specific special-orthogonal viewpoint drift R_c = W0 G(phi_c) W0^T in O(N*d)
        proto_w = manifold["veh_proto_w"][v_t]      # [N, d]
        cos_c = manifold["cos_cam"][c_t]            # [N, d/2]
        sin_c = manifold["sin_cam"][c_t]            # [N, d/2]

        x_even = proto_w[:, 0::2]
        x_odd = proto_w[:, 1::2]
        rot_w = torch.empty_like(proto_w)
        rot_w[:, 0::2] = cos_c * x_even - sin_c * x_odd
        rot_w[:, 1::2] = sin_c * x_even + cos_c * x_odd
        del proto_w, cos_c, sin_c, x_even, x_odd
        mu_vc = rot_w @ manifold["q_mat"].T         # [N, d] on S^{d-1}
        del rot_w
        mu_vc = torch.nn.functional.normalize(mu_vc, dim=-1)

        # 2. Sample directional dispersion on S^{d-1}: exact vMF vs. uncalibrated isotropic tangent noise
        if self.use_vmf:
            kappa_s = manifold["cam_kappa"][c_t]
            w_rad = sample_vmf_radial_w(kappa_s, d, self.device)
            del kappa_s
            xi = torch.randn(n_snaps, d, device=self.device)
            v_perp = xi - (xi * mu_vc).sum(dim=-1, keepdim=True) * mu_vc
            del xi
            v_perp = torch.nn.functional.normalize(v_perp, dim=-1)
            emb = w_rad.unsqueeze(-1) * mu_vc + torch.sqrt(torch.clamp(1.0 - w_rad * w_rad, min=0.0)).unsqueeze(-1) * v_perp
            del mu_vc, v_perp, w_rad
            emb = torch.nn.functional.normalize(emb, dim=-1)
        else:
            # Ablation w/o vMF & SO(d) Manifold: uncalibrated isotropic Euclidean Gaussian noise without SO(d)/vMF concentration
            del mu_vc
            raw_proto = manifold["veh_proto_w"][v_t] @ manifold["q_mat"].T
            emb = torch.nn.functional.normalize(
                raw_proto + 0.036 * torch.randn(n_snaps, d, device=self.device), dim=-1
            )
            del raw_proto
        del v_t, c_t
        emb_deglared = emb.clone() if return_deglared else None

        # 3. Embedding Black-Hole (Hubness) Defect Injection (nonlinear multi-mode glare & channel saturation)
        is_bh = np.zeros(n_snaps, dtype=bool)
        alpha_eff = self.spec.alpha_hub if self.use_blackhole else 0.0
        n_bh_snaps = int(round(n_snaps * alpha_eff))
        if n_bh_snaps > 0 and n_snaps > 80:
            t_ord = np.argsort(flat_obs_t)
            ord_vehs = flat_veh[t_ord]
            ord_all_cams = flat_cam[t_ord]
            ord_times = flat_obs_t[t_ord]

            # Vectorized identification of mid-trajectory snapshots in chronological order
            _, first_pos, inv_veh = np.unique(ord_vehs, return_index=True, return_inverse=True)
            _, rev_first = np.unique(ord_vehs[::-1], return_index=True)
            last_pos = (n_snaps - 1) - rev_first
            pos_grid = np.arange(n_snaps)
            is_mid_ord = (pos_grid > first_pos[inv_veh]) & (pos_grid < last_pos[inv_veh])

            veh_cid_np = manifold["veh_cluster_id"].cpu().numpy()
            lo_pos = max(1, int(n_snaps * 0.05))
            hi_pos = max(lo_pos + 1, int(n_snaps * 0.95))
            valid_pos = np.arange(lo_pos, hi_pos)
            ord_cams = ord_all_cams[valid_pos]
            mid_w = np.where(is_mid_ord[valid_pos], 3.0, 0.25)
            art_w = np.where(self.topology.is_arterial[ord_cams], 2.8, 0.5)
            pop_w = np.where(veh_cid_np[ord_vehs[valid_pos]] < 5, 2.8, 0.5)
            bh_weights = mid_w * art_w * pop_w
            bh_weights /= bh_weights.sum()

            bh_centers = manifold["bh_centers"]
            sat_gain = 1.65 * math.sqrt(d)

            # Unified GPU-vectorized spatio-temporal event-wave glare kernel across ALL scales (40 to 5,000+ cameras)
            n_events = max(1, n_bh_snaps // 2)
            event_pos = rng.choice(valid_pos, size=min(n_events, len(valid_pos)), replace=False, p=bh_weights)
            n_ev = len(event_pos)
            ev_cams = ord_all_cams[event_pos]
            ev_times = ord_times[event_pos]

            # Scale-proportional glare plateau cap so partial-glare tuples stay O(alpha_hub * N) at all scales
            max_total_plat = max(2, int(round(n_bh_snaps * (1.15 if n_snaps < 8000 else 2.2))))
            plat_per_ev = max(1, min(10, int(math.ceil(max_total_plat / max(1, n_ev)))))

            # Vectorized (camera, 26s time-bin) grid assignment covering exact +-52.0s road-adjacent zones
            t_base = float(ord_times[0])
            bin_w = 26.0
            n_bins = int((float(ord_times[-1]) - t_base) // bin_w) + 3
            snap_bins = np.clip(((ord_times - t_base) // bin_w).astype(np.int32), 0, n_bins - 1)
            ev_bins = np.clip(((ev_times - t_base) // bin_w).astype(np.int32), 0, n_bins - 1)

            grid_ev = np.full((self.topology.n_cam, n_bins), -1, dtype=np.int32)
            ev_rev = np.arange(n_ev - 1, -1, -1, dtype=np.int32)
            ev_cams_rev = ev_cams[ev_rev]
            ev_bins_rev = ev_bins[ev_rev]
            nbr_rev = self.topology.nbr_table[ev_cams_rev]
            for db in (-2, 2, -1, 1, 0):
                b_idx = np.clip(ev_bins_rev + db, 0, n_bins - 1)
                for k_nb in range(self.topology.max_deg):
                    nb_c = nbr_rev[:, k_nb]
                    v_nb = nb_c >= 0
                    if np.any(v_nb):
                        grid_ev[nb_c[v_nb], b_idx[v_nb]] = ev_rev[v_nb]
                grid_ev[ev_cams_rev, b_idx] = ev_rev

            cand_ev = grid_ev[ord_all_cams, snap_bins]
            del grid_ev
            valid_zone = (cand_ev >= 0) & is_mid_ord
            if np.any(valid_zone):
                vz_pos = np.nonzero(valid_zone)[0]
                vz_ev = cand_ev[vz_pos]
                dt_ok = np.abs(ord_times[vz_pos] - ev_times[vz_ev]) <= 52.0
                vz_pos = vz_pos[dt_ok]
                vz_ev = vz_ev[dt_ok]
            else:
                vz_pos = np.empty(0, dtype=np.int64)
                vz_ev = np.empty(0, dtype=np.int32)

            touched_ord = np.zeros(n_snaps, dtype=bool)
            if len(vz_pos) > 0:
                ev_sort = np.argsort(vz_ev, kind="mergesort")
                s_pos = vz_pos[ev_sort]
                s_ev = vz_ev[ev_sort]
                is_new_ev = np.empty(len(s_ev), dtype=bool)
                is_new_ev[0] = True
                is_new_ev[1:] = s_ev[1:] != s_ev[:-1]
                ev_start_idx = np.maximum.accumulate(np.where(is_new_ev, np.arange(len(s_ev)), 0))
                rank_in_ev = np.arange(len(s_ev)) - ev_start_idx

                core_sel = rank_in_ev < 2
                c_pos_ord = s_pos[core_sel][:n_bh_snaps]
                c_ev_ids = s_ev[core_sel][:n_bh_snaps]
                touched_ord[c_pos_ord] = True

                plat_sel = (rank_in_ev >= 2) & (rank_in_ev < 2 + plat_per_ev)
                p_pos_ord = s_pos[plat_sel][:max_total_plat]
                p_ev_ids = s_ev[plat_sel][:max_total_plat]
                touched_ord[p_pos_ord] = True
            else:
                c_pos_ord = np.empty(0, dtype=np.int64)
                c_ev_ids = np.empty(0, dtype=np.int32)
                p_pos_ord = np.empty(0, dtype=np.int64)
                p_ev_ids = np.empty(0, dtype=np.int32)

            core_orig_np = t_ord[c_pos_ord]
            core_ev_np = c_ev_ids.astype(np.int64)
            if len(core_orig_np) < n_bh_snaps:
                rem_ord = valid_pos[~touched_ord[valid_pos]]
                ex_n = min(n_bh_snaps - len(core_orig_np), len(rem_ord))
                if ex_n > 0:
                    ex_sel = rng.choice(rem_ord, size=ex_n, replace=False)
                    touched_ord[ex_sel] = True
                    core_orig_np = np.concatenate([core_orig_np, t_ord[ex_sel]])
                    core_ev_np = np.concatenate([core_ev_np, np.arange(ex_n, dtype=np.int64) % max(1, n_ev)])

            plat_orig_np = t_ord[p_pos_ord]
            plat_ev_np = p_ev_ids.astype(np.int64)

            n_poles = max(4, min(24, self.topology.n_cam // 4))
            pole_probs = np.arange(1, n_poles + 1, dtype=np.float64) ** (-1.15)
            pole_probs /= pole_probs.sum()
            pole_assign = torch.from_numpy(rng.choice(n_poles, size=n_ev, p=pole_probs)).long().to(self.device)
            ev_idx_t = torch.arange(n_ev, device=self.device)

            is_bh[core_orig_np] = True
            if self.glare_family == "gamma_rank4":
                # Out-of-family rank-4 non-orthogonal glare + asymmetric ReLU & gamma=0.6 power-law channel clipping
                pole_g = torch.nn.functional.normalize(torch.randn(n_poles, 4, d, device=self.device), dim=-1)
                ev_g = pole_g[pole_assign]  # [n_ev, 4, d]
                anchor_g = bh_centers[ev_idx_t % 6]
                g1 = torch.nn.functional.normalize(0.42 * anchor_g + 0.85 * ev_g[:, 0, :], dim=-1)
                g2 = torch.nn.functional.normalize(0.35 * g1 + 0.88 * ev_g[:, 1, :], dim=-1)
                g3 = torch.nn.functional.normalize(0.28 * g1 + 0.25 * g2 + 0.88 * ev_g[:, 2, :], dim=-1)
                g4 = torch.nn.functional.normalize(0.22 * g1 + 0.92 * ev_g[:, 3, :], dim=-1)

                def _apply_gamma_rank4(z_in: torch.Tensor) -> torch.Tensor:
                    z_scaled = z_in * math.sqrt(d)
                    z_rect = torch.relu(z_scaled + 0.35) - 0.18
                    z_pow = torch.sign(z_rect) * torch.pow(torch.abs(z_rect) + 1e-6, 0.60)
                    return torch.nn.functional.normalize(z_pow, dim=-1)

                if len(core_orig_np) > 0:
                    bh_idx_t = torch.from_numpy(core_orig_np).long().to(self.device)
                    cev_t = torch.from_numpy(core_ev_np).long().to(self.device)
                    m_cnt = len(core_orig_np)
                    w2 = (0.32 + 0.18 * torch.rand(m_cnt, 1, device=self.device)) * torch.sign(torch.randn(m_cnt, 1, device=self.device))
                    w3 = (0.24 + 0.16 * torch.rand(m_cnt, 1, device=self.device)) * torch.sign(torch.randn(m_cnt, 1, device=self.device))
                    w4 = (0.18 + 0.14 * torch.rand(m_cnt, 1, device=self.device)) * torch.sign(torch.randn(m_cnt, 1, device=self.device))
                    bh_jitter = torch.nn.functional.normalize(torch.randn(m_cnt, d, device=self.device), dim=-1)
                    raw_core = 0.72 * g1[cev_t] + w2 * g2[cev_t] + w3 * g3[cev_t] + w4 * g4[cev_t] + 0.05 * bh_jitter
                    emb[bh_idx_t] = _apply_gamma_rank4(raw_core)

                if len(plat_orig_np) > 0:
                    pl_t = torch.from_numpy(plat_orig_np).long().to(self.device)
                    pev_t = torch.from_numpy(plat_ev_np).long().to(self.device)
                    p_cnt = len(plat_orig_np)
                    beta_np = rng.uniform(0.72, 0.86, size=p_cnt).astype(np.float32)
                    beta_t = torch.from_numpy(beta_np).unsqueeze(1).to(self.device)
                    w2p = (0.32 + 0.18 * torch.rand(p_cnt, 1, device=self.device)) * torch.sign(torch.randn(p_cnt, 1, device=self.device))
                    w3p = (0.24 + 0.16 * torch.rand(p_cnt, 1, device=self.device)) * torch.sign(torch.randn(p_cnt, 1, device=self.device))
                    w4p = (0.18 + 0.14 * torch.rand(p_cnt, 1, device=self.device)) * torch.sign(torch.randn(p_cnt, 1, device=self.device))
                    illum_vec = torch.nn.functional.normalize(
                        0.72 * g1[pev_t] + w2p * g2[pev_t] + w3p * g3[pev_t] + w4p * g4[pev_t], dim=-1
                    )
                    mixed = torch.sqrt(1.0 - beta_t ** 2) * emb[pl_t] + beta_t * illum_vec
                    emb[pl_t] = _apply_gamma_rank4(mixed)
            else:
                # Exact batched QR orthonormalization for 3-mode local illumination frame (glare_family == "tanh_3mode")
                pole_frames = torch.randn(n_poles, d, 3, device=self.device)
                raw_frames = pole_frames[pole_assign].clone()  # [n_ev, d, 3]
                raw_frames[:, :, 0] = 0.34 * bh_centers[ev_idx_t % 4] + 0.94 * torch.nn.functional.normalize(
                    raw_frames[:, :, 0], dim=-1
                )
                q_frames, r_frames = torch.linalg.qr(raw_frames)
                r_diag = torch.diagonal(r_frames, dim1=-2, dim2=-1)
                r_sign = torch.where(r_diag < 0.0, -torch.ones_like(r_diag), torch.ones_like(r_diag))
                q_frames = q_frames * r_sign.unsqueeze(1)
                ev_m1 = q_frames[:, :, 0]
                ev_m2 = q_frames[:, :, 1]
                ev_m3 = q_frames[:, :, 2]
                del pole_frames, raw_frames, q_frames, r_frames, r_diag, r_sign

                if len(core_orig_np) > 0:
                    bh_idx_t = torch.from_numpy(core_orig_np).long().to(self.device)
                    cev_t = torch.from_numpy(core_ev_np).long().to(self.device)
                    m_cnt = len(core_orig_np)
                    w2 = (0.16 + 0.14 * torch.rand(m_cnt, 1, device=self.device)) * torch.sign(
                        torch.randn(m_cnt, 1, device=self.device)
                    )
                    w3 = (0.08 + 0.08 * torch.rand(m_cnt, 1, device=self.device)) * torch.sign(
                        torch.randn(m_cnt, 1, device=self.device)
                    )
                    bh_jitter = torch.nn.functional.normalize(torch.randn(m_cnt, d, device=self.device), dim=-1)
                    raw_core = 0.94 * ev_m1[cev_t] + w2 * ev_m2[cev_t] + w3 * ev_m3[cev_t] + 0.04 * bh_jitter
                    emb[bh_idx_t] = torch.nn.functional.normalize(torch.tanh(sat_gain * raw_core), dim=-1)

                if len(plat_orig_np) > 0:
                    pl_t = torch.from_numpy(plat_orig_np).long().to(self.device)
                    pev_t = torch.from_numpy(plat_ev_np).long().to(self.device)
                    p_cnt = len(plat_orig_np)
                    beta_np = rng.uniform(0.76, 0.88, size=p_cnt).astype(np.float32)
                    beta_t = torch.from_numpy(beta_np).unsqueeze(1).to(self.device)
                    w2p = (0.18 + 0.14 * torch.rand(p_cnt, 1, device=self.device)) * torch.sign(
                        torch.randn(p_cnt, 1, device=self.device)
                    )
                    w3p = (0.08 + 0.08 * torch.rand(p_cnt, 1, device=self.device)) * torch.sign(
                        torch.randn(p_cnt, 1, device=self.device)
                    )
                    illum_vec = torch.nn.functional.normalize(
                        ev_m1[pev_t] + w2p * ev_m2[pev_t] + w3p * ev_m3[pev_t], dim=-1
                    )
                    mixed = torch.sqrt(1.0 - beta_t ** 2) * emb[pl_t] + beta_t * illum_vec
                    emb[pl_t] = torch.nn.functional.normalize(torch.tanh(sat_gain * mixed), dim=-1)

        # 4. Edge-Cloud Privacy-Preserving Transformation Module (Orthogonal Projection + d_X-Metric / Gaussian DP Noise)
        if manifold["p_priv"] is not None:
            emb = emb @ manifold["p_priv"]
            if emb_deglared is not None:
                emb_deglared = emb_deglared @ manifold["p_priv"]
            sigma_noise = float(self.spec.sigma_dp)
            if self.spec.epsilon_dp > 0.0 and sigma_noise == 0.0:
                # Gaussian mechanism calibrated on S^{d-1} (global Delta_2 = 2.0, delta = 1e-5; local d_X-LDP eps = ||x-x'||_2 * eps_global / 2)
                sigma_noise = (2.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5))) / self.spec.epsilon_dp
            if sigma_noise > 0.0:
                dp_draw = sigma_noise * torch.randn(n_snaps, d, device=self.device)
                emb = torch.nn.functional.normalize(emb + dp_draw, dim=-1)
                if emb_deglared is not None:
                    emb_deglared = torch.nn.functional.normalize(emb_deglared + dp_draw, dim=-1)

        if return_deglared:
            return emb, is_bh, emb_deglared
        return emb, is_bh

    def _effective_sigma_dp(self) -> float:
        """Resolve effective per-coordinate Gaussian DP noise standard deviation."""
        if self.spec.sigma_dp > 0.0:
            return float(self.spec.sigma_dp)
        if self.spec.epsilon_dp > 0.0:
            return float((2.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5))) / self.spec.epsilon_dp)
        return 0.0

    def synthesize(self, stream_only: bool = False, generate_workloads: bool = True) -> Dict:
        """
        Execute end-to-end synthesis.
        If stream_only=True (used for 5.0M CityScale throughput/memory benchmark), processes in
        bounded-memory GPU streaming batches without retaining all 5.0M FP32 vectors in host RAM.
        """
        gc.collect()
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        t_start = time.perf_counter()

        rng = np.random.default_rng(self.spec.seed)
        torch.manual_seed(self.spec.seed)

        n_cam = self.topology.n_cam
        n_veh = self.spec.n_veh
        avg_obs_per_veh = self.spec.target_snapshots / max(1, n_veh)
        avg_raw = avg_obs_per_veh / max(0.25, 1.0 - self.spec.rho_occ)
        steps_h = max(5, int(round(avg_raw * (1.42 if avg_raw < 7.5 else 1.55))))

        cam_clock_offset = rng.normal(0.0, self.spec.sigma_tau, size=n_cam)
        rush_center = self.topology.coords.mean(axis=0)
        max_radius = max(1.0, float(np.linalg.norm(self.topology.coords - rush_center, axis=1).max()))

        manifold = self._prepare_manifold_parameters(rng)

        batch_veh = 50000
        total_snapshots = 0

        all_veh_chunks = []
        all_cam_chunks = []
        all_obs_t_chunks = []
        all_true_t_chunks = []
        all_emb_chunks = []
        all_bh_chunks = []
        all_convoy_pairs: List[Tuple[int, int]] = []

        for v_start in range(0, n_veh, batch_veh):
            v_end = min(n_veh, v_start + batch_veh)
            f_veh, f_cam, f_obs_t, f_true_t, c_pairs = self._synthesize_batch_mobility(
                v_start, v_end, steps_h, rng, cam_clock_offset, rush_center, max_radius
            )
            emb_gpu, is_bh = self._synthesize_batch_embeddings(f_veh, f_cam, f_obs_t, manifold, rng)
            total_snapshots += int(f_veh.shape[0])
            all_convoy_pairs.extend(c_pairs)

            if stream_only:
                # Synchronize GPU wave execution before releasing per-wave CUDA tensors
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                del emb_gpu
            else:
                all_veh_chunks.append(f_veh)
                all_cam_chunks.append(f_cam)
                all_obs_t_chunks.append(f_obs_t)
                all_true_t_chunks.append(f_true_t)
                all_emb_chunks.append(emb_gpu.cpu())
                all_bh_chunks.append(is_bh)

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            gpu_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        else:
            gpu_peak_mb = 0.0

        elapsed_sec = time.perf_counter() - t_start
        cpu_peak_mb = _get_peak_rss_mb()

        if stream_only:
            return {
                "name": f"Portolan ({self.spec.scale_name})",
                "total_snapshots": total_snapshots,
                "elapsed_sec": elapsed_sec,
                "throughput_snaps_per_sec": total_snapshots / max(1e-6, elapsed_sec),
                "peak_gpu_mb": gpu_peak_mb,
                "peak_cpu_mb": cpu_peak_mb,
                "peak_total_mb": max(gpu_peak_mb, cpu_peak_mb),
            }

        snap_veh = np.concatenate(all_veh_chunks)
        snap_cam = np.concatenate(all_cam_chunks)
        snap_obs_t = np.concatenate(all_obs_t_chunks)
        snap_true_t = np.concatenate(all_true_t_chunks)
        is_blackhole = np.concatenate(all_bh_chunks)
        embeddings = torch.cat(all_emb_chunks, dim=0)

        # Sort chronologically to form Snapshots_Stream
        order = np.argsort(snap_obs_t, kind="mergesort")
        snap_veh = snap_veh[order]
        snap_cam = snap_cam[order]
        snap_obs_t = snap_obs_t[order]
        snap_true_t = snap_true_t[order]
        is_blackhole = is_blackhole[order]
        embeddings = embeddings[torch.from_numpy(order).long()]

        gt_trajs: Dict[int, List[int]] = {v: [] for v in range(n_veh)}
        for sid, vid in enumerate(snap_veh.tolist()):
            gt_trajs[vid].append(sid)

        dataset = {
            "name": "Portolan",
            "embeddings": embeddings,
            "snap_veh": snap_veh,
            "snap_cam": snap_cam,
            "snap_time": snap_obs_t,
            "snap_true_time": snap_true_t,
            "cam_clock_offset": cam_clock_offset,
            "is_blackhole": is_blackhole,
            "gt_trajectories": gt_trajs,
            "injected_convoy_pairs": all_convoy_pairs,
            "topology": self.topology,
            "spec": self.spec,
            "effective_sigma_dp": self._effective_sigma_dp(),
            "profile": self.profile,
            "elapsed_sec": elapsed_sec,
            "peak_gpu_mb": gpu_peak_mb,
            "peak_cpu_mb": cpu_peak_mb,
            "total_snapshots": int(snap_veh.shape[0]),
        }

        if generate_workloads:
            dataset["workloads"] = build_multimodal_query_workloads(dataset, rng)

        return dataset

    def synthesize_oracle_waterfall(self, generate_workloads: bool = True) -> Dict[str, Dict]:
        """
        Synthesize the 5-stage Counterfactual Oracle Error-Attribution Waterfall from ONE
        shared latent vehicle route tape (zero PRNG desynchronization across stages):
          - S0_Full_Noisy: Full realistic defects (rho_occ, alpha_hub, R_c/vMF, sigma_tau)
          - S1_Oracle_Detection: Restores occluded detections on identical trips (rho_occ = 0)
          - S2_Oracle_DeGlare: Removes ONLY alpha_hub glare hubs while preserving 100% of R_c & vMF draws
          - S3_Oracle_Viewpoint: Removes camera SO(d) viewpoint drift R_c & finite-kappa dispersion
          - S4_Oracle_Clock_Kin: Synchronizes camera clocks (sigma_tau = 0)
        """
        rng = np.random.default_rng(self.spec.seed)
        torch.manual_seed(self.spec.seed)

        n_cam = self.topology.n_cam
        n_veh = self.spec.n_veh
        d = self.spec.embed_dim
        avg_obs_per_veh = self.spec.target_snapshots / max(1, n_veh)
        avg_raw = avg_obs_per_veh / max(0.25, 1.0 - self.spec.rho_occ)
        steps_h = max(5, int(round(avg_raw * (1.42 if avg_raw < 7.5 else 1.55))))

        cam_clock_offset = rng.normal(0.0, self.spec.sigma_tau, size=n_cam)
        rush_center = self.topology.coords.mean(axis=0)
        max_radius = max(1.0, float(np.linalg.norm(self.topology.coords - rush_center, axis=1).max()))
        manifold = self._prepare_manifold_parameters(rng)

        batch_veh = 50000
        veh_c, cam_c, obs_c, true_c, keep_c, jit_c = [], [], [], [], [], []
        noisy_emb_c, deglared_emb_c, clean_emb_c, bh_c = [], [], [], []
        all_convoy_pairs: List[Tuple[int, int]] = []

        kappa_clean_val = max(1400.0, (d - 1.0) / (2.0 * (1.0 - 0.92)))

        for v_start in range(0, n_veh, batch_veh):
            v_end = min(n_veh, v_start + batch_veh)
            f_veh, f_cam, f_obs_t, f_true_t, f_keep, f_jit, c_pairs = self._synthesize_batch_mobility(
                v_start, v_end, steps_h, rng, cam_clock_offset, rush_center, max_radius,
                return_unoccluded_tape=True,
            )
            emb_noisy_gpu, is_bh, emb_deglared_gpu = self._synthesize_batch_embeddings(
                f_veh, f_cam, f_obs_t, manifold, rng, return_deglared=True
            )

            # Camera-invariant clean vehicle prototypes with high-kappa sensor floor (zero drift, zero Black-Holes)
            v_t = torch.from_numpy(f_veh).long().to(self.device)
            raw_proto = torch.nn.functional.normalize(manifold["veh_proto_w"][v_t] @ manifold["q_mat"].T, dim=-1)
            k_clean = torch.full((f_veh.shape[0],), kappa_clean_val, device=self.device, dtype=torch.float32)
            w_rad = sample_vmf_radial_w(k_clean, d, self.device)
            xi = torch.randn(f_veh.shape[0], d, device=self.device)
            v_perp = torch.nn.functional.normalize(xi - (xi * raw_proto).sum(dim=-1, keepdim=True) * raw_proto, dim=-1)
            emb_clean_gpu = torch.nn.functional.normalize(
                w_rad.unsqueeze(-1) * raw_proto + torch.sqrt(torch.clamp(1.0 - w_rad * w_rad, min=0.0)).unsqueeze(-1) * v_perp,
                dim=-1,
            )

            veh_c.append(f_veh)
            cam_c.append(f_cam)
            obs_c.append(f_obs_t)
            true_c.append(f_true_t)
            keep_c.append(f_keep)
            jit_c.append(f_jit)
            noisy_emb_c.append(emb_noisy_gpu.cpu())
            deglared_emb_c.append(emb_deglared_gpu.cpu())
            clean_emb_c.append(emb_clean_gpu.cpu())
            bh_c.append(is_bh)
            all_convoy_pairs.extend(c_pairs)

        full_veh = np.concatenate(veh_c)
        full_cam = np.concatenate(cam_c)
        full_obs_t = np.concatenate(obs_c)
        full_true_t = np.concatenate(true_c)
        full_keep = np.concatenate(keep_c)
        full_jit = np.concatenate(jit_c)
        full_bh = np.concatenate(bh_c)
        full_emb_noisy = torch.cat(noisy_emb_c, dim=0)
        full_emb_deglared = torch.cat(deglared_emb_c, dim=0)
        full_emb_clean = torch.cat(clean_emb_c, dim=0)
        full_sync_t = full_true_t + 0.05 * full_jit
        latent_uid = np.arange(len(full_veh), dtype=np.int32)

        def _pack_stage(
            st_name: str,
            mask_sel: Optional[np.ndarray],
            obs_times: np.ndarray,
            embs: torch.Tensor,
            bh_arr: np.ndarray,
            clk_off: np.ndarray,
            sp_ov: Dict[str, float],
        ) -> Tuple[Dict, Dict[int, int]]:
            sp_s = copy.deepcopy(self.spec)
            for k_ov, v_ov in sp_ov.items():
                setattr(sp_s, k_ov, v_ov)
            if mask_sel is not None:
                sv = full_veh[mask_sel]
                sc = full_cam[mask_sel]
                so = obs_times[mask_sel]
                st = full_true_t[mask_sel]
                sb = bh_arr[mask_sel]
                se = embs[torch.from_numpy(np.nonzero(mask_sel)[0]).long()]
                su = latent_uid[mask_sel]
            else:
                sv, sc, so, st, sb, se, su = (
                    full_veh.copy(), full_cam.copy(), obs_times.copy(),
                    full_true_t.copy(), bh_arr.copy(), embs.clone(), latent_uid.copy(),
                )
            ord_s = np.argsort(so, kind="mergesort")
            sv, sc, so, st, sb, su = sv[ord_s], sc[ord_s], so[ord_s], st[ord_s], sb[ord_s], su[ord_s]
            se = se[torch.from_numpy(ord_s).long()]
            gt_tr: Dict[int, List[int]] = {v: [] for v in range(n_veh)}
            uid_to_sid: Dict[int, int] = {}
            for sid_i, (vid_i, uid_i) in enumerate(zip(sv.tolist(), su.tolist())):
                gt_tr[vid_i].append(sid_i)
                uid_to_sid[uid_i] = sid_i
            ds_stage = {
                "name": f"Portolan-{st_name}",
                "embeddings": se,
                "snap_veh": sv,
                "snap_cam": sc,
                "snap_time": so,
                "snap_true_time": st,
                "cam_clock_offset": clk_off.copy(),
                "is_blackhole": sb,
                "gt_trajectories": gt_tr,
                "injected_convoy_pairs": all_convoy_pairs,
                "topology": self.topology,
                "spec": sp_s,
                "effective_sigma_dp": self._effective_sigma_dp(),
                "profile": copy.deepcopy(self.profile),
                "elapsed_sec": 0.0,
                "peak_gpu_mb": 0.0,
                "peak_cpu_mb": 0.0,
                "total_snapshots": int(sv.shape[0]),
            }
            return ds_stage, uid_to_sid

        zeros_bh = np.zeros_like(full_bh, dtype=bool)
        zeros_clk = np.zeros_like(cam_clock_offset)

        ds_s0, map_s0 = _pack_stage("S0", full_keep, full_obs_t, full_emb_noisy, full_bh, cam_clock_offset, {})
        ds_s1, map_s1 = _pack_stage("S1", None, full_obs_t, full_emb_noisy, full_bh, cam_clock_offset, {"rho_occ": 0.0})
        ds_s2, map_s2 = _pack_stage("S2", None, full_obs_t, full_emb_deglared, zeros_bh, cam_clock_offset, {"rho_occ": 0.0, "alpha_hub": 0.0})
        ds_s3, map_s3 = _pack_stage("S3", None, full_obs_t, full_emb_clean, zeros_bh, cam_clock_offset, {"rho_occ": 0.0, "alpha_hub": 0.0})
        ds_s4, map_s4 = _pack_stage("S4", None, full_sync_t, full_emb_clean, zeros_bh, zeros_clk, {"rho_occ": 0.0, "alpha_hub": 0.0, "sigma_tau": 0.0})

        out = {
            "S0_Full_Noisy": ds_s0,
            "S1_Oracle_Detection": ds_s1,
            "S2_Oracle_DeGlare": ds_s2,
            "S3_Oracle_Viewpoint": ds_s3,
            "S4_Oracle_Clock_Kin": ds_s4,
        }

        if generate_workloads:
            # Build shared physical entity workloads from true un-occluded latent stream (S4),
            # anchoring Q1 queries on physical snapshots that also survived occlusion in S0!
            wl_ref = build_multimodal_query_workloads(ds_s4, rng)
            rev_s0 = {sid: uid for uid, sid in map_s0.items()}
            elig_vehs = [v for v, sids in ds_s0["gt_trajectories"].items() if len(sids) >= 5]
            q1_cnt = min(self.spec.q1_num_queries, len(elig_vehs))
            q1_vehs = rng.choice(elig_vehs, size=q1_cnt, replace=False)
            shared_anchors: List[Tuple[int, int]] = []
            for vid in q1_vehs:
                s0_sids = ds_s0["gt_trajectories"][int(vid)]
                clean_s0 = [s for s in s0_sids if not ds_s0["is_blackhole"][s]]
                chosen_s0_sid = clean_s0[0] if clean_s0 else s0_sids[0]
                shared_anchors.append((int(vid), int(rev_s0[chosen_s0_sid])))

            for st_key, u_map in [
                ("S0_Full_Noisy", map_s0),
                ("S1_Oracle_Detection", map_s1),
                ("S2_Oracle_DeGlare", map_s2),
                ("S3_Oracle_Viewpoint", map_s3),
                ("S4_Oracle_Clock_Kin", map_s4),
            ]:
                ds_st = out[st_key]
                q1_st = []
                for vid, anchor_uid in shared_anchors:
                    st_sids = ds_st["gt_trajectories"][vid]
                    q1_st.append({
                        "vehicle_id": vid,
                        "seed_snapshot_id": int(u_map[anchor_uid]),
                        "gt_snapshot_ids": st_sids,
                        "gt_cameras": [int(ds_st["snap_cam"][s]) for s in st_sids],
                    })
                ds_st["workloads"] = {
                    "Q1": q1_st,
                    "Q2": copy.deepcopy(wl_ref["Q2"]),
                    "Q3": copy.deepcopy(wl_ref["Q3"]),
                    "Q4": copy.deepcopy(wl_ref["Q4"]),
                }

        return out


def export_standardized_tables(
    dataset: Dict, cpg_out: Optional[List[List[Tuple[int, float]]]] = None
) -> Dict[str, object]:
    """
    Package a synthesized dataset into Portolan's 4 standardized Output-Layer tables/tensors:
      1. Snapshots_Stream(sid, cam_id, t_obs, emb) -- strictly observable attributes (zero GT leak)
      2. GroundTruth_Trajectories(veh_id, hop, sid, t_true, is_bh, convoy_gid)
      3. Competitive_Provenance_Graph(src_sid, dst_sid, rank_k, s_cpg)
      4. Query_Workloads (Q1, Q2, Q3, Q4)
    """
    n_snaps = int(dataset.get("total_snapshots", len(dataset["snap_veh"])))
    sids_arr = np.arange(n_snaps, dtype=np.int32)
    hop_arr = np.zeros(n_snaps, dtype=np.int32)
    for sids in dataset["gt_trajectories"].values():
        for h_idx, sid in enumerate(sids):
            hop_arr[int(sid)] = h_idx

    convoy_pairs = list(dataset.get("injected_convoy_pairs", []))
    veh_to_gid: Dict[int, int] = {}
    next_gid = 0
    for va, vb in convoy_pairs:
        va_i, vb_i = int(va), int(vb)
        gid = veh_to_gid.get(va_i, veh_to_gid.get(vb_i, -1))
        if gid < 0:
            gid = next_gid
            next_gid += 1
        veh_to_gid[va_i] = gid
        veh_to_gid[vb_i] = gid
    if not convoy_pairs and "convoy_groups" in dataset:
        for grp in dataset["convoy_groups"]:
            if len(grp) >= 2:
                gid = next_gid
                next_gid += 1
                for v_member in grp:
                    veh_to_gid[int(v_member)] = gid
                for idx_m in range(1, len(grp)):
                    convoy_pairs.append((int(grp[0]), int(grp[idx_m])))
    convoy_gid_arr = np.array(
        [veh_to_gid.get(int(v), -1) for v in dataset["snap_veh"]], dtype=np.int32
    )

    provenance_edges = []
    if cpg_out is not None:
        for u, nbrs in enumerate(cpg_out):
            for rank_idx, (v, score) in enumerate(nbrs):
                provenance_edges.append((int(u), int(v), int(rank_idx + 1), float(score)))

    if provenance_edges:
        src_sid = np.array([e[0] for e in provenance_edges], dtype=np.int32)
        dst_sid = np.array([e[1] for e in provenance_edges], dtype=np.int32)
        rank_k = np.array([e[2] for e in provenance_edges], dtype=np.int32)
        s_cpg = np.array([e[3] for e in provenance_edges], dtype=np.float32)
    else:
        src_sid = np.empty(0, dtype=np.int32)
        dst_sid = np.empty(0, dtype=np.int32)
        rank_k = np.empty(0, dtype=np.int32)
        s_cpg = np.empty(0, dtype=np.float32)

    return {
        "Snapshots_Stream": {
            "sid": sids_arr,
            "cam_id": dataset["snap_cam"],
            "t_obs": dataset["snap_time"],
            "emb": dataset["embeddings"],
        },
        "GroundTruth_Trajectories": {
            "veh_id": dataset["snap_veh"],
            "hop": hop_arr,
            "sid": sids_arr,
            "t_true": dataset.get("snap_true_time", dataset["snap_time"]),
            "is_bh": dataset.get("is_blackhole", np.zeros(n_snaps, dtype=bool)),
            "convoy_gid": convoy_gid_arr,
            "trajectories_by_vehicle": dataset["gt_trajectories"],
            "convoy_pairs": convoy_pairs,
        },
        "Competitive_Provenance_Graph": {
            "src_sid": src_sid,
            "dst_sid": dst_sid,
            "rank_k": rank_k,
            "s_cpg": s_cpg,
            "num_nodes": n_snaps,
            "top_k_edges": provenance_edges,
        },
        "Query_Workloads": dataset.get("workloads", {}),
    }


def build_multimodal_query_workloads(dataset: Dict, rng: np.random.Generator) -> Dict:
    """
    Engine 3: Multi-Modal Declarative Query Workload Synthesizer (Q1, Q2, Q3, Q4) with exact GT.
    """
    spec: WorkloadSpec = dataset["spec"]
    gt_trajs: Dict[int, List[int]] = dataset["gt_trajectories"]
    snap_cam: np.ndarray = dataset["snap_cam"]
    snap_time: np.ndarray = dataset["snap_time"]
    snap_veh: np.ndarray = dataset["snap_veh"]
    is_bh: np.ndarray = dataset["is_blackhole"]
    n_snaps = len(snap_veh)

    # Q1: Point Trajectory Recovery Queries (seed snapshot -> full GT trajectory sequence)
    eligible_vehs = [v for v, sids in gt_trajs.items() if len(sids) >= 5]
    if len(eligible_vehs) < 25:
        eligible_vehs = [v for v, sids in gt_trajs.items() if len(sids) >= 3]
    if not eligible_vehs:
        eligible_vehs = [v for v, sids in gt_trajs.items() if len(sids) >= 2]
    q1_count = min(spec.q1_num_queries, len(eligible_vehs))
    q1_vehs = rng.choice(eligible_vehs, size=q1_count, replace=False)
    q1_queries = []
    for vid in q1_vehs:
        sids = gt_trajs[int(vid)]
        # Pick a non-black-hole anchor snapshot near the start/middle of the trajectory
        clean_sids = [s for s in sids if not is_bh[s]]
        seed_sid = clean_sids[0] if clean_sids else sids[0]
        q1_queries.append({
            "vehicle_id": int(vid),
            "seed_snapshot_id": int(seed_sid),
            "gt_snapshot_ids": sids,
            "gt_cameras": [int(snap_cam[s]) for s in sids],
        })

    # Q2: Zero-Image Co-Movement / Convoy Relational Join (exact GT co-traversing vehicle pairs)
    # Build inverted camera-visit buckets to compute exact ground-truth pairs for k in [2..8]
    cam_visits: Dict[int, List[Tuple[float, int]]] = {}
    for vid, sids in gt_trajs.items():
        seen_c = set()
        for s in sids:
            c = int(snap_cam[s])
            if c not in seen_c:
                seen_c.add(c)
                cam_visits.setdefault(c, []).append((float(snap_time[s]), int(vid)))

    pair_shared_cams: Dict[Tuple[int, int], int] = {}
    delta_t = spec.q2_time_window_sec
    for c, visits in cam_visits.items():
        visits.sort(key=lambda x: x[0])
        m = len(visits)
        for i in range(m):
            ti, vi = visits[i]
            j = i + 1
            while j < m and (visits[j][0] - ti) <= delta_t:
                vj = visits[j][1]
                if vi != vj:
                    pair = (vi, vj) if vi < vj else (vj, vi)
                    pair_shared_cams[pair] = pair_shared_cams.get(pair, 0) + 1
                j += 1

    gt_q2_by_k: Dict[int, set] = {}
    for k_hops in range(2, 9):
        gt_q2_by_k[k_hops] = {pair for pair, cnt in pair_shared_cams.items() if cnt >= k_hops}

    # Q3: Cross-Scale Corridor Flow & OD Aggregation Query (turn-level transition flows & top bottlenecks)
    od_transition_counts: Dict[Tuple[int, int], int] = {}
    corridor_3hop_counts: Dict[Tuple[int, int, int], int] = {}
    for vid, sids in gt_trajs.items():
        cams = [int(snap_cam[s]) for s in sids]
        for i in range(len(cams) - 1):
            if cams[i] != cams[i + 1]:
                edge = (cams[i], cams[i + 1])
                od_transition_counts[edge] = od_transition_counts.get(edge, 0) + 1
        for i in range(len(cams) - 2):
            if cams[i] != cams[i + 1] and cams[i + 1] != cams[i + 2]:
                tri = (cams[i], cams[i + 1], cams[i + 2])
                corridor_3hop_counts[tri] = corridor_3hop_counts.get(tri, 0) + 1

    sorted_corridors = sorted(corridor_3hop_counts.items(), key=lambda x: x[1], reverse=True)
    top_bottlenecks = [cor for cor, _ in sorted_corridors[:25]]

    # Q4: Incremental View Maintenance & What-If Transaction Workload
    n_cert = max(10, int(round(n_snaps * spec.q4_cert_budget_ratio)))
    unlearn_vehs = [
        int(v) for v in rng.choice(
            eligible_vehs, size=min(spec.q4_unlearn_vehicles, len(eligible_vehs)), replace=False
        )
    ]
    n_drop_cams = max(1, int(round(dataset["topology"].n_cam * spec.q4_dropout_cam_ratio)))
    dropout_cams = [
        int(c) for c in rng.choice(dataset["topology"].n_cam, size=n_drop_cams, replace=False)
    ]

    return {
        "Q1": q1_queries,
        "Q2": {
            "delta_t_sec": delta_t,
            "default_k": spec.q2_min_hops,
            "gt_pairs_by_k": gt_q2_by_k,
        },
        "Q3": {
            "gt_od_counts": od_transition_counts,
            "gt_corridor_3hop": corridor_3hop_counts,
            "gt_top_bottlenecks": top_bottlenecks,
        },
        "Q4": {
            "cert_budget_count": n_cert,
            "unlearn_vehicle_ids": unlearn_vehs,
            "whatif_dropout_cameras": dropout_cams,
            "clock_calibration_cams": dropout_cams,
        },
    }
