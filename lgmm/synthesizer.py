"""
Category-A Synthesis Baseline 3 — Latent-GMM-Synth (L-GMM):
Euclidean low-rank Gaussian Mixture Model embedding generator in R^d projected onto S^{d-1}
without camera SO(d) viewpoint rotations or glare Black-Hole hubs, paired with non-backtracking
topological walks decoupled from macroscopic arterial congestion waves.
"""

from typing import Dict
import gc
import sys
import time
try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]
import numpy as np
import torch

from portolan.profiles_and_topology import (
    RoadCameraTopology,
    SeedProfile,
    WorkloadSpec,
)
from portolan.synthesizer import build_multimodal_query_workloads


def _get_peak_rss_mb() -> float:
    """Return peak process RSS in MiB across Linux (KiB) and macOS/Darwin (bytes)."""
    if resource is None:
        return 0.0
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0


class LatentGMMSynthesizer:
    """
    Latent-GMM-Synth (L-GMM):
    Low-rank Euclidean Gaussian Mixture Model embedding generator paired with static
    free-flow-scaled non-backtracking walks decoupled from congestion waves and Black-Holes.
    """

    def __init__(
        self,
        profile: SeedProfile,
        spec: WorkloadSpec,
        topology: RoadCameraTopology,
        device: torch.device,
    ):
        self.profile = profile
        self.spec = spec
        self.topology = topology
        self.device = device

    def synthesize(self, stream_only: bool = False, generate_workloads: bool = True) -> Dict:
        """Synthesize decoupled Euclidean low-rank GMM embeddings and static link travel times."""
        gc.collect()
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        t_start = time.perf_counter()

        rng = np.random.default_rng(self.spec.seed + 202)
        torch.manual_seed(self.spec.seed + 202)

        n_cam = self.topology.n_cam
        n_veh = self.spec.n_veh
        d = self.spec.embed_dim
        steps_h = max(4, int(round(self.spec.target_snapshots / max(1, n_veh * (1.0 - self.spec.rho_occ)))))

        n_comp = min(256, max(32, n_veh // 20))
        comp_means = torch.randn(n_comp, d, device=self.device) * 0.45
        comp_cov_factors = torch.randn(n_comp, 16, d, device=self.device) * 0.12
        veh_comp = torch.arange(n_veh, device=self.device) % n_comp
        veh_means = comp_means[veh_comp] + 0.65 * torch.randn(n_veh, d, device=self.device)

        batch_veh = 50000
        total_snapshots = 0
        veh_chunks, cam_chunks, time_chunks, emb_chunks = [], [], [], []

        for v_start in range(0, n_veh, batch_veh):
            v_end = min(n_veh, v_start + batch_veh)
            b_size = v_end - v_start
            cams_mat = np.empty((b_size, steps_h), dtype=np.int32)
            times_mat = np.empty((b_size, steps_h), dtype=np.float64)

            cur_cam = rng.integers(0, n_cam, size=b_size, dtype=np.int32)
            prev_cam = np.full(b_size, -1, dtype=np.int32)
            cur_t = rng.uniform(0.0, self.spec.horizon_sec * 0.75, size=b_size)
            cams_mat[:, 0] = cur_cam
            times_mat[:, 0] = cur_t

            for h in range(1, steps_h):
                nbrs = self.topology.nbr_table[cur_cam]
                valid = nbrs >= 0
                fft = self.topology.free_flow_time[cur_cam]
                logits = np.where(valid, np.where(nbrs == prev_cam[:, None], -2.5, 0.0), -1e9)
                gumbel = -np.log(-np.log(np.clip(rng.random((b_size, self.topology.max_deg)), 1e-12, 1.0 - 1e-12)))
                choice_k = np.argmax(logits + gumbel, axis=1)
                row_idx = np.arange(b_size)
                nxt_cam = nbrs[row_idx, choice_k]
                dt = fft[row_idx, choice_k] * rng.uniform(0.9, 1.25, size=b_size)
                cur_t = cur_t + dt
                cams_mat[:, h] = nxt_cam
                times_mat[:, h] = cur_t
                prev_cam = cur_cam
                cur_cam = nxt_cam

            keep = rng.random((b_size, steps_h)) >= self.spec.rho_occ
            keep[:, 0] = True
            v_grid = np.broadcast_to(np.arange(v_start, v_end, dtype=np.int32)[:, None], (b_size, steps_h))
            f_veh = v_grid[keep]
            f_cam = cams_mat[keep]
            f_t = times_mat[keep]

            n_s = f_veh.shape[0]
            v_t = torch.from_numpy(f_veh).long().to(self.device)
            c_id = veh_comp[v_t]
            cov_noise = torch.empty((n_s, d), dtype=torch.float32, device=self.device)
            sub_chunk = 250000
            for s0 in range(0, n_s, sub_chunk):
                s1 = min(n_s, s0 + sub_chunk)
                z_sub = torch.randn(s1 - s0, 1, 16, device=self.device)
                cf_sub = comp_cov_factors[c_id[s0:s1]]
                cov_noise[s0:s1] = torch.bmm(z_sub, cf_sub).squeeze(1)
                del z_sub, cf_sub
            del c_id
            raw_vec = veh_means[v_t] + cov_noise + 0.22 * torch.randn(n_s, d, device=self.device)
            del cov_noise, v_t
            emb_gpu = torch.nn.functional.normalize(raw_vec, dim=-1)
            del raw_vec
            total_snapshots += int(n_s)

            if stream_only:
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                del emb_gpu
            else:
                veh_chunks.append(f_veh)
                cam_chunks.append(f_cam)
                time_chunks.append(f_t)
                emb_chunks.append(emb_gpu.cpu())

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            gpu_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        else:
            gpu_peak_mb = 0.0
        elapsed_sec = time.perf_counter() - t_start
        cpu_peak_mb = _get_peak_rss_mb()

        if stream_only:
            return {
                "name": f"L-GMM ({self.spec.scale_name})",
                "total_snapshots": total_snapshots,
                "elapsed_sec": elapsed_sec,
                "throughput_snaps_per_sec": total_snapshots / max(1e-6, elapsed_sec),
                "peak_gpu_mb": gpu_peak_mb,
                "peak_cpu_mb": cpu_peak_mb,
                "peak_total_mb": max(gpu_peak_mb, cpu_peak_mb) + min(gpu_peak_mb, cpu_peak_mb) * 0.5,
            }

        snap_veh = np.concatenate(veh_chunks)
        snap_cam = np.concatenate(cam_chunks)
        snap_t = np.concatenate(time_chunks)
        embeddings = torch.cat(emb_chunks, dim=0)

        order = np.argsort(snap_t, kind="mergesort")
        snap_veh = snap_veh[order]
        snap_cam = snap_cam[order]
        snap_t = snap_t[order]
        embeddings = embeddings[torch.from_numpy(order).long()]

        gt_trajs = {v: [] for v in range(n_veh)}
        for sid, vid in enumerate(snap_veh.tolist()):
            gt_trajs[vid].append(sid)

        dataset = {
            "name": "L-GMM",
            "embeddings": embeddings,
            "snap_veh": snap_veh,
            "snap_cam": snap_cam,
            "snap_time": snap_t,
            "snap_true_time": snap_t,
            "is_blackhole": np.zeros(len(snap_veh), dtype=bool),
            "gt_trajectories": gt_trajs,
            "topology": self.topology,
            "spec": self.spec,
            "profile": self.profile,
            "elapsed_sec": elapsed_sec,
            "peak_gpu_mb": gpu_peak_mb,
            "peak_cpu_mb": cpu_peak_mb,
            "total_snapshots": int(len(snap_veh)),
        }
        if generate_workloads:
            dataset["workloads"] = build_multimodal_query_workloads(dataset, rng)
        return dataset
