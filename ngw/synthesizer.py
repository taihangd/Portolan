"""
Category-A Synthesis Baseline 2 — Normalized-Gaussian-Walk (NGW):
Uniform random topological walks (uncoupled link delays without arterial congestion waves)
paired with normalized isotropic Gaussian vectors around uniform vehicle prototypes on S^{d-1},
omitting hard-negative trim twins, camera SO(d) rotations, and Black-Hole glare hubs.
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


class NormalizedGaussianWalkSynthesizer:
    """
    Normalized-Gaussian-Walk (NGW):
    Uniform random topological walks + normalized isotropic Gaussian perturbations around uniform
    vehicle prototypes on S^{d-1} without twin cones, SO(d) drift, or Black-Hole hubs.
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
        """Synthesize uncalibrated random-walk trajectories and isotropic Gaussian embeddings."""
        gc.collect()
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        t_start = time.perf_counter()

        rng = np.random.default_rng(self.spec.seed + 101)
        torch.manual_seed(self.spec.seed + 101)

        n_cam = self.topology.n_cam
        n_veh = self.spec.n_veh
        d = self.spec.embed_dim
        steps_h = max(4, int(round(self.spec.target_snapshots / max(1, n_veh * (1.0 - self.spec.rho_occ)))))

        veh_proto = torch.nn.functional.normalize(torch.randn(n_veh, d, device=self.device), dim=-1)

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
                valid = (nbrs >= 0) & (nbrs != prev_cam[:, None])
                no_choice = ~np.any(valid, axis=1)
                if np.any(no_choice):
                    valid[no_choice] = nbrs[no_choice] >= 0
                logits = np.where(valid, 0.0, -1e9)
                gumbel = -np.log(-np.log(np.clip(rng.random((b_size, self.topology.max_deg)), 1e-12, 1.0 - 1e-12)))
                choice_k = np.argmax(logits + gumbel, axis=1)
                row_idx = np.arange(b_size)
                nxt_cam = nbrs[row_idx, choice_k]
                dt = rng.uniform(6.0, 130.0, size=b_size)
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

            v_t = torch.from_numpy(f_veh).long().to(self.device)
            emb_gpu = torch.nn.functional.normalize(
                veh_proto[v_t] + 0.032 * torch.randn(f_veh.shape[0], d, device=self.device), dim=-1
            )
            total_snapshots += int(f_veh.shape[0])

            if stream_only:
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                del v_t, emb_gpu
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
                "name": f"NGW ({self.spec.scale_name})",
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
            "name": "NGW",
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
