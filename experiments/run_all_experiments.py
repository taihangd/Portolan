#!/usr/bin/env python3
"""
Unified End-to-End Experimental Suite for IEEE ICDE 2027 EAB Track:
"[Experiment, Analysis, and Benchmark] Portolan: Simulation-Free Hyperspherical Stream and
Workload Synthesis for City-Scale Cross-Camera Trajectory Databases"

Executes equal-budget 9-point validation tuning (seed=101, strictly interior optima, 0% GT on
evaluation seeds), strict out-of-sample real-stream calibration/evaluation splits (0% overlap,
zero-GT physical road free-flow travel times, n=1 real stream with B=200 vehicle-block bootstrap SE),
unified GPU-vectorized 3-mode glare + out-of-family rank-4 gamma glare stress tests, fair shared
CSLS+kinematic edge scoring with arrival-trough sharded MCF (`MCF-Sharded`), 100% serial solver
SLA timing up to 5.10M snapshots, and true O(|Delta|) incremental view maintenance.
"""

import copy
import csv
import gc
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent
PKG_ROOT = EXP_DIR.parent
if (PKG_ROOT / "prov_ivm").is_dir():
    if str(PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(PKG_ROOT))
else:
    WORKSPACE_ROOT = PKG_ROOT.parent
    _SIBLING_PKGS = (
        "portolan",
        "sim3d_pipeline",
        "ngw",
        "lgmm",
        "global_bm",
        "mcf_tracker",
        "online_clust",
        "vec_filter",
        "prob_path",
        "prov_ivm",
    )
    for _pkg in reversed(_SIBLING_PKGS):
        _p = str(WORKSPACE_ROOT / _pkg)
        if _p not in sys.path:
            sys.path.insert(0, _p)

from portolan.profiles_and_topology import (
    RoadCameraTopology,
    calibrate_seed_profile_from_stream,
    get_benchmark_spec,
    get_seed_profile,
    load_real_benchmark_stream,
    select_best_device,
)
from portolan.evaluation import (
    compute_fidelity_metrics,
    compute_mtmc_metrics,
    compute_multiconfig_ranking_stats,
    compute_ranking_consistency,
    compute_vehicle_block_bootstrap_se,
    extract_stream_distributions,
)
from portolan.synthesizer import (
    PortolanSynthesizer,
    build_multimodal_query_workloads,
)
from portolan.candidate_graph import SpatioTemporalCandidateGraph
from portolan.query_eval import (
    BaseDBEngine,
    evaluate_q1_from_trajs,
    evaluate_q2_eager_from_trajs,
    evaluate_q3_from_trajs,
)
from sim3d_pipeline import ThreeDSimPipelineProxy
from ngw import NormalizedGaussianWalkSynthesizer
from lgmm import LatentGMMSynthesizer
from global_bm import EagerHungarianEngine
from mcf_tracker import MinCostNetFlowEngine
from online_clust import StreamGreedyClustEngine
from vec_filter import VecDBFilteredANNEngine
from prob_path import TopoMarkovMAPEngine
from prov_ivm import ProvenanceCascadingDBEngine


DEFAULT_TUNED_PARAMS: Dict[str, float] = {
    "Global-BM": 0.30,
    "MCF-Tracker": 0.18,
    "MCF-Sharded": 0.18,
    "Online-Clust": 0.30,
    "Vec-Filter": 0.30,
    "Prob-Path": 0.24,
    "Prov-IVM": 0.18,
}


def get_all_db_engines(tuned_params: Optional[Dict[str, float]] = None) -> List[BaseDBEngine]:
    """Instantiate the 6 core Category-B paradigms with equal-budget tuned thresholds."""
    tp = tuned_params or DEFAULT_TUNED_PARAMS
    return [
        EagerHungarianEngine(sim_threshold=float(tp["Global-BM"])),
        MinCostNetFlowEngine(match_reward=float(tp["MCF-Tracker"]), sharded=False),
        StreamGreedyClustEngine(sim_threshold=float(tp["Online-Clust"])),
        VecDBFilteredANNEngine(sim_threshold=float(tp["Vec-Filter"])),
        TopoMarkovMAPEngine(base_prior=float(tp["Prob-Path"])),
        ProvenanceCascadingDBEngine(base_threshold=float(tp["Prov-IVM"])),
    ]


def get_mcf_sharded_engine(tuned_params: Optional[Dict[str, float]] = None) -> MinCostNetFlowEngine:
    """Instantiate arrival-trough sharded Top-K MCF (`MCF-Sharded`) sharing Prov-IVM's exact sharding."""
    tp = tuned_params or DEFAULT_TUNED_PARAMS
    th = float(tp.get("MCF-Sharded", tp["MCF-Tracker"]))
    return MinCostNetFlowEngine(match_reward=th, sharded=True, top_k=5)


def build_m18_engine_configs(tuned_params: Dict[str, float]) -> List[Tuple[str, BaseDBEngine]]:
    """
    Construct M = 18 paradigm-hyperparameter configurations (6 paradigms x 3 window/threshold settings)
    around each engine's held-out validation operating point for high-power ranking evaluation.
    """
    tp = tuned_params
    bm_t = float(tp["Global-BM"])
    mc_t = float(tp["MCF-Tracker"])
    oc_t = float(tp["Online-Clust"])
    vf_t = float(tp["Vec-Filter"])
    pp_t = float(tp["Prob-Path"])
    pv_t = float(tp["Prov-IVM"])
    return [
        ("Global-BM:low_W350", EagerHungarianEngine(sim_threshold=max(0.04, bm_t - 0.08), window_size=350, overlap=240)),
        ("Global-BM:tuned_W550", EagerHungarianEngine(sim_threshold=bm_t, window_size=550, overlap=380)),
        ("Global-BM:high_W800", EagerHungarianEngine(sim_threshold=bm_t + 0.08, window_size=800, overlap=520)),
        ("MCF-Tracker:low", MinCostNetFlowEngine(match_reward=max(0.04, mc_t - 0.08), sharded=False)),
        ("MCF-Tracker:tuned", MinCostNetFlowEngine(match_reward=mc_t, sharded=False)),
        ("MCF-Tracker:high", MinCostNetFlowEngine(match_reward=mc_t + 0.08, sharded=False)),
        ("Online-Clust:low_110s", StreamGreedyClustEngine(sim_threshold=max(0.08, oc_t - 0.08), window_sec=110.0)),
        ("Online-Clust:tuned_160s", StreamGreedyClustEngine(sim_threshold=oc_t, window_sec=160.0)),
        ("Online-Clust:high_210s", StreamGreedyClustEngine(sim_threshold=min(0.62, oc_t + 0.08), window_sec=210.0)),
        ("Vec-Filter:low", VecDBFilteredANNEngine(sim_threshold=max(0.08, vf_t - 0.08))),
        ("Vec-Filter:tuned", VecDBFilteredANNEngine(sim_threshold=vf_t)),
        ("Vec-Filter:high", VecDBFilteredANNEngine(sim_threshold=vf_t + 0.08)),
        ("Prob-Path:low", TopoMarkovMAPEngine(base_prior=max(0.06, pp_t - 0.08))),
        ("Prob-Path:tuned", TopoMarkovMAPEngine(base_prior=pp_t)),
        ("Prob-Path:high", TopoMarkovMAPEngine(base_prior=pp_t + 0.08)),
        ("Prov-IVM:low_K3", ProvenanceCascadingDBEngine(base_threshold=max(0.06, pv_t - 0.08), top_k=3)),
        ("Prov-IVM:tuned_K5", ProvenanceCascadingDBEngine(base_threshold=pv_t, top_k=5)),
        ("Prov-IVM:high_K8", ProvenanceCascadingDBEngine(base_threshold=pv_t + 0.08, top_k=8)),
    ]


RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def _neighborhood_agreement_pct(
    next_ivm: np.ndarray,
    next_exact: np.ndarray,
    neighborhood_nodes: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
) -> float:
    """Compute link agreement strictly over the affected candidate neighborhood `N_cand(Delta)`."""
    if neighborhood_nodes is None or len(neighborhood_nodes) == 0:
        return 100.0
    nb = neighborhood_nodes
    if active_mask is not None:
        nb = nb[active_mask[nb]]
    if len(nb) == 0:
        return 100.0
    return float(100.0 * np.mean(next_ivm[nb] == next_exact[nb]))


def tune_engines_equal_budget(device: torch.device, val_seed: int = 101) -> Dict:
    """
    Equal-Budget 9-Point Validation Tuning Protocol:
    Tunes the primary acceptance threshold of every Category-B paradigm (including `MCF-Sharded`)
    over a wide 9-point grid `[0.06, 0.58]` maximizing `0.5 * IDF1 + 0.5 * MOTA_link` strictly on
    a held-out synthetic validation cohort (`val_seed=101`), verifying every optimum lies strictly
    in the interior of the grid (`0 < best_idx < 8`). Zero ground-truth labels are ever accessed
    on evaluation seeds `{42, 43, 44}` or held-out real evaluation streams.
    """
    print("\n================ [Validation Tuning] Wide 9-Point Interior Grid on Held-Out Seed 101 ================")
    wide_grid = [0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.50, 0.58]
    tuning_grids = {
        "Global-BM": ("sim_threshold", wide_grid, lambda v: EagerHungarianEngine(sim_threshold=v)),
        "MCF-Tracker": ("match_reward", wide_grid, lambda v: MinCostNetFlowEngine(match_reward=v, sharded=False)),
        "MCF-Sharded": ("match_reward", wide_grid, lambda v: MinCostNetFlowEngine(match_reward=v, sharded=True)),
        "Online-Clust": ("sim_threshold", wide_grid, lambda v: StreamGreedyClustEngine(sim_threshold=v)),
        "Vec-Filter": ("sim_threshold", wide_grid, lambda v: VecDBFilteredANNEngine(sim_threshold=v)),
        "Prob-Path": ("base_prior", wide_grid, lambda v: TopoMarkovMAPEngine(base_prior=v)),
        "Prov-IVM": ("base_threshold", wide_grid, lambda v: ProvenanceCascadingDBEngine(base_threshold=v)),
    }

    out = {
        "validation_seed": val_seed,
        "zero_gt_on_evaluation_seeds": True,
        "grid_points_per_engine": len(wide_grid),
        "grid_values": wide_grid,
        "objective": "0.5*IDF1 + 0.5*MOTA_link",
        "all_optima_strictly_interior": True,
        "profiles": {},
    }
    for prof_name in ["CityFlow-Seed", "VeRi-Seed"]:
        cal_ds = load_real_benchmark_stream(prof_name, fold_idx=1, split="cal")
        profile, _ = calibrate_seed_profile_from_stream(cal_ds, name=prof_name)

        val_spec = get_benchmark_spec("Portolan-Tiny", seed=val_seed)
        val_spec.n_veh = 800
        val_spec.target_snapshots = 8000
        val_spec.horizon_sec = 1200.0
        val_topo = RoadCameraTopology(val_spec.n_cam, seed=val_seed)
        val_ds = PortolanSynthesizer(profile, val_spec, val_topo, device).synthesize(generate_workloads=False)
        val_cg = SpatioTemporalCandidateGraph(val_ds, device)

        selected = {}
        grid_details = {}
        prof_interior = True
        for eng_name, (param_name, grid_vals, builder) in tuning_grids.items():
            best_idx = 0
            best_v = grid_vals[0]
            best_obj = -1e9
            best_idf1 = -1.0
            scores = {}
            for idx_g, gv in enumerate(grid_vals):
                eng = builder(gv)
                tr, _, _ = eng.recover_trajectories(val_cg)
                mt_v = compute_mtmc_metrics(tr, val_ds["snap_veh"])
                idf1_v = float(mt_v["idf1"])
                obj_v = 0.5 * idf1_v + 0.5 * float(mt_v["mota_link"])
                scores[f"{gv:.2f}"] = {
                    "idf1": round(idf1_v, 4),
                    "mota_link": round(float(mt_v["mota_link"]), 4),
                    "obj": round(obj_v, 4),
                }
                if obj_v > best_obj:
                    best_obj = obj_v
                    best_idf1 = idf1_v
                    best_v = gv
                    best_idx = idx_g
            is_interior = bool(0 < best_idx < (len(grid_vals) - 1))
            prof_interior = prof_interior and is_interior
            out["all_optima_strictly_interior"] = bool(out["all_optima_strictly_interior"] and is_interior)
            selected[eng_name] = float(best_v)
            grid_details[eng_name] = {
                "param_name": param_name,
                "grid_scores": scores,
                "best_param": float(best_v),
                "best_grid_index": int(best_idx),
                "is_strictly_interior": is_interior,
                "best_val_idf1": round(best_idf1, 4),
                "best_val_obj": round(best_obj, 4),
            }
        out["profiles"][prof_name] = {
            "all_strictly_interior": prof_interior,
            "selected_thresholds": selected,
            "details": grid_details,
        }
        print(
            f"[Tuning {prof_name}] Selected thresholds (seed={val_seed}, all_interior={prof_interior}): {selected}"
        )
    return out


def run_exp1_scalability(device: torch.device) -> Dict:
    """Exp 1: Synthesis Throughput, Memory Scalability (25K to 5.1M) & End-to-End Workload Compilation."""
    print("\n================ [Exp 1] Synthesis Throughput & Memory Scalability (Unified 3-Mode Glare) ================")
    profile = get_seed_profile("CityFlow-Seed")
    sim_proxy = ThreeDSimPipelineProxy(device=device, embed_dim=256)
    dev_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    n_prof_snaps = 64 if device.type == "cuda" else 4
    n_prof_frames = 10 if device.type == "cuda" else 2
    hw_prof = sim_proxy.profile_hardware_throughput(
        n_sample_snapshots=n_prof_snaps, frames_per_snapshot=n_prof_frames
    )
    print(
        f"Profiled {dev_label} 3D-Sim micro-benchmark + analytical extrapolation: "
        f"{hw_prof['snaps_per_sec']:.2f} snaps/s (active compute VRAM={hw_prof['active_pipeline_vram_mb']:.1f} MB)"
    )

    scales = [
        "Portolan-Tiny",
        "Portolan-Small",
        "Portolan-Medium",
        "Portolan-CityScale",
    ]
    out_scales = {}
    csv_rows = []

    # Symmetric CUDA kernel & cuBLAS warmup across all four GPU synthesizers
    warm_spec = get_benchmark_spec("Portolan-Tiny", seed=99, embed_dim=256)
    warm_spec_512 = get_benchmark_spec("Portolan-Tiny", seed=99, embed_dim=512)
    warm_topo = RoadCameraTopology(warm_spec.n_cam, seed=99)
    _ = NormalizedGaussianWalkSynthesizer(profile, warm_spec, warm_topo, device).synthesize(
        stream_only=True, generate_workloads=False
    )
    _ = LatentGMMSynthesizer(profile, warm_spec, warm_topo, device).synthesize(
        stream_only=True, generate_workloads=False
    )
    _ = PortolanSynthesizer(profile, warm_spec, warm_topo, device).synthesize(
        stream_only=True, generate_workloads=False
    )
    _ = PortolanSynthesizer(profile, warm_spec_512, warm_topo, device).synthesize(
        stream_only=True, generate_workloads=False
    )

    seeds_exp1 = [42, 43, 44]

    def _bench_3seed(cls_syn, sc_key: str, topo_obj: RoadCameraTopology, dim: int) -> Dict[str, float]:
        times, g_mems, c_mems, snaps = [], [], [], []
        for r_seed in seeds_exp1:
            sp_r = get_benchmark_spec(sc_key, seed=r_seed, embed_dim=dim)
            res_r = cls_syn(profile, sp_r, topo_obj, device).synthesize(
                stream_only=True, generate_workloads=False
            )
            times.append(float(res_r["elapsed_sec"]))
            g_mems.append(float(res_r["peak_gpu_mb"]))
            c_mems.append(float(res_r["peak_cpu_mb"]))
            snaps.append(int(res_r["total_snapshots"]))
        sec_mean = float(np.mean(times))
        sec_std = float(np.std(times))
        snaps_mean = int(round(float(np.mean(snaps))))
        return {
            "actual_snapshots": snaps_mean,
            "elapsed_sec": sec_mean,
            "elapsed_sec_std": sec_std,
            "throughput_snaps_per_sec": float(snaps_mean / max(1e-6, sec_mean)),
            "peak_mem_mb": float(np.mean(g_mems)),
            "peak_cpu_mb": float(np.mean(c_mems)),
        }

    for sc_name in scales:
        spec = get_benchmark_spec(sc_name, seed=42, embed_dim=256)
        topo = RoadCameraTopology(spec.n_cam, seed=42)

        ngw_stats = _bench_3seed(NormalizedGaussianWalkSynthesizer, sc_name, topo, 256)
        lgmm_stats = _bench_3seed(LatentGMMSynthesizer, sc_name, topo, 256)
        tb_stats = _bench_3seed(PortolanSynthesizer, sc_name, topo, 256)
        tb512_stats = _bench_3seed(PortolanSynthesizer, sc_name, topo, 512)

        tb_snaps = int(tb_stats["actual_snapshots"])
        res_3d = sim_proxy.estimate_scale(spec, actual_snapshots=tb_snaps)
        t_3d = float(res_3d["elapsed_sec"])

        for st in (ngw_stats, lgmm_stats, tb_stats, tb512_stats):
            st["speedup_vs_3d_sim"] = float(t_3d / max(1e-6, st["elapsed_sec"]))

        d512_overhead_pct = 100.0 * (tb512_stats["elapsed_sec"] - tb_stats["elapsed_sec"]) / max(1e-6, tb_stats["elapsed_sec"])
        tb512_stats["overhead_vs_d256_pct"] = float(d512_overhead_pct)

        out_scales[sc_name] = {
            "n_cam": spec.n_cam,
            "n_veh": spec.n_veh,
            "target_snapshots": spec.target_snapshots,
            "actual_snapshots": tb_snaps,
            "unified_3mode_glare_kernel": True,
            "3D-Sim-Pipeline": {
                "is_projected_cost_model": True,
                "estimation_mode": "profiled_microbenchmark_plus_analytical_extrapolation",
                "actual_snapshots": int(res_3d["total_snapshots"]),
                "elapsed_sec": t_3d,
                "throughput_snaps_per_sec": float(res_3d["throughput_snaps_per_sec"]),
                "peak_mem_mb": float(res_3d["peak_gpu_mb"]),
                "exceeds_24gb_vram": bool(res_3d["exceeds_24gb_vram"]),
                "speedup_vs_3d_sim": 1.0,
            },
            "NGW": ngw_stats,
            "L-GMM": lgmm_stats,
            "Portolan": tb_stats,
            "Portolan-d512": tb512_stats,
        }

        print(
            f"{sc_name:20s} ({tb_snaps:7d} snaps): "
            f"3D-Sim(Analytical)={t_3d:8.1f}s ({res_3d['peak_total_mb']/1024:.1f}GB) | "
            f"L-GMM={lgmm_stats['elapsed_sec']:5.3f}s | NGW={ngw_stats['elapsed_sec']:5.3f}s | "
            f"Portolan={tb_stats['elapsed_sec']:5.3f}±{tb_stats['elapsed_sec_std']:.3f}s "
            f"({tb_stats['peak_mem_mb']:.1f}MB, d512={d512_overhead_pct:+.1f}%, {tb_stats['speedup_vs_3d_sim']:.0f}x vs Analytical 3D-Sim)"
        )

        for m_name, m_dict in [
            ("3D-Sim-Pipeline (Analytical Est.)", out_scales[sc_name]["3D-Sim-Pipeline"]),
            ("NGW", out_scales[sc_name]["NGW"]),
            ("L-GMM", out_scales[sc_name]["L-GMM"]),
            ("Portolan (d=256)", out_scales[sc_name]["Portolan"]),
            ("Portolan (d=512)", out_scales[sc_name]["Portolan-d512"]),
        ]:
            csv_rows.append({
                "scale": sc_name,
                "n_cam": spec.n_cam,
                "n_veh": spec.n_veh,
                "snapshots": int(m_dict["actual_snapshots"]),
                "synthesizer": m_name,
                "elapsed_sec": round(m_dict["elapsed_sec"], 4),
                "throughput_snaps_per_sec": round(m_dict["throughput_snaps_per_sec"], 1),
                "peak_mem_mb": round(m_dict["peak_mem_mb"], 1),
                "speedup_vs_3d_sim": round(m_dict["speedup_vs_3d_sim"], 1),
            })

    # Explicitly benchmark BOTH (a) stream + relational Q1-Q4 compilation AND (b) full 5-stage S0->S4 waterfall + Q1-Q4
    e2e_compilation = {}
    for sc_e2e in ["Portolan-Tiny", "Portolan-Small"]:
        sp_e = get_benchmark_spec(sc_e2e, seed=42, embed_dim=256)
        tp_e = RoadCameraTopology(sp_e.n_cam, seed=42)
        t_e0 = time.perf_counter()
        res_e = PortolanSynthesizer(profile, sp_e, tp_e, device).synthesize(
            stream_only=False, generate_workloads=True
        )
        tot_e2e = time.perf_counter() - t_e0

        t_wf0 = time.perf_counter()
        _ = PortolanSynthesizer(profile, sp_e, tp_e, device).synthesize_oracle_waterfall(
            generate_workloads=True
        )
        tot_wf = time.perf_counter() - t_wf0

        e2e_compilation[sc_e2e] = {
            "total_snapshots": int(res_e["total_snapshots"]),
            "e2e_elapsed_sec": float(tot_e2e),
            "e2e_with_s0_s4_waterfall_sec": float(tot_wf),
            "e2e_throughput_snaps_per_sec": float(res_e["total_snapshots"] / max(1e-6, tot_e2e)),
            "peak_gpu_mb": float(res_e["peak_gpu_mb"]),
            "peak_cpu_mb": float(res_e["peak_cpu_mb"]),
        }
        print(
            f"[Exp1 E2E] {sc_e2e:18s}: {res_e['total_snapshots']} snaps | "
            f"Stream+Q1..Q4={tot_e2e:.3f}s | S0->S4 Waterfall+Q1..Q4={tot_wf:.3f}s"
        )

    with open(RESULTS_DIR / "table1_scalability.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    return {
        "unified_3mode_glare_across_all_scales": True,
        "seeds_evaluated": seeds_exp1,
        "hardware_3d_profile": hw_prof,
        "scales": out_scales,
        "e2e_workload_compilation": e2e_compilation,
    }


def run_exp2_and_exp3(device: torch.device, tuning_info: Dict) -> Dict:
    """
    Exp 2: Out-of-Sample Statistical & Manifold Fidelity Alignment against Held-Out Real-CityFlow & Real-VeRi
    Exp 3a: Benchmark Ranking Consistency across 6 Tuned Paradigms + M=18 Config Grid + Exact Permutation p-Values
    """
    print("\n================ [Exp 2 & Exp 3a] Out-of-Sample Real Fidelity & Ranking (0% Overlap) ================")
    seeds = [42, 43, 44]
    profiles = ["CityFlow-Seed", "VeRi-Seed"]
    engine_names = [e.name for e in get_all_db_engines()]
    metric_names = [
        "idf1",
        "mota",
        "mota_link",
        "precision",
        "recall",
        "latency_ms",
        "q1_f1",
        "q2_f1",
        "q3_score",
    ]

    exp2_out = {}
    exp3_out = {}
    fid_csv_rows = []
    rank_csv_rows = []
    plot_curves_exp2 = {}

    for prof_name in profiles:
        tp_prof = tuning_info["profiles"][prof_name]["selected_thresholds"]
        real_label = "Real-CityFlow" if "CityFlow" in prof_name else "Real-VeRi"
        q2_hops_real = 4 if "CityFlow" in prof_name else 3

        # 1. Fit SeedProfile STRICTLY on disjoint Calibration Split (`split="cal"`, 0% overlap with `eval`)
        cal_ds = load_real_benchmark_stream(prof_name, fold_idx=1, split="cal")
        cal_profile, _ = calibrate_seed_profile_from_stream(cal_ds, name=prof_name)

        # 2. Load Held-Out Evaluation Split ONCE (`n=1` canonical mid-tracklet stream `fold_idx=1`)
        real_ds = load_real_benchmark_stream(
            prof_name,
            fold_idx=1,
            split="eval",
            q2_min_hops=q2_hops_real,
            q2_time_window_sec=25.0,
            seed_profile_override=cal_profile,
        )
        overlap_vehs = len(set(cal_ds["raw_veh_ids"].tolist()) & set(real_ds["raw_veh_ids"].tolist()))
        assert overlap_vehs == 0, f"Detected vehicle/trip overlap ({overlap_vehs}) between cal and eval!"

        real_ds["workloads"] = build_multimodal_query_workloads(real_ds, np.random.default_rng(542))
        d_real = extract_stream_distributions(real_ds, device)
        real_zipf_val = float(d_real["zipf_gamma_est"][0])

        # Evaluate n=1 Held-Out RealStream ONCE + compute B=200 vehicle-block bootstrap standard errors
        cg_real = SpatioTemporalCandidateGraph(real_ds, device)
        wl_real = real_ds["workloads"]
        k_q2_r = wl_real["Q2"]["default_k"]
        gt_q2_r = wl_real["Q2"]["gt_pairs_by_k"][k_q2_r]

        real_scores_single: Dict[str, Dict[str, float]] = {}
        real_boot_se: Dict[str, Dict[str, float]] = {}
        for eng in get_all_db_engines(tp_prof):
            trajs_r, _, lat_r = eng.recover_trajectories(cg_real)
            mt_r = compute_mtmc_metrics(trajs_r, real_ds["snap_veh"])
            mt_r["latency_ms"] = lat_r * 1000.0
            mt_r["q1_f1"] = evaluate_q1_from_trajs(trajs_r, cg_real.n_snaps, wl_real["Q1"])
            if isinstance(eng, ProvenanceCascadingDBEngine):
                q2_r, _ = eng.execute_coupled_q2_join(cg_real, k_q2_r, wl_real["Q2"]["delta_t_sec"], gt_q2_r)
            else:
                q2_r = evaluate_q2_eager_from_trajs(
                    trajs_r, real_ds["snap_cam"], real_ds["snap_time"], real_ds["snap_veh"],
                    k_q2_r, wl_real["Q2"]["delta_t_sec"], gt_q2_r,
                )
            mt_r["q2_f1"] = q2_r["f1"]
            mt_r["q3_score"] = evaluate_q3_from_trajs(trajs_r, real_ds["snap_cam"], wl_real["Q3"])["q3_score"]
            real_scores_single[eng.name] = {mk: float(mt_r[mk]) for mk in metric_names}
            real_boot_se[eng.name] = compute_vehicle_block_bootstrap_se(
                trajs_r,
                real_ds["snap_veh"],
                snap_cam=real_ds["snap_cam"],
                workloads=wl_real,
                n_boot=200,
                seed=42,
            )

        # Also evaluate MCF-Sharded on RealStream for direct head-to-head diagnostic transparency
        mcf_sh_real_tr, _, mcf_sh_real_lat = get_mcf_sharded_engine(tp_prof).recover_trajectories(cg_real)
        mcf_sh_real_mt = compute_mtmc_metrics(mcf_sh_real_tr, real_ds["snap_veh"])

        # Evaluate M=18 multi-configuration vector on Held-Out RealStream
        m18_cfgs = build_m18_engine_configs(tp_prof)
        m18_names = [nm for nm, _ in m18_cfgs]
        m18_real_idf1 = []
        for _, eng_cfg in m18_cfgs:
            tr_c, _, _ = eng_cfg.recover_trajectories(cg_real)
            m18_real_idf1.append(float(compute_mtmc_metrics(tr_c, real_ds["snap_veh"])["idf1"]))
        m18_real_vec = np.array(m18_real_idf1, dtype=np.float64)

        # 3. Synthesize & evaluate Portolan, L-GMM, and NGW across 3 genuinely independent random seeds {42, 43, 44}
        syn_fid_acc = {s: [] for s in ["Portolan", "L-GMM", "NGW"]}
        syn_eng_acc = {
            s: {en: {m: [] for m in metric_names} for en in engine_names}
            for s in ["Portolan", "L-GMM", "NGW"]
        }
        mcf_sh_syn_idf1 = {s: [] for s in ["Portolan", "L-GMM", "NGW"]}
        m18_syn_acc = {s: [] for s in ["Portolan", "L-GMM", "NGW"]}
        rank_per_seed_acc = {s: [] for s in ["Portolan", "L-GMM", "NGW"]}
        cgraph_ms_acc = {"RealStream": [float(cg_real.build_time_sec * 1000.0)], "Portolan": [], "L-GMM": [], "NGW": []}
        snaps_acc = {"RealStream": [int(cg_real.n_snaps)], "Portolan": [], "L-GMM": [], "NGW": []}

        topo = real_ds["topology"]
        for seed_idx, sd in enumerate(seeds):
            spec = copy.deepcopy(real_ds["spec"])
            spec.seed = sd

            tb_ds = PortolanSynthesizer(cal_profile, spec, topo, device).synthesize(generate_workloads=True)
            lgmm_ds = LatentGMMSynthesizer(cal_profile, spec, topo, device).synthesize(generate_workloads=True)
            ngw_ds = NormalizedGaussianWalkSynthesizer(cal_profile, spec, topo, device).synthesize(generate_workloads=True)

            syn_map = {"Portolan": tb_ds, "L-GMM": lgmm_ds, "NGW": ngw_ds}
            dists_this_seed = {"RealStream": d_real}

            for syn_k, ds_syn in syn_map.items():
                d_syn = extract_stream_distributions(ds_syn, device)
                dists_this_seed[syn_k] = d_syn
                syn_fid_acc[syn_k].append(compute_fidelity_metrics(d_real, d_syn))

                cg_s = SpatioTemporalCandidateGraph(ds_syn, device)
                cgraph_ms_acc[syn_k].append(float(cg_s.build_time_sec * 1000.0))
                snaps_acc[syn_k].append(int(cg_s.n_snaps))
                wl_s = ds_syn["workloads"]
                k_q2_s = wl_s["Q2"]["default_k"]
                gt_q2_s = wl_s["Q2"]["gt_pairs_by_k"][k_q2_s]

                this_seed_scores: Dict[str, Dict[str, float]] = {}
                for eng in get_all_db_engines(tp_prof):
                    trajs_s, _, lat_s = eng.recover_trajectories(cg_s)
                    mt_s = compute_mtmc_metrics(trajs_s, ds_syn["snap_veh"])
                    mt_s["latency_ms"] = lat_s * 1000.0
                    mt_s["q1_f1"] = evaluate_q1_from_trajs(trajs_s, cg_s.n_snaps, wl_s["Q1"])
                    if isinstance(eng, ProvenanceCascadingDBEngine):
                        q2_s, _ = eng.execute_coupled_q2_join(cg_s, k_q2_s, wl_s["Q2"]["delta_t_sec"], gt_q2_s)
                    else:
                        q2_s = evaluate_q2_eager_from_trajs(
                            trajs_s, ds_syn["snap_cam"], ds_syn["snap_time"], ds_syn["snap_veh"],
                            k_q2_s, wl_s["Q2"]["delta_t_sec"], gt_q2_s,
                        )
                    mt_s["q2_f1"] = q2_s["f1"]
                    mt_s["q3_score"] = evaluate_q3_from_trajs(trajs_s, ds_syn["snap_cam"], wl_s["Q3"])["q3_score"]
                    this_seed_scores[eng.name] = {mk: float(mt_s[mk]) for mk in metric_names}
                    for mk in metric_names:
                        syn_eng_acc[syn_k][eng.name][mk].append(float(mt_s[mk]))

                tr_msh, _, _ = get_mcf_sharded_engine(tp_prof).recover_trajectories(cg_s)
                mcf_sh_syn_idf1[syn_k].append(float(compute_mtmc_metrics(tr_msh, ds_syn["snap_veh"])["idf1"]))

                rank_per_seed_acc[syn_k].append(
                    compute_ranking_consistency(real_scores_single, this_seed_scores, engine_names)
                )

                # Evaluate M=18 config vector on this synthetic seed
                vec_18 = []
                for _, eng_cfg in build_m18_engine_configs(tp_prof):
                    tr_c, _, _ = eng_cfg.recover_trajectories(cg_s)
                    vec_18.append(float(compute_mtmc_metrics(tr_c, ds_syn["snap_veh"])["idf1"]))
                m18_syn_acc[syn_k].append(vec_18)

            # Save seed=43 (index 1) curves of Held-Out Real-CityFlow & Synthesizers for Figure 4 plotting
            if prof_name == "CityFlow-Seed" and seed_idx == 1:
                for k_ds, d_obj in dists_this_seed.items():
                    m_bins = np.linspace(-0.55, 0.75, 49)
                    m_hist, _ = np.histogram(d_obj["cosine_margin"], bins=m_bins, density=True)

                    hub_vals = d_obj["rev_knn_hubness"]
                    x_hub = np.unique(np.concatenate([np.arange(1, 30), np.geomspace(30, 320, 28).astype(int)]))
                    ccdf_hub = [float(np.mean(hub_vals >= xv)) for xv in x_hub]

                    t_bins = np.linspace(0.4, 3.8, 45)
                    t_hist, _ = np.histogram(d_obj["travel_time_ratio"], bins=t_bins, density=True)

                    plot_curves_exp2[k_ds] = {
                        "margin_bin_centers": ((m_bins[:-1] + m_bins[1:]) * 0.5).tolist(),
                        "margin_density": m_hist.tolist(),
                        "hub_x": x_hub.tolist(),
                        "hub_ccdf": ccdf_hub,
                        "tt_bin_centers": ((t_bins[:-1] + t_bins[1:]) * 0.5).tolist(),
                        "tt_density": t_hist.tolist(),
                        "zipf_gamma": float(d_obj["zipf_gamma_est"][0]),
                    }

        exp2_out[prof_name] = {
            "real_stream_name": real_label,
            "split_protocol": {
                "calibration_split": "cam < 9 (S01+S02)" if "CityFlow" in prof_name else "veh_id % 2 == 0",
                "evaluation_split": "cam >= 9 (S03+S04+S05, re-indexed 0..30)" if "CityFlow" in prof_name else "veh_id % 2 == 1",
                "cal_snapshots": int(cal_ds["total_snapshots"]),
                "cal_vehicles": int(cal_ds["spec"].n_veh),
                "eval_snapshots": int(real_ds["total_snapshots"]),
                "eval_vehicles": int(real_ds["spec"].n_veh),
                "vehicle_overlap": int(overlap_vehs),
                "n_real_streams": 1,
                "n_synthetic_seeds": len(seeds),
            },
            "real_snapshots": int(real_ds["total_snapshots"]),
            "real_cameras": int(real_ds["spec"].n_cam),
            "real_vehicles": int(real_ds["spec"].n_veh),
            "real_zipf_gamma": {"mean": real_zipf_val, "std": 0.0},
            "synthesizers": {},
        }
        for syn_k, flist in syn_fid_acc.items():
            agg = {
                "cosine_margin_W1": _mean_std([x["cosine_margin"]["W1"] for x in flist]),
                "cosine_margin_JSD": _mean_std([x["cosine_margin"]["JSD"] for x in flist]),
                "cosine_margin_MMD": _mean_std([x["cosine_margin"]["MMD"] for x in flist]),
                "hubness_W1": _mean_std([x["hubness_tail"]["W1"] for x in flist]),
                "hubness_JSD": _mean_std([x["hubness_tail"]["JSD"] for x in flist]),
                "hubness_MMD": _mean_std([x["hubness_tail"]["MMD"] for x in flist]),
                "zipf_gamma": _mean_std([x["hubness_tail"]["zipf_gamma"] for x in flist]),
                "max_rev_knn": _mean_std([x["hubness_tail"]["max_rev_knn"] for x in flist]),
                "travel_time_W1": _mean_std([x["travel_time"]["W1"] for x in flist]),
                "travel_time_JSD": _mean_std([x["travel_time"]["JSD"] for x in flist]),
                "travel_time_MMD": _mean_std([x["travel_time"]["MMD"] for x in flist]),
                "mean_W1": _mean_std([x["mean_W1"] for x in flist]),
                "mean_JSD": _mean_std([x["mean_JSD"] for x in flist]),
                "mean_MMD": _mean_std([x["mean_MMD"] for x in flist]),
            }
            exp2_out[prof_name]["synthesizers"][syn_k] = agg
            print(
                f"[Exp2 {real_label} Held-Out] {syn_k:10s}: "
                f"mean_W1={agg['mean_W1']['mean']:.4f}±{agg['mean_W1']['std']:.4f}, "
                f"mean_JSD={agg['mean_JSD']['mean']:.4f}, mean_MMD={agg['mean_MMD']['mean']:.4f}, "
                f"Zipf={agg['zipf_gamma']['mean']:.2f}"
            )
            fid_csv_rows.append({
                "profile": prof_name,
                "real_stream": real_label,
                "synthesizer": syn_k,
                "margin_W1": round(agg["cosine_margin_W1"]["mean"], 4),
                "margin_JSD": round(agg["cosine_margin_JSD"]["mean"], 4),
                "margin_MMD": round(agg["cosine_margin_MMD"]["mean"], 4),
                "hubness_W1": round(agg["hubness_W1"]["mean"], 4),
                "hubness_JSD": round(agg["hubness_JSD"]["mean"], 4),
                "hubness_MMD": round(agg["hubness_MMD"]["mean"], 4),
                "zipf_gamma": round(agg["zipf_gamma"]["mean"], 2),
                "travel_time_W1": round(agg["travel_time_W1"]["mean"], 4),
                "travel_time_JSD": round(agg["travel_time_JSD"]["mean"], 4),
                "travel_time_MMD": round(agg["travel_time_MMD"]["mean"], 4),
                "mean_W1": round(agg["mean_W1"]["mean"], 4),
                "mean_JSD": round(agg["mean_JSD"]["mean"], 4),
                "mean_MMD": round(agg["mean_MMD"]["mean"], 4),
            })

        mean_eng_scores: Dict[str, Dict[str, Dict[str, float]]] = {"RealStream": real_scores_single}
        mean_eng_stds: Dict[str, Dict[str, Dict[str, float]]] = {"RealStream": real_boot_se}
        for syn_k, edict in syn_eng_acc.items():
            mean_eng_scores[syn_k] = {
                en: {mk: float(np.mean(vals)) for mk, vals in mdict.items()}
                for en, mdict in edict.items()
            }
            mean_eng_stds[syn_k] = {
                en: {mk: float(np.std(vals, ddof=0)) for mk, vals in mdict.items()}
                for en, mdict in edict.items()
            }

        rank_summary = {}
        m18_summary = {}
        for syn_k in ["Portolan", "L-GMM", "NGW"]:
            rc_mean_scores = compute_ranking_consistency(
                mean_eng_scores["RealStream"], mean_eng_scores[syn_k], engine_names
            )
            rc_mean_scores["per_seed_idf1_spearman_rho_mean"] = float(
                np.mean([r["idf1_spearman_rho"] for r in rank_per_seed_acc[syn_k]])
            )
            rc_mean_scores["per_seed_idf1_spearman_rho_std"] = float(
                np.std([r["idf1_spearman_rho"] for r in rank_per_seed_acc[syn_k]])
            )
            rc_mean_scores["per_seed_mean_rho_mean"] = float(
                np.mean([r["mean_spearman_rho"] for r in rank_per_seed_acc[syn_k]])
            )
            rc_mean_scores["per_seed_mean_rho_std"] = float(
                np.std([r["mean_spearman_rho"] for r in rank_per_seed_acc[syn_k]])
            )

            syn_18_mean = np.mean(np.array(m18_syn_acc[syn_k], dtype=np.float64), axis=0)
            m18_stats = compute_multiconfig_ranking_stats(m18_real_vec, syn_18_mean, n_perm=100000, seed=42)
            m18_stats["real_idf1_vector"] = [round(float(v), 4) for v in m18_real_vec]
            m18_stats["syn_idf1_vector"] = [round(float(v), 4) for v in syn_18_mean]
            m18_summary[syn_k] = m18_stats
            rc_mean_scores["m18_idf1_spearman_rho"] = m18_stats["spearman_rho"]
            rc_mean_scores["m18_idf1_pvalue_perm"] = m18_stats["spearman_pvalue_perm"]
            rc_mean_scores["m18_idf1_kendall_tau_b"] = m18_stats["kendall_tau_b"]
            rc_mean_scores["m18_idf1_mae"] = m18_stats["mae"]
            rc_mean_scores["m18_idf1_lin_ccc"] = m18_stats["lin_ccc"]

            rank_summary[syn_k] = rc_mean_scores
            print(
                f"[Exp3a {real_label}] [{syn_k:10s}]: "
                f"n=6 IDF1_rho={rc_mean_scores['idf1_spearman_rho']:.3f} (exact p={rc_mean_scores['idf1_spearman_pvalue_exact']:.4f}), "
                f"M=18 rho={m18_stats['spearman_rho']:.3f} (tau_b={m18_stats['kendall_tau_b']:.3f}, perm p={m18_stats['spearman_pvalue_perm']:.5f}), "
                f"MAE={rc_mean_scores['idf1_mae']:.3f}, CCC={rc_mean_scores['idf1_lin_ccc']:.3f}"
            )

        exp3_out[prof_name] = {
            "real_stream_name": real_label,
            "uncertainty_protocol": {
                "real_stream": "n=1 held-out evaluation split (fold_idx=1, 0% cal overlap) with B=200 vehicle-block bootstrap SE",
                "synthetic_streams": "3 independent random seeds {42, 43, 44} standard deviation",
            },
            "stream_mean_snapshots": {k: float(np.mean(v)) for k, v in snaps_acc.items()},
            "cgraph_build_ms": {
                k: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                }
                for k, v in cgraph_ms_acc.items()
            },
            "engine_order": engine_names,
            "engine_means": mean_eng_scores,
            "engine_stds": mean_eng_stds,
            "mcf_sharded_diagnostic": {
                "RealStream_idf1": float(mcf_sh_real_mt["idf1"]),
                "RealStream_latency_ms": float(mcf_sh_real_lat * 1000.0),
                "Portolan_idf1": _mean_std(mcf_sh_syn_idf1["Portolan"]),
            },
            "multiconfig_m18_names": m18_names,
            "multiconfig_m18_ranking": m18_summary,
            "ranking_consistency": rank_summary,
        }

        for ds_k in ["RealStream", "Portolan", "L-GMM", "NGW"]:
            for en in engine_names:
                ms = mean_eng_scores[ds_k][en]
                se = mean_eng_stds[ds_k][en]
                rank_csv_rows.append({
                    "profile": prof_name,
                    "stream": real_label if ds_k == "RealStream" else ds_k,
                    "engine": en,
                    "idf1": round(ms["idf1"], 4),
                    "idf1_se_or_sd": round(se["idf1"], 4),
                    "mota_link": round(ms["mota_link"], 4),
                    "precision": round(ms["precision"], 4),
                    "recall": round(ms["recall"], 4),
                    "latency_ms": round(ms["latency_ms"], 2),
                    "q1_f1": round(ms["q1_f1"], 4),
                    "q2_f1": round(ms["q2_f1"], 4),
                    "q3_score": round(ms["q3_score"], 4),
                })

    with open(RESULTS_DIR / "table2_fidelity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fid_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fid_csv_rows)

    with open(RESULTS_DIR / "exp3_ranking_consistency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rank_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rank_csv_rows)

    with open(RESULTS_DIR / "plot_data_exp2.json", "w") as f:
        json.dump(plot_curves_exp2, f, indent=2)

    return {"exp2_fidelity": exp2_out, "exp3_ranking": exp3_out}


def run_exp4_stress_testing(device: torch.device, tuned_params: Dict[str, float]) -> Dict:
    """
    Exp 4: Stress-Testing Pathological Failure Modes:
      4a: Embedding Black-Hole Cascading Avalanche (`tanh_3mode` primary + out-of-family `gamma_rank4` robustness)
      4b: Zero-Image Co-Movement Join Recall Collapse (`k in [2, 8]` at `rho_occ=0.20`, including `MCF-Sharded`)
      4c: Edge-Cloud Privacy Error Compensation (`sigma_dp` across hops `k in [1, 6]`, 100% blind noise estimator)
      4d: 4-Step Component Ablation of `Prov-IVM` starting from `MCF-Sharded`
    """
    print("\n================ [Exp 4] Stress-Testing Pathological Failure Modes + Out-of-Family Glare ================")
    profile = get_seed_profile("CityFlow-Seed")
    base_spec = get_benchmark_spec("Portolan-Tiny", seed=42)
    topo = RoadCameraTopology(base_spec.n_cam, seed=42)

    # --- 4a: Black-Hole Cascading Avalanche (In-Family `tanh_3mode` + Out-of-Family `gamma_rank4`) ---
    alphas = [0.00, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    exp4a = {"alphas": alphas, "glare_family": "tanh_3mode", "series": {}}
    series_names_4a = (
        [e.name for e in get_all_db_engines(tuned_params)]
        + ["MCF-Sharded", "Prov-IVM (No-Proj)"]
    )
    for sn in series_names_4a:
        exp4a["series"][sn] = {"idf1": [], "mota": [], "id_switches": [], "corrupted_veh_ratio": []}

    for alpha in alphas:
        sp_a = copy.deepcopy(base_spec)
        sp_a.alpha_hub = alpha
        sp_a.glare_family = "tanh_3mode"
        ds_a = PortolanSynthesizer(profile, sp_a, topo, device).synthesize(generate_workloads=False)
        cg_a = SpatioTemporalCandidateGraph(ds_a, device)

        eval_4a = list(get_all_db_engines(tuned_params)) + [get_mcf_sharded_engine(tuned_params)]
        for eng in eval_4a:
            trajs, _, _ = eng.recover_trajectories(cg_a)
            mt = compute_mtmc_metrics(trajs, ds_a["snap_veh"])
            exp4a["series"][eng.name]["idf1"].append(float(mt["idf1"]))
            exp4a["series"][eng.name]["mota"].append(float(mt["mota"]))
            exp4a["series"][eng.name]["id_switches"].append(int(mt["id_switches"]))
            exp4a["series"][eng.name]["corrupted_veh_ratio"].append(float(mt["corrupted_vehicle_ratio"]))

        pv_noproj = ProvenanceCascadingDBEngine(
            use_active_cert=False, base_threshold=float(tuned_params["Prov-IVM"])
        )
        tr_np, _, _ = pv_noproj.recover_trajectories(cg_a)
        mt_np = compute_mtmc_metrics(tr_np, ds_a["snap_veh"])
        exp4a["series"]["Prov-IVM (No-Proj)"]["idf1"].append(float(mt_np["idf1"]))
        exp4a["series"]["Prov-IVM (No-Proj)"]["mota"].append(float(mt_np["mota"]))
        exp4a["series"]["Prov-IVM (No-Proj)"]["id_switches"].append(int(mt_np["id_switches"]))
        exp4a["series"]["Prov-IVM (No-Proj)"]["corrupted_veh_ratio"].append(float(mt_np["corrupted_vehicle_ratio"]))

    idx_0 = alphas.index(0.00)
    idx_2 = alphas.index(0.02)
    exp4a["relative_idf1_drop_at_2pct"] = {}
    for sn in series_names_4a:
        v0 = exp4a["series"][sn]["idf1"][idx_0]
        v2 = exp4a["series"][sn]["idf1"][idx_2]
        drop_pct = 100.0 * (v0 - v2) / max(1e-6, v0)
        exp4a["relative_idf1_drop_at_2pct"][sn] = float(drop_pct)
        print(f"[Exp4a tanh_3mode] {sn:18s}: alpha=0% IDF1={v0:.3f} -> alpha=2% IDF1={v2:.3f} (drop={drop_pct:.1f}%)")

    # Out-of-family `gamma_rank4` glare stress evaluation (non-orthogonal rank-4 + ReLU/power-law clipping)
    oof_alphas = [0.00, 0.03, 0.05]
    oof_series: Dict[str, Dict[str, List]] = {
        sn: {"idf1": [], "mota": [], "id_switches": []}
        for sn in ["Prov-IVM", "Prov-IVM (No-Proj)", "MCF-Sharded", "MCF-Tracker", "Global-BM", "Prob-Path"]
    }
    for alpha_oof in oof_alphas:
        sp_oof = copy.deepcopy(base_spec)
        sp_oof.alpha_hub = alpha_oof
        sp_oof.glare_family = "gamma_rank4"
        ds_oof = PortolanSynthesizer(profile, sp_oof, topo, device).synthesize(generate_workloads=False)
        cg_oof = SpatioTemporalCandidateGraph(ds_oof, device)
        oof_engines = [
            ("Prov-IVM", ProvenanceCascadingDBEngine(base_threshold=float(tuned_params["Prov-IVM"]))),
            ("Prov-IVM (No-Proj)", ProvenanceCascadingDBEngine(use_active_cert=False, base_threshold=float(tuned_params["Prov-IVM"]))),
            ("MCF-Sharded", get_mcf_sharded_engine(tuned_params)),
            ("MCF-Tracker", MinCostNetFlowEngine(match_reward=float(tuned_params["MCF-Tracker"]), sharded=False)),
            ("Global-BM", EagerHungarianEngine(sim_threshold=float(tuned_params["Global-BM"]))),
            ("Prob-Path", TopoMarkovMAPEngine(base_prior=float(tuned_params["Prob-Path"]))),
        ]
        for sn, eng_o in oof_engines:
            tr_o, _, _ = eng_o.recover_trajectories(cg_oof)
            mt_o = compute_mtmc_metrics(tr_o, ds_oof["snap_veh"])
            oof_series[sn]["idf1"].append(float(mt_o["idf1"]))
            oof_series[sn]["mota"].append(float(mt_o["mota"]))
            oof_series[sn]["id_switches"].append(int(mt_o["id_switches"]))
        print(
            f"[Exp4a OOF gamma_rank4 alpha={alpha_oof:.2f}] "
            f"Prov-IVM={oof_series['Prov-IVM']['idf1'][-1]:.3f} (IDSW={oof_series['Prov-IVM']['id_switches'][-1]}) vs "
            f"No-Proj={oof_series['Prov-IVM (No-Proj)']['idf1'][-1]:.3f} vs "
            f"MCF-Sharded={oof_series['MCF-Sharded']['idf1'][-1]:.3f} (IDSW={oof_series['MCF-Sharded']['id_switches'][-1]})"
        )
    exp4a["out_of_family_gamma_rank4"] = {
        "alphas": oof_alphas,
        "description": "Out-of-family rank-4 non-orthogonal correlated glare + ReLU/gamma^0.60 channel clipping",
        "series": oof_series,
    }

    # --- 4b: Q2 Co-Movement Join Recall Collapse (k in [2..8], rho_occ=0.20) ---
    sp_q2 = copy.deepcopy(base_spec)
    sp_q2.rho_occ = 0.20
    sp_q2.target_snapshots = 28000
    ds_q2 = PortolanSynthesizer(profile, sp_q2, topo, device).synthesize(generate_workloads=True)
    cg_q2 = SpatioTemporalCandidateGraph(ds_q2, device)

    eng_map = {e.name: e for e in get_all_db_engines(tuned_params)}
    eng_map["MCF-Sharded"] = get_mcf_sharded_engine(tuned_params)
    trajs_q2 = {}
    for en, eng in eng_map.items():
        trajs_q2[en], _, _ = eng.recover_trajectories(cg_q2)

    k_list = [2, 3, 4, 5, 6, 7, 8]
    eager_4b = ["Global-BM", "MCF-Tracker", "MCF-Sharded", "Prob-Path", "Vec-Filter", "Online-Clust"]
    exp4b = {
        "k_hops": k_list,
        "rho_occ": 0.20,
        "delta_t_sec": float(sp_q2.q2_time_window_sec),
        "gt_pair_counts": [len(ds_q2["workloads"]["Q2"]["gt_pairs_by_k"][k]) for k in k_list],
        "theoretical_p2k_recall": [float((0.78) ** (2 * (k - 1))) for k in k_list],
        "engines": {
            en: {"f1": [], "recall": [], "precision": []}
            for en in eager_4b + ["Prov-IVM"]
        },
    }
    prov_eng: ProvenanceCascadingDBEngine = eng_map["Prov-IVM"]
    for k_h in k_list:
        gt_p = ds_q2["workloads"]["Q2"]["gt_pairs_by_k"][k_h]
        for en in eager_4b:
            res_e = evaluate_q2_eager_from_trajs(
                trajs_q2[en], ds_q2["snap_cam"], ds_q2["snap_time"], ds_q2["snap_veh"],
                k_h, sp_q2.q2_time_window_sec, gt_p,
            )
            exp4b["engines"][en]["f1"].append(float(res_e["f1"]))
            exp4b["engines"][en]["recall"].append(float(res_e["recall"]))
            exp4b["engines"][en]["precision"].append(float(res_e["precision"]))

        res_pv, _ = prov_eng.execute_coupled_q2_join(cg_q2, k_h, sp_q2.q2_time_window_sec, gt_p)
        exp4b["engines"]["Prov-IVM"]["f1"].append(float(res_pv["f1"]))
        exp4b["engines"]["Prov-IVM"]["recall"].append(float(res_pv["recall"]))
        exp4b["engines"]["Prov-IVM"]["precision"].append(float(res_pv["precision"]))
        print(
            f"[Exp4b] k={k_h}: Prov-IVM F1={res_pv['f1']:.3f} vs MCF-Sharded={exp4b['engines']['MCF-Sharded']['f1'][-1]:.3f} "
            f"vs MCF={exp4b['engines']['MCF-Tracker']['f1'][-1]:.3f} vs BM={exp4b['engines']['Global-BM']['f1'][-1]:.3f}"
        )

    # --- 4c: Edge-Cloud Privacy Compensation (sigma_dp vs hops k in [1..6], 100% blind noise estimator) ---
    sigmas_dp = [0.00, 0.03, 0.05, 0.07, 0.10, 0.13]
    hops_4c = [1, 2, 3, 4, 6]
    exp4c = {
        "zero_oracle_sigma_dp": True,
        "sigma_dp": sigmas_dp,
        "hops": hops_4c,
        "no_topology_k6": [],
        "no_topology_k3": [],
        "by_hops": {str(kh): [] for kh in hops_4c},
        "path_purity_by_hops": {str(kh): [] for kh in hops_4c},
    }
    for sdp in sigmas_dp:
        sp_dp = copy.deepcopy(base_spec)
        sp_dp.sigma_dp = sdp
        sp_dp.alpha_hub = 0.005
        ds_dp = PortolanSynthesizer(profile, sp_dp, topo, device).synthesize(generate_workloads=True)
        cg_dp = SpatioTemporalCandidateGraph(ds_dp, device)
        pv_dp = ProvenanceCascadingDBEngine(
            use_active_cert=False, base_threshold=float(tuned_params["Prov-IVM"])
        )
        q1_qs = ds_dp["workloads"]["Q1"]

        nt_k6 = pv_dp.evaluate_privacy_trellis_recovery(cg_dp, q1_qs, k_hops=6, topology_free=True)
        nt_k3 = pv_dp.evaluate_privacy_trellis_recovery(cg_dp, q1_qs, k_hops=3, topology_free=True)
        exp4c["no_topology_k6"].append(float(nt_k6["reid_accuracy"]))
        exp4c["no_topology_k3"].append(float(nt_k3["reid_accuracy"]))
        for kh in hops_4c:
            rk = pv_dp.evaluate_privacy_trellis_recovery(cg_dp, q1_qs, k_hops=kh, topology_free=False)
            exp4c["by_hops"][str(kh)].append(float(rk["reid_accuracy"]))
            exp4c["path_purity_by_hops"][str(kh)].append(float(rk["path_purity"]))

        print(
            f"[Exp4c blind-DP] sigma_dp={sdp:.2f}: NoTopo(k=6)={exp4c['no_topology_k6'][-1]:.3f} | "
            + " | ".join(f"k={kh}:{exp4c['by_hops'][str(kh)][-1]:.3f}" for kh in hops_4c)
        )

    # --- 4d: 4-Step Component Ablation of Prov-IVM starting from MCF-Sharded ---
    sp_ab = copy.deepcopy(base_spec)
    sp_ab.alpha_hub = 0.03
    sp_ab.rho_occ = 0.20
    ds_ab = PortolanSynthesizer(profile, sp_ab, topo, device).synthesize(generate_workloads=True)
    cg_ab = SpatioTemporalCandidateGraph(ds_ab, device)
    wl_ab = ds_ab["workloads"]
    gt_q2_k5 = wl_ab["Q2"]["gt_pairs_by_k"][5]
    unlearn_ab = {i for i, v in enumerate(ds_ab["snap_veh"]) if int(v) in set(wl_ab["Q4"]["unlearn_vehicle_ids"])}
    active_ab = np.ones(cg_ab.n_snaps, dtype=bool)
    for r_i in unlearn_ab:
        active_ab[r_i] = False

    mcf_sh_ab = get_mcf_sharded_engine(tuned_params)
    tr_s0, _, _ = mcf_sh_ab.recover_trajectories(cg_ab)
    mt_s0 = compute_mtmc_metrics(tr_s0, ds_ab["snap_veh"])
    q2_s0 = evaluate_q2_eager_from_trajs(tr_s0, ds_ab["snap_cam"], ds_ab["snap_time"], ds_ab["snap_veh"], 5, sp_ab.q2_time_window_sec, gt_q2_k5)
    _, _, t_sh_aff = mcf_sh_ab.recompute_affected_shards(cg_ab, unlearn_ab, active_mask=active_ab)

    pv_s1 = ProvenanceCascadingDBEngine(use_active_cert=True, base_threshold=float(tuned_params["Prov-IVM"]))
    tr_s1, _, _ = pv_s1.recover_trajectories(cg_ab)
    mt_s1 = compute_mtmc_metrics(tr_s1, ds_ab["snap_veh"])
    q2_s1_eager = evaluate_q2_eager_from_trajs(tr_s1, ds_ab["snap_cam"], ds_ab["snap_time"], ds_ab["snap_veh"], 5, sp_ab.q2_time_window_sec, gt_q2_k5)
    q2_s2_cpg, _ = pv_s1.execute_coupled_q2_join(cg_ab, 5, sp_ab.q2_time_window_sec, gt_q2_k5)
    _, _, t_pv_full_ab = ProvenanceCascadingDBEngine(base_threshold=float(tuned_params["Prov-IVM"])).recover_trajectories(cg_ab, active_mask=active_ab)
    _, _, t_pv_ivm_ab = pv_s1.execute_bounded_ivm_repair(cg_ab, unlearn_ab, materialize_view=True)
    mt_s3_ivm = compute_mtmc_metrics(pv_s1.last_repaired_trajs, ds_ab["snap_veh"], active_mask=active_ab)

    prov_ablation = {
        "conditions": {"alpha_hub": 0.03, "rho_occ": 0.20, "q2_k_hops": 5, "q4b_deleted_snaps": len(unlearn_ab)},
        "steps": {
            "0_Base_MCF_Sharded": {
                "idf1": float(mt_s0["idf1"]),
                "id_switches": int(mt_s0["id_switches"]),
                "q2_k5_f1": float(q2_s0["f1"]),
                "q4b_update_ms": float(t_sh_aff * 1000.0),
                "q4b_mode": "affected_shard_recompute",
            },
            "1_Plus_Conflict_SVD": {
                "idf1": float(mt_s1["idf1"]),
                "id_switches": int(mt_s1["id_switches"]),
                "q2_k5_f1": float(q2_s1_eager["f1"]),
                "q4b_update_ms": float(t_pv_full_ab * 1000.0),
                "q4b_mode": "full_solver_recompute",
            },
            "2_Plus_Horizon_CPG_Q2": {
                "idf1": float(mt_s1["idf1"]),
                "id_switches": int(mt_s1["id_switches"]),
                "q2_k5_f1": float(q2_s2_cpg["f1"]),
                "q4b_update_ms": float(t_pv_full_ab * 1000.0),
                "q4b_mode": "full_solver_recompute",
            },
            "3_Full_Prov_IVM_Plus_CPG_Q4": {
                "idf1": float(mt_s1["idf1"]),
                "post_q4b_idf1": float(mt_s3_ivm["idf1"]),
                "id_switches": int(mt_s1["id_switches"]),
                "q2_k5_f1": float(q2_s2_cpg["f1"]),
                "q4b_update_ms": float(t_pv_ivm_ab * 1000.0),
                "q4b_propagation_ms": float(pv_s1.last_ivm_propagation_ms),
                "q4b_mode": "damped_cpg_ivm",
            },
        },
    }
    print(f"[Exp4d Prov-IVM 4-Step Ablation] {json.dumps(prov_ablation['steps'])}")

    exp4_rows = []
    for i, a_val in enumerate(alphas):
        for sn in series_names_4a:
            exp4_rows.append({
                "sub_exp": "4a_blackhole",
                "x_param": a_val,
                "series": sn,
                "metric_primary": round(exp4a["series"][sn]["idf1"][i], 4),
                "metric_secondary": round(exp4a["series"][sn]["mota"][i], 4),
            })
    for i, k_val in enumerate(k_list):
        for en in exp4b["engines"]:
            exp4_rows.append({
                "sub_exp": "4b_comovement_q2",
                "x_param": k_val,
                "series": en,
                "metric_primary": round(exp4b["engines"][en]["f1"][i], 4),
                "metric_secondary": round(exp4b["engines"][en]["recall"][i], 4),
            })
    for i, s_val in enumerate(sigmas_dp):
        exp4_rows.append({
            "sub_exp": "4c_privacy_dp",
            "x_param": s_val,
            "series": "NoTopo-k6",
            "metric_primary": round(exp4c["no_topology_k6"][i], 4),
            "metric_secondary": 0.0,
        })
        for kh in hops_4c:
            exp4_rows.append({
                "sub_exp": "4c_privacy_dp",
                "x_param": s_val,
                "series": f"Topo-Trellis-k{kh}",
                "metric_primary": round(exp4c["by_hops"][str(kh)][i], 4),
                "metric_secondary": round(exp4c["path_purity_by_hops"][str(kh)][i], 4),
            })

    with open(RESULTS_DIR / "exp4_stress_testing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp4_rows[0].keys()))
        writer.writeheader()
        writer.writerows(exp4_rows)

    return {
        "exp4a_blackhole": exp4a,
        "exp4b_comovement": exp4b,
        "exp4c_privacy": exp4c,
        "prov_ivm_component_ablation": prov_ablation,
    }


def run_exp3b_district_scale(device: torch.device, tuned_params: Dict[str, float]) -> Dict:
    """
    Exp 3b: Multi-Scale Downstream Engine Benchmark at District Scale
    (Portolan-Small: 200 cameras, 15,000 vehicles, ~202K snapshots)
    Evaluates all 6 tuned paradigms + `MCF-Sharded` + sparse `Global-BM (Time-W)` + Q4b GDPR unlearning IVM
    with affected-shard `MCF-Sharded` baseline and strict neighborhood link agreement.
    """
    print("\n================ [Exp 3b] Downstream Scaling on Portolan-Small (200 Cams, 202K Snaps) ================")
    profile = get_seed_profile("CityFlow-Seed")
    spec = get_benchmark_spec("Portolan-Small", seed=42)
    topo = RoadCameraTopology(spec.n_cam, seed=42)
    ds = PortolanSynthesizer(profile, spec, topo, device).synthesize(generate_workloads=True)
    cg = SpatioTemporalCandidateGraph(ds, device)
    wl = ds["workloads"]
    k_q2 = wl["Q2"]["default_k"]
    gt_q2 = wl["Q2"]["gt_pairs_by_k"][k_q2]

    engines_res = {}
    csv_rows = []
    pv_small = None
    mcf_sh_small = get_mcf_sharded_engine(tuned_params)
    mcf_lat_ms = 0.0
    mcf_sh_lat_ms = 0.0
    bm_lat_ms = 0.0

    eval_list: List[Tuple[str, BaseDBEngine]] = [(e.name, e) for e in get_all_db_engines(tuned_params)]
    eval_list.append(("MCF-Sharded", mcf_sh_small))
    eval_list.append(
        (
            "Global-BM (Time-W)",
            EagerHungarianEngine(
                sim_threshold=float(tuned_params["Global-BM"]),
                window_sec=45.0,
                overlap_sec=35.0,
            ),
        )
    )

    for disp_name, eng in eval_list:
        trajs, _, lat_s = eng.recover_trajectories(cg)
        mt = compute_mtmc_metrics(trajs, ds["snap_veh"])
        lat_ms = lat_s * 1000.0
        q1_f1 = evaluate_q1_from_trajs(trajs, cg.n_snaps, wl["Q1"])
        if isinstance(eng, ProvenanceCascadingDBEngine):
            q2_r, q2_s = eng.execute_coupled_q2_join(cg, k_q2, wl["Q2"]["delta_t_sec"], gt_q2)
            pv_small = eng
        else:
            q2_r = evaluate_q2_eager_from_trajs(
                trajs, ds["snap_cam"], ds["snap_time"], ds["snap_veh"],
                k_q2, wl["Q2"]["delta_t_sec"], gt_q2,
            )
            q2_s = 0.0
        q3_sc = evaluate_q3_from_trajs(trajs, ds["snap_cam"], wl["Q3"])["q3_score"]
        if disp_name == "MCF-Tracker":
            mcf_lat_ms = lat_ms
        elif disp_name == "MCF-Sharded":
            mcf_sh_lat_ms = lat_ms
        elif disp_name == "Global-BM":
            bm_lat_ms = lat_ms

        engines_res[disp_name] = {
            "idf1": float(mt["idf1"]),
            "mota": float(mt["mota"]),
            "mota_link": float(mt["mota_link"]),
            "precision": float(mt["precision"]),
            "recall": float(mt["recall"]),
            "q1_f1": float(q1_f1),
            "q2_f1": float(q2_r["f1"]),
            "q3_score": float(q3_sc),
            "latency_ms": float(lat_ms),
            "q2_join_ms": float(q2_s * 1000.0),
        }
        csv_rows.append({"engine": disp_name, **{k: round(v, 4) for k, v in engines_res[disp_name].items()}})
        print(
            f"[Exp3b Small 202K] {disp_name:18s}: IDF1={mt['idf1']:.3f}, MOTA={mt['mota_link']:.3f}, "
            f"Q1={q1_f1:.3f}, Q2={q2_r['f1']:.3f}, Q3={q3_sc:.3f}, Lat={lat_ms:.1f}ms"
        )

    # Measure Q4b 35-vehicle GDPR unlearning IVM + affected-shard MCF-Sharded + same-engine full recomputation
    unlearn_set = set(wl["Q4"]["unlearn_vehicle_ids"])
    rem_unlearn = {i for i, v in enumerate(ds["snap_veh"]) if int(v) in unlearn_set}
    active_m = np.ones(cg.n_snaps, dtype=bool)
    for r_i in rem_unlearn:
        active_m[r_i] = False

    ivm_times, prop_times = [], []
    touched_cnt = 0
    next_ivm_sm = None
    for _ in range(3):
        next_ivm_sm, touched_cnt, t_ivm = pv_small.execute_bounded_ivm_repair(cg, rem_unlearn, materialize_view=True)
        ivm_times.append(t_ivm * 1000.0)
        prop_times.append(float(pv_small.last_ivm_propagation_ms))
    ivm_small_ms = float(np.mean(ivm_times))
    ivm_prop_ms = float(np.mean(prop_times))
    tr_ivm_sm = pv_small.last_repaired_trajs

    _, _, t_msh_aff = mcf_sh_small.recompute_affected_shards(cg, rem_unlearn, active_mask=active_m)
    msh_aff_ms = t_msh_aff * 1000.0

    pv_exact_small = ProvenanceCascadingDBEngine(base_threshold=float(tuned_params["Prov-IVM"]))
    tr_ex_sm, next_ex_sm, t_pv_exact = pv_exact_small.recover_trajectories(cg, active_mask=active_m)
    pv_exact_small_ms = t_pv_exact * 1000.0
    cpg_mem_small_mb = pv_small.get_cpg_memory_mb()
    cpg_bpt = pv_small.get_cpg_bytes_per_tuple()
    idf1_ivm_sm = float(compute_mtmc_metrics(tr_ivm_sm, ds["snap_veh"], active_mask=active_m)["idf1"])
    idf1_ex_sm = float(compute_mtmc_metrics(tr_ex_sm, ds["snap_veh"], active_mask=active_m)["idf1"])
    link_agr_sm = float(100.0 * np.mean(next_ivm_sm[active_m] == next_ex_sm[active_m]))
    nbr_agr_sm = _neighborhood_agreement_pct(next_ivm_sm, next_ex_sm, pv_small.last_neighborhood_nodes, active_m)
    idf1_ret_sm = float(100.0 * idf1_ivm_sm / max(1e-6, idf1_ex_sm))

    print(
        f"[Exp3b Small Q4b] |Delta|={len(rem_unlearn)} snaps, touched={touched_cnt}, CPG={cpg_mem_small_mb:.2f}MB ({cpg_bpt:.1f}B/tup): "
        f"Prov-IVM={ivm_small_ms:.2f}ms (prop={ivm_prop_ms:.2f}ms) vs MCF-Sharded-Aff={msh_aff_ms:.1f}ms ({msh_aff_ms/max(1e-3, ivm_small_ms):.1f}x) "
        f"vs Prov-Full={pv_exact_small_ms:.1f}ms ({pv_exact_small_ms/max(1e-3, ivm_small_ms):.1f}x) "
        f"[nbr_agr={nbr_agr_sm:.2f}%, global_agr={link_agr_sm:.2f}%]"
    )

    with open(RESULTS_DIR / "exp3b_district_scale.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    return {
        "n_cam": spec.n_cam,
        "n_veh": spec.n_veh,
        "total_snapshots": cg.n_snaps,
        "cgraph_build_sec": float(cg.build_time_sec),
        "cpg_memory_mb": float(cpg_mem_small_mb),
        "cpg_bytes_per_tuple": float(cpg_bpt),
        "engines": engines_res,
        "q4b_district_ivm": {
            "delta_snapshots": len(rem_unlearn),
            "touched_snapshots": int(touched_cnt),
            "touched_fraction_pct": float(100.0 * touched_cnt / cg.n_snaps),
            "neighborhood_snapshots": int(len(pv_small.last_neighborhood_nodes)),
            "max_wave_depth": int(pv_small.last_max_wave_depth),
            "cpg_memory_mb": float(cpg_mem_small_mb),
            "cpg_bytes_per_tuple": float(cpg_bpt),
            "ivm_propagation_ms": ivm_prop_ms,
            "prov_ivm_ms": ivm_small_ms,
            "mcf_sharded_affected_ms": msh_aff_ms,
            "mcf_sharded_full_ms": mcf_sh_lat_ms,
            "prov_exact_recomp_ms": pv_exact_small_ms,
            "speedup_vs_mcf_sharded_affected": float(msh_aff_ms / max(1e-3, ivm_small_ms)),
            "speedup_vs_prov_exact": float(pv_exact_small_ms / max(1e-3, ivm_small_ms)),
            "speedup_vs_mcf": float(mcf_lat_ms / max(1e-3, ivm_small_ms)),
            "speedup_vs_bm": float(bm_lat_ms / max(1e-3, ivm_small_ms)),
            "prov_ivm_idf1": idf1_ivm_sm,
            "prov_exact_idf1": idf1_ex_sm,
            "neighborhood_link_agreement_pct": nbr_agr_sm,
            "link_agreement_pct": link_agr_sm,
            "idf1_retention_pct": idf1_ret_sm,
        },
    }


_HEAVY_SHARED: Dict = {}


def _heavy_solver_worker(name: str, q_prog, q_out) -> None:
    cg = _HEAVY_SHARED["cg"]
    ds = _HEAVY_SHARED["ds"]
    tp = _HEAVY_SHARED["tp"]
    wl = ds["workloads"]
    k_q2 = wl["Q2"]["default_k"]
    gt_q2 = wl["Q2"]["gt_pairs_by_k"][k_q2]
    if name == "Global-BM (Time-W)":
        eng = EagerHungarianEngine(sim_threshold=float(tp["Global-BM"]), window_sec=45.0, overlap_sec=35.0)
    elif name == "Online-Clust":
        eng = StreamGreedyClustEngine(sim_threshold=float(tp["Online-Clust"]))
    else:
        eng = MinCostNetFlowEngine(match_reward=float(tp["MCF-Tracker"]), sharded=False)
    trajs, _, lat_s = eng.recover_trajectories(cg)
    q_prog.put(float(lat_s))
    mt = compute_mtmc_metrics(trajs, ds["snap_veh"])
    q1 = evaluate_q1_from_trajs(trajs, cg.n_snaps, wl["Q1"])
    q2_r = evaluate_q2_eager_from_trajs(
        trajs, ds["snap_cam"], ds["snap_time"], ds["snap_veh"], k_q2, wl["Q2"]["delta_t_sec"], gt_q2
    )
    q3 = evaluate_q3_from_trajs(trajs, ds["snap_cam"], wl["Q3"])["q3_score"]
    q_out.put({
        "status": "OK",
        "idf1": float(mt["idf1"]),
        "mota_link": float(mt["mota_link"]),
        "precision": float(mt["precision"]),
        "recall": float(mt["recall"]),
        "q1_f1": float(q1),
        "q2_f1": float(q2_r["f1"]),
        "q3_score": float(q3),
        "latency_sec": float(lat_s),
        "q2_join_sec": 0.0,
    })


def _run_single_solver_serial_with_timeout(h_name: str, sla_timeout_sec: int) -> Dict:
    """
    Execute a single heavy unsharded solver (`Global-BM (Time-W)` or `MCF-Tracker`) SERIALLY in
    complete CPU isolation (never concurrent with any other solver) with a hard wall-clock cap.
    """
    ctx = mp.get_context("fork")
    qp, qo = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=_heavy_solver_worker, args=(h_name, qp, qo))
    t0 = time.perf_counter()
    proc.start()
    lat_v = None
    while time.perf_counter() < t0 + sla_timeout_sec:
        if not qp.empty():
            lat_v = qp.get()
            break
        if not proc.is_alive():
            break
        time.sleep(0.25)
    if lat_v is None and not qp.empty():
        lat_v = qp.get()

    if lat_v is None or lat_v > sla_timeout_sec:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
        proc.join()
        return {"status": f"TIMEOUT (>{sla_timeout_sec}s)", "latency_sec": lat_v}

    # Solver finished within SLA; wait cleanly for post-hoc metric evaluation
    res = qo.get()
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return res


def run_exp3c_metropolitan_scaling(
    device: torch.device,
    tuned_params: Dict[str, float],
    sla_timeout_sec: int = 600,
    tiers_filter: Optional[List[str]] = None,
    existing_3c: Optional[Dict] = None,
    tier_ckpt_hook=None,
) -> Dict:
    """Exp 3c: Metropolitan (1,000 cams, 1.24M snaps) & CityScale (5,000 cams, 5.09M snaps) Serial SLA Scaling."""
    print(f"\n================ [Exp 3c] Metropolitan & CityScale Serial SLA Scaling (Timeout={sla_timeout_sec}s) ================")
    profile = get_seed_profile("CityFlow-Seed")
    tiers_out = dict(existing_3c.get("tiers", {})) if existing_3c else {}
    run_tiers = tiers_filter or ["Portolan-Medium", "Portolan-CityScale"]

    for tier_name in run_tiers:
        spec = get_benchmark_spec(tier_name, seed=42)
        topo = RoadCameraTopology(spec.n_cam, seed=42)
        ds = PortolanSynthesizer(profile, spec, topo, device).synthesize(generate_workloads=True)
        cg = SpatioTemporalCandidateGraph(ds, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        wl = ds["workloads"]
        k_q2 = wl["Q2"]["default_k"]
        gt_q2 = wl["Q2"]["gt_pairs_by_k"][k_q2]
        eng_res = {}
        q4_info = None
        mcf_sh_tier = get_mcf_sharded_engine(tuned_params)

        for name, eng in [
            ("Vec-Filter", VecDBFilteredANNEngine(sim_threshold=float(tuned_params["Vec-Filter"]))),
            ("Prob-Path", TopoMarkovMAPEngine(base_prior=float(tuned_params["Prob-Path"]))),
            ("Global-BM", EagerHungarianEngine(sim_threshold=float(tuned_params["Global-BM"]))),
            ("MCF-Sharded", mcf_sh_tier),
            ("Prov-IVM", ProvenanceCascadingDBEngine(base_threshold=float(tuned_params["Prov-IVM"]))),
        ]:
            trajs, _, lat_s = eng.recover_trajectories(cg)
            if lat_s > sla_timeout_sec:
                eng_res[name] = {"status": f"TIMEOUT (>{sla_timeout_sec}s)", "latency_sec": float(lat_s)}
                continue
            mt = compute_mtmc_metrics(trajs, ds["snap_veh"])
            q1 = evaluate_q1_from_trajs(trajs, cg.n_snaps, wl["Q1"])
            if isinstance(eng, ProvenanceCascadingDBEngine):
                q2_r, q2_s = eng.execute_coupled_q2_join(cg, k_q2, wl["Q2"]["delta_t_sec"], gt_q2)
                unlearn_set = set(wl["Q4"]["unlearn_vehicle_ids"])
                rem_unlearn = {i for i, v in enumerate(ds["snap_veh"]) if int(v) in unlearn_set}
                active_m = np.ones(cg.n_snaps, dtype=bool)
                for r_i in rem_unlearn:
                    active_m[r_i] = False
                ivm_times, prop_times = [], []
                touched_cnt = 0
                next_ivm = None
                for _ in range(3):
                    next_ivm, touched_cnt, t_ivm = eng.execute_bounded_ivm_repair(cg, rem_unlearn, materialize_view=True)
                    ivm_times.append(t_ivm * 1000.0)
                    prop_times.append(float(eng.last_ivm_propagation_ms))
                ivm_ms = float(np.mean(ivm_times))
                prop_ms = float(np.mean(prop_times))

                _, _, t_msh_aff = mcf_sh_tier.recompute_affected_shards(cg, rem_unlearn, active_mask=active_m)
                msh_aff_ms = t_msh_aff * 1000.0

                pv_exact = ProvenanceCascadingDBEngine(base_threshold=float(tuned_params["Prov-IVM"]))
                _, next_ex, t_pv_ex = pv_exact.recover_trajectories(cg, active_mask=active_m)
                pv_ex_ms = float(t_pv_ex * 1000.0)
                nbr_agr = _neighborhood_agreement_pct(next_ivm, next_ex, eng.last_neighborhood_nodes, active_m)
                q4_info = {
                    "delta_snapshots": len(rem_unlearn),
                    "touched_snapshots": int(touched_cnt),
                    "touched_fraction_pct": float(100.0 * touched_cnt / cg.n_snaps),
                    "neighborhood_snapshots": int(len(eng.last_neighborhood_nodes)),
                    "ivm_propagation_ms": prop_ms,
                    "prov_ivm_ms": ivm_ms,
                    "mcf_sharded_affected_ms": msh_aff_ms,
                    "prov_exact_recomp_ms": pv_ex_ms,
                    "speedup_vs_mcf_sharded_affected": float(msh_aff_ms / max(1e-3, ivm_ms)),
                    "speedup_vs_prov_exact": float(pv_ex_ms / max(1e-3, ivm_ms)),
                    "neighborhood_link_agreement_pct": nbr_agr,
                    "link_agreement_pct": float(100.0 * np.mean(next_ivm[active_m] == next_ex[active_m])),
                    "cpg_memory_mb": float(eng.get_cpg_memory_mb()),
                    "cpg_bytes_per_tuple": float(eng.get_cpg_bytes_per_tuple()),
                }
            else:
                q2_r = evaluate_q2_eager_from_trajs(
                    trajs, ds["snap_cam"], ds["snap_time"], ds["snap_veh"], k_q2, wl["Q2"]["delta_t_sec"], gt_q2
                )
                q2_s = 0.0
            q3 = evaluate_q3_from_trajs(trajs, ds["snap_cam"], wl["Q3"])["q3_score"]
            eng_res[name] = {
                "status": "OK",
                "idf1": float(mt["idf1"]),
                "mota_link": float(mt["mota_link"]),
                "precision": float(mt["precision"]),
                "recall": float(mt["recall"]),
                "q1_f1": float(q1),
                "q2_f1": float(q2_r["f1"]),
                "q3_score": float(q3),
                "latency_sec": float(lat_s),
                "q2_join_sec": float(q2_s),
            }
            print(
                f"[Exp3c {tier_name}] {name:18s}: IDF1={mt['idf1']:.3f}, MOTA={mt['mota_link']:.3f}, "
                f"Q1={q1:.3f}, Q2={q2_r['f1']:.3f}, Q3={q3:.3f}, Lat={lat_s:.2f}s"
            )

        # Time Online-Clust, Global-BM (Time-W), and unsharded MCF-Tracker STRICTLY ONE AT A TIME (zero CPU contention!)
        _HEAVY_SHARED.clear()
        _HEAVY_SHARED["cg"] = cg
        _HEAVY_SHARED["ds"] = ds
        _HEAVY_SHARED["tp"] = tuned_params
        for h_name in ["Online-Clust", "Global-BM (Time-W)", "MCF-Tracker"]:
            res_h = _run_single_solver_serial_with_timeout(h_name, sla_timeout_sec=sla_timeout_sec)
            eng_res[h_name] = res_h
            print(f"[Exp3c {tier_name} Serial] {h_name:18s}: status={res_h['status']}, lat={res_h.get('latency_sec')}")
        _HEAVY_SHARED.clear()

        if q4_info is not None:
            print(
                f"[Exp3c {tier_name} Q4b] |Delta|={q4_info['delta_snapshots']}, Prov-IVM={q4_info['prov_ivm_ms']:.2f}ms "
                f"(prop={q4_info['ivm_propagation_ms']:.2f}ms) vs MCF-Sharded-Aff={q4_info['mcf_sharded_affected_ms']:.1f}ms "
                f"({q4_info['speedup_vs_mcf_sharded_affected']:.1f}x) vs Prov-Full={q4_info['prov_exact_recomp_ms']:.1f}ms "
                f"({q4_info['speedup_vs_prov_exact']:.1f}x), nbr_agr={q4_info['neighborhood_link_agreement_pct']:.2f}%"
            )

        t_rec = {
            "tier": tier_name,
            "n_cam": int(spec.n_cam),
            "n_veh": int(spec.n_veh),
            "total_snapshots": int(cg.n_snaps),
            "serial_isolated_timing": True,
            "sla_timeout_sec": int(sla_timeout_sec),
            "cgraph_build_sec": float(cg.build_time_sec),
            "q4b_ivm": q4_info,
            "engines": eng_res,
        }
        tiers_out[tier_name] = t_rec
        cur_res = {"serial_isolated_timing": True, "sla_timeout_sec": sla_timeout_sec, "tiers": tiers_out}
        if tier_ckpt_hook is not None:
            tier_ckpt_hook(cur_res)

        del ds, cg, wl, eng_res, mcf_sh_tier
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_rows = []
    for t_key in ["Portolan-Medium", "Portolan-CityScale"]:
        if t_key not in tiers_out:
            continue
        tr_d = tiers_out[t_key]
        for eng_name, d in tr_d["engines"].items():
            csv_rows.append({
                "tier": t_key,
                "n_cam": tr_d["n_cam"],
                "total_snapshots": tr_d["total_snapshots"],
                "engine": eng_name,
                "status": d["status"],
                "idf1": round(d["idf1"], 4) if "idf1" in d else "",
                "mota_link": round(d["mota_link"], 4) if "mota_link" in d else "",
                "q1_f1": round(d["q1_f1"], 4) if "q1_f1" in d else "",
                "q2_f1": round(d["q2_f1"], 4) if "q2_f1" in d else "",
                "q3_score": round(d["q3_score"], 4) if "q3_score" in d else "",
                "latency_sec": round(d["latency_sec"], 2) if d.get("latency_sec") is not None else f">{sla_timeout_sec}",
            })
    if csv_rows:
        with open(RESULTS_DIR / "exp3c_metropolitan_scaling.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    return {"serial_isolated_timing": True, "sla_timeout_sec": sla_timeout_sec, "tiers": tiers_out}


def run_exp5_error_attribution_and_ivm(
    device: torch.device, tuned_params: Dict[str, float], existing_5a: Dict = None
) -> Dict:
    """
    Exp 5:
      5a: Exact Unclamped 5-Stage Shared-Tape Counterfactual Attribution (S0 -> S1[Det] -> S2[DeGlare] -> S3[Viewpoint] -> S4[Clock])
      5b: True O(|Delta|) Q4 Incremental View Maintenance vs. Affected-Shard `MCF-Sharded` & Full Recomputation
    """
    print("\n================ [Exp 5] Exact Unclamped 5-Stage Shared-Tape Attribution & True O(|Delta|) Q4 IVM ================")
    profile = get_seed_profile("CityFlow-Seed")
    seeds = [42, 43, 44]
    eval_engines = ["Global-BM", "MCF-Tracker", "MCF-Sharded", "Prob-Path", "Prov-IVM"]
    stage_names = [
        "S0_Full_Noisy",
        "S1_Oracle_Detection",
        "S2_Oracle_DeGlare",
        "S3_Oracle_Viewpoint",
        "S4_Oracle_Clock_Kin",
    ]

    if existing_5a is not None and "waterfall" in existing_5a and "stage_attribution" in existing_5a:
        waterfall = existing_5a["waterfall"]
        attribution = existing_5a["stage_attribution"]
        csv_rows = []
    else:
        waterfall_acc = {
            st: {en: {qk: [] for qk in ["idf1", "q1_f1", "q2_f1", "q3_score"]} for en in eval_engines}
            for st in stage_names
        }

        for sd in seeds:
            sp_wf = get_benchmark_spec("Portolan-Tiny", seed=sd)
            sp_wf.rho_occ = 0.18
            sp_wf.alpha_hub = 0.02
            sp_wf.sigma_tau = 4.5
            topo_wf = RoadCameraTopology(sp_wf.n_cam, seed=sd)
            stages_map = PortolanSynthesizer(profile, sp_wf, topo_wf, device).synthesize_oracle_waterfall(
                generate_workloads=True
            )
            for st_name in stage_names:
                ds_s = stages_map[st_name]
                oracle_clk = st_name == "S4_Oracle_Clock_Kin"
                cg_s = SpatioTemporalCandidateGraph(ds_s, device, use_oracle_clock=oracle_clk)
                wl_s = ds_s["workloads"]
                k_q2 = 5
                gt_q2 = wl_s["Q2"]["gt_pairs_by_k"][k_q2]
                eng_objs = {e.name: e for e in get_all_db_engines(tuned_params)}
                eng_objs["MCF-Sharded"] = get_mcf_sharded_engine(tuned_params)
                for en in eval_engines:
                    eng = eng_objs[en]
                    trajs, _, _ = eng.recover_trajectories(cg_s)
                    mt = compute_mtmc_metrics(trajs, ds_s["snap_veh"])
                    q1_f1 = evaluate_q1_from_trajs(trajs, cg_s.n_snaps, wl_s["Q1"])
                    if en == "Prov-IVM":
                        q2_f1, _ = eng.execute_coupled_q2_join(cg_s, k_q2, wl_s["Q2"]["delta_t_sec"], gt_q2)
                        q2_val = q2_f1["f1"]
                    else:
                        q2_val = evaluate_q2_eager_from_trajs(
                            trajs, ds_s["snap_cam"], ds_s["snap_time"], ds_s["snap_veh"],
                            k_q2, wl_s["Q2"]["delta_t_sec"], gt_q2,
                        )["f1"]
                    q3_val = evaluate_q3_from_trajs(trajs, ds_s["snap_cam"], wl_s["Q3"])["q3_score"]
                    waterfall_acc[st_name][en]["idf1"].append(float(mt["idf1"]))
                    waterfall_acc[st_name][en]["q1_f1"].append(float(q1_f1))
                    waterfall_acc[st_name][en]["q2_f1"].append(float(q2_val))
                    waterfall_acc[st_name][en]["q3_score"].append(float(q3_val))

        waterfall = {
            st: {
                en: {qk: float(np.mean(vals)) for qk, vals in qdict.items()}
                for en, qdict in edict.items()
            }
            for st, edict in waterfall_acc.items()
        }

        attribution = {}
        csv_rows = []
        for en in eval_engines:
            attribution[en] = {}
            for q_key in ["q1_f1", "q2_f1", "q3_score"]:
                s0 = float(waterfall["S0_Full_Noisy"][en][q_key])
                s1 = float(waterfall["S1_Oracle_Detection"][en][q_key])
                s2 = float(waterfall["S2_Oracle_DeGlare"][en][q_key])
                s3 = float(waterfall["S3_Oracle_Viewpoint"][en][q_key])
                s4 = float(waterfall["S4_Oracle_Clock_Kin"][en][q_key])

                total_err = 1.0 - s0
                err_det = s1 - s0
                err_hub = s2 - s1
                err_rot = s3 - s2
                err_clk = s4 - s3
                err_alg = 1.0 - s4

                attribution[en][q_key] = {
                    "noisy_score": s0,
                    "s1_score": s1,
                    "s2_score": s2,
                    "s3_score": s3,
                    "s4_score": s4,
                    "total_error": total_err,
                    "detection_occlusion_err": err_det,
                    "glare_hub_err": err_hub,
                    "viewpoint_drift_err": err_rot,
                    "visual_drift_blackhole_err": err_hub + err_rot,
                    "clock_kinematic_err": err_clk,
                    "matching_algorithm_err": err_alg,
                    "blackhole_amplification_factor": float(err_hub / 0.02),
                    "occlusion_amplification_factor": float(err_det / 0.18),
                }
                csv_rows.append({
                    "section": "5a_error_attribution",
                    "engine": en,
                    "query_or_tx": q_key,
                    "noisy_score": round(s0, 4),
                    "err_detection": round(err_det, 4),
                    "err_glare_hub": round(err_hub, 4),
                    "err_viewpoint": round(err_rot, 4),
                    "err_clock_kin": round(err_clk, 4),
                    "err_algorithm": round(err_alg, 4),
                    "bh_amp_factor": round(err_hub / 0.02, 2),
                    "latency_ms": 0.0,
                    "speedup_vs_prov_exact": 0.0,
                    "speedup_vs_mcf": 0.0,
                })
            print(
                f"[Exp5a 5-Stage] {en:12s} Q1={waterfall['S0_Full_Noisy'][en]['q1_f1']:.3f} "
                f"(Occ={attribution[en]['q1_f1']['detection_occlusion_err']:+.3f}, "
                f"Hub={attribution[en]['q1_f1']['glare_hub_err']:+.3f} [{attribution[en]['q1_f1']['blackhole_amplification_factor']:.2f}x]) | "
                f"Q2={waterfall['S0_Full_Noisy'][en]['q2_f1']:.3f} "
                f"(Occ={attribution[en]['q2_f1']['detection_occlusion_err']:+.3f}, "
                f"Hub={attribution[en]['q2_f1']['glare_hub_err']:+.3f} [{attribution[en]['q2_f1']['blackhole_amplification_factor']:.2f}x])"
            )

    # --- 5b: True O(|Delta|) Q4 Incremental View Maintenance & Affected-Shard MCF-Sharded Comparison ---
    base_sp = get_benchmark_spec("Portolan-Tiny", seed=42)
    base_sp.sigma_tau = 6.0
    topo = RoadCameraTopology(base_sp.n_cam, seed=42)
    ds_ivm = PortolanSynthesizer(profile, base_sp, topo, device).synthesize(generate_workloads=True)
    cg_ivm = SpatioTemporalCandidateGraph(ds_ivm, device)
    wl_q4 = ds_ivm["workloads"]["Q4"]

    bm_eng = EagerHungarianEngine(sim_threshold=float(tuned_params["Global-BM"]))
    mcf_eng = MinCostNetFlowEngine(match_reward=float(tuned_params["MCF-Tracker"]), sharded=False)
    mcf_sh_eng = get_mcf_sharded_engine(tuned_params)
    _, _, _ = mcf_sh_eng.recover_trajectories(cg_ivm)
    pv_eng = ProvenanceCascadingDBEngine(base_threshold=float(tuned_params["Prov-IVM"]))
    _, _, _ = pv_eng.recover_trajectories(cg_ivm)

    unlearn_set = set(wl_q4["unlearn_vehicle_ids"])
    rem_unlearn = {i for i, v in enumerate(ds_ivm["snap_veh"]) if int(v) in unlearn_set}

    drop_cams = set(wl_q4["whatif_dropout_cameras"])
    rem_camdrop = {i for i, c in enumerate(ds_ivm["snap_cam"]) if int(c) in drop_cams}

    pv_noproj = ProvenanceCascadingDBEngine(
        use_active_cert=False, base_threshold=float(tuned_params["Prov-IVM"])
    )
    _, _, _ = pv_noproj.recover_trajectories(cg_ivm)
    budget_1pct = max(1, int(round(cg_ivm.n_snaps * 0.01)))
    active_suspects = [
        int(x) for x in np.argsort(-cg_ivm.hub_suspicion)
        if int(pv_noproj.prev_ptr[int(x)]) >= 0
    ]
    rem_quar = set(active_suspects[:budget_1pct])

    q4_del_specs = [
        ("Q4a_Hub_Quarantine", rem_quar, pv_noproj),
        ("Q4b_Privacy_Unlearning", rem_unlearn, pv_eng),
        ("Q4c_WhatIf_Cam_Dropout", rem_camdrop, pv_eng),
    ]

    ivm_results = {
        "_stream_info": {
            "n_snapshots": int(cg_ivm.n_snaps),
            "cgraph_build_ms": float(cg_ivm.build_time_sec * 1000.0),
            "cpg_bytes_per_tuple": float(pv_eng.get_cpg_bytes_per_tuple()),
            "recomp_includes_cgraph_rebuild": False,
        }
    }
    for tx_name, rem_nodes, pv_instance in q4_del_specs:
        active_m = np.ones(cg_ivm.n_snaps, dtype=bool)
        for r_i in rem_nodes:
            active_m[r_i] = False

        ivm_times, prop_times = [], []
        touched_cnt = 0
        next_ivm = None
        for _ in range(3):
            next_ivm, touched_cnt, t_ivm = pv_instance.execute_bounded_ivm_repair(
                cg_ivm, rem_nodes, materialize_view=True
            )
            ivm_times.append(t_ivm * 1000.0)
            prop_times.append(float(pv_instance.last_ivm_propagation_ms))
        ivm_ms = float(np.mean(ivm_times))
        prop_ms = float(np.mean(prop_times))
        tr_pv_ivm = pv_instance.last_repaired_trajs

        _, _, t_bm = bm_eng.recover_trajectories(cg_ivm, active_mask=active_m)
        tr_mcf, _, t_mcf = mcf_eng.recover_trajectories(cg_ivm, active_mask=active_m)
        _, _, t_msh_full = get_mcf_sharded_engine(tuned_params).recover_trajectories(cg_ivm, active_mask=active_m)
        tr_msh_aff, _, t_msh_aff = mcf_sh_eng.recompute_affected_shards(cg_ivm, rem_nodes, active_mask=active_m)
        bm_ms = t_bm * 1000.0
        mcf_ms = t_mcf * 1000.0
        msh_full_ms = t_msh_full * 1000.0
        msh_aff_ms = t_msh_aff * 1000.0

        pv_exact = ProvenanceCascadingDBEngine(
            use_active_cert=pv_instance.use_active_cert,
            base_threshold=float(tuned_params["Prov-IVM"]),
        )
        tr_pv_exact, next_exact, t_pv_ex = pv_exact.recover_trajectories(cg_ivm, active_mask=active_m)
        pv_exact_ms = t_pv_ex * 1000.0

        idf1_ivm = compute_mtmc_metrics(tr_pv_ivm, ds_ivm["snap_veh"], active_mask=active_m)["idf1"]
        idf1_exact = compute_mtmc_metrics(tr_pv_exact, ds_ivm["snap_veh"], active_mask=active_m)["idf1"]
        idf1_mcf = compute_mtmc_metrics(tr_mcf, ds_ivm["snap_veh"], active_mask=active_m)["idf1"]
        idf1_msh_aff = compute_mtmc_metrics(tr_msh_aff, ds_ivm["snap_veh"], active_mask=active_m)["idf1"]
        idf1_ret_pct = 100.0 * idf1_ivm / max(1e-6, idf1_exact)
        link_agr_pct = float(100.0 * np.mean(next_ivm[active_m] == next_exact[active_m]))
        nbr_agr_pct = _neighborhood_agreement_pct(next_ivm, next_exact, pv_instance.last_neighborhood_nodes, active_m)

        sp_exact = pv_exact_ms / max(1e-3, ivm_ms)
        sp_bm = bm_ms / max(1e-3, ivm_ms)
        sp_mcf = mcf_ms / max(1e-3, ivm_ms)
        sp_msh_aff = msh_aff_ms / max(1e-3, ivm_ms)
        cpg_mem_mb = pv_instance.get_cpg_memory_mb()

        ivm_results[tx_name] = {
            "delta_snapshots": len(rem_nodes),
            "touched_snapshots": int(touched_cnt),
            "touched_fraction_pct": float(100.0 * touched_cnt / cg_ivm.n_snaps),
            "neighborhood_snapshots": int(len(pv_instance.last_neighborhood_nodes)),
            "max_wave_depth": int(pv_instance.last_max_wave_depth),
            "cpg_memory_mb": float(cpg_mem_mb),
            "cpg_bytes_per_tuple": float(pv_instance.get_cpg_bytes_per_tuple()),
            "ivm_propagation_ms": prop_ms,
            "prov_ivm_ms": ivm_ms,
            "mcf_sharded_affected_ms": msh_aff_ms,
            "mcf_sharded_recomp_ms": msh_full_ms,
            "prov_exact_recomp_ms": pv_exact_ms,
            "global_bm_recomp_ms": bm_ms,
            "mcf_recomp_ms": mcf_ms,
            "speedup_vs_mcf_sharded_affected": float(sp_msh_aff),
            "speedup_vs_prov_exact": float(sp_exact),
            "speedup_vs_global_bm": float(sp_bm),
            "speedup_vs_mcf": float(sp_mcf),
            "prov_ivm_idf1": float(idf1_ivm),
            "prov_exact_idf1": float(idf1_exact),
            "mcf_sharded_affected_idf1": float(idf1_msh_aff),
            "mcf_recomp_idf1": float(idf1_mcf),
            "neighborhood_link_agreement_pct": nbr_agr_pct,
            "link_agreement_pct": link_agr_pct,
            "idf1_retention_pct": float(idf1_ret_pct),
            "accuracy_parity_pct": float(idf1_ret_pct),
        }
        print(
            f"[Exp5b] {tx_name:23s} (|Delta|={len(rem_nodes):4d}, touched={touched_cnt:4d}): "
            f"Prov-IVM={ivm_ms:5.2f}ms (prop={prop_ms:4.2f}ms) vs MCF-Sharded-Aff={msh_aff_ms:6.1f}ms ({sp_msh_aff:5.1f}x) "
            f"vs Prov-Full={pv_exact_ms:6.1f}ms ({sp_exact:5.1f}x) [nbr_agr={nbr_agr_pct:.2f}%, global_agr={link_agr_pct:.2f}%]"
        )
        csv_rows.append({
            "section": "5b_q4_ivm",
            "engine": "Prov-IVM",
            "query_or_tx": tx_name,
            "noisy_score": round(idf1_ivm, 4),
            "err_detection": len(rem_nodes),
            "err_glare_hub": touched_cnt,
            "err_viewpoint": round(nbr_agr_pct, 2),
            "err_clock_kin": round(pv_exact_ms, 2),
            "err_algorithm": round(mcf_ms, 2),
            "bh_amp_factor": round(sp_msh_aff, 1),
            "latency_ms": round(ivm_ms, 3),
            "speedup_vs_prov_exact": round(sp_exact, 1),
            "speedup_vs_mcf": round(sp_mcf, 1),
        })

    # Authentic Q4d Retroactive Clock Calibration via kinematic edge re-weighting (zero snapshot deletion)
    cal_cams = set(wl_q4["clock_calibration_cams"])
    cal_cg, cal_snaps, aff_sources = cg_ivm.clone_with_calibrated_clocks(cal_cams)
    ivm_clk_times, prop_clk_times = [], []
    touched_clk = 0
    next_clk_ivm = None
    for _ in range(3):
        next_clk_ivm, touched_clk, t_c_ivm = pv_eng.execute_clock_calibration_ivm(
            cal_cg, aff_sources, materialize_view=True
        )
        ivm_clk_times.append(t_c_ivm * 1000.0)
        prop_clk_times.append(float(pv_eng.last_ivm_propagation_ms))
    ivm_clk_ms = float(np.mean(ivm_clk_times))
    prop_clk_ms = float(np.mean(prop_clk_times))
    tr_pv_clk_ivm = pv_eng.last_repaired_trajs

    _, _, t_bm_cal = bm_eng.recover_trajectories(cal_cg)
    tr_mcf_cal, _, t_mcf_cal = mcf_eng.recover_trajectories(cal_cg)
    _, _, t_msh_full_cal = get_mcf_sharded_engine(tuned_params).recover_trajectories(cal_cg)
    tr_msh_aff_cal, _, t_msh_aff_cal = mcf_sh_eng.recompute_affected_shards(cal_cg, set(cal_snaps))
    bm_cal_ms = t_bm_cal * 1000.0
    mcf_cal_ms = t_mcf_cal * 1000.0
    msh_full_cal_ms = t_msh_full_cal * 1000.0
    msh_aff_cal_ms = t_msh_aff_cal * 1000.0

    pv_exact_cal = ProvenanceCascadingDBEngine(
        use_active_cert=True, base_threshold=float(tuned_params["Prov-IVM"])
    )
    tr_pv_exact_cal, next_clk_exact, t_pv_ex_cal = pv_exact_cal.recover_trajectories(cal_cg)
    pv_exact_cal_ms = t_pv_ex_cal * 1000.0

    idf1_clk_ivm = compute_mtmc_metrics(tr_pv_clk_ivm, ds_ivm["snap_veh"])["idf1"]
    idf1_clk_exact = compute_mtmc_metrics(tr_pv_exact_cal, ds_ivm["snap_veh"])["idf1"]
    idf1_clk_mcf = compute_mtmc_metrics(tr_mcf_cal, ds_ivm["snap_veh"])["idf1"]
    idf1_clk_msh_aff = compute_mtmc_metrics(tr_msh_aff_cal, ds_ivm["snap_veh"])["idf1"]
    idf1_ret_clk = 100.0 * idf1_clk_ivm / max(1e-6, idf1_clk_exact)
    link_agr_clk = float(100.0 * np.mean(next_clk_ivm == next_clk_exact))
    nbr_agr_clk = _neighborhood_agreement_pct(next_clk_ivm, next_clk_exact, pv_eng.last_neighborhood_nodes, None)
    sp_ex_clk = pv_exact_cal_ms / max(1e-3, ivm_clk_ms)
    sp_bm_clk = bm_cal_ms / max(1e-3, ivm_clk_ms)
    sp_mcf_clk = mcf_cal_ms / max(1e-3, ivm_clk_ms)
    sp_msh_clk = msh_aff_cal_ms / max(1e-3, ivm_clk_ms)

    ivm_results["Q4d_Clock_Calibration"] = {
        "delta_snapshots": len(cal_snaps),
        "affected_cpg_sources": int(getattr(pv_eng, "last_clock_cpg_sources", 0)),
        "affected_edge_sources": len(aff_sources),
        "touched_snapshots": int(touched_clk),
        "touched_fraction_pct": float(100.0 * touched_clk / cg_ivm.n_snaps),
        "neighborhood_snapshots": int(len(pv_eng.last_neighborhood_nodes)),
        "max_wave_depth": int(pv_eng.last_max_wave_depth),
        "cpg_memory_mb": float(pv_eng.get_cpg_memory_mb()),
        "cpg_bytes_per_tuple": float(pv_eng.get_cpg_bytes_per_tuple()),
        "ivm_propagation_ms": prop_clk_ms,
        "prov_ivm_ms": ivm_clk_ms,
        "mcf_sharded_affected_ms": msh_aff_cal_ms,
        "mcf_sharded_recomp_ms": msh_full_cal_ms,
        "prov_exact_recomp_ms": pv_exact_cal_ms,
        "global_bm_recomp_ms": bm_cal_ms,
        "mcf_recomp_ms": mcf_cal_ms,
        "speedup_vs_mcf_sharded_affected": float(sp_msh_clk),
        "speedup_vs_prov_exact": float(sp_ex_clk),
        "speedup_vs_global_bm": float(sp_bm_clk),
        "speedup_vs_mcf": float(sp_mcf_clk),
        "prov_ivm_idf1": float(idf1_clk_ivm),
        "prov_exact_idf1": float(idf1_clk_exact),
        "mcf_sharded_affected_idf1": float(idf1_clk_msh_aff),
        "mcf_recomp_idf1": float(idf1_clk_mcf),
        "neighborhood_link_agreement_pct": nbr_agr_clk,
        "link_agreement_pct": link_agr_clk,
        "idf1_retention_pct": float(idf1_ret_clk),
        "accuracy_parity_pct": float(idf1_ret_clk),
    }
    print(
        f"[Exp5b] {'Q4d_Clock_Calibration':23s} (|Delta|={len(cal_snaps):4d}, touched={touched_clk:4d}): "
        f"Prov-IVM={ivm_clk_ms:5.2f}ms (prop={prop_clk_ms:4.2f}ms) vs MCF-Sharded-Aff={msh_aff_cal_ms:6.1f}ms ({sp_msh_clk:5.1f}x) "
        f"vs Prov-Full={pv_exact_cal_ms:6.1f}ms ({sp_ex_clk:5.1f}x) [nbr_agr={nbr_agr_clk:.2f}%, global_agr={link_agr_clk:.2f}%]"
    )
    csv_rows.append({
        "section": "5b_q4_ivm",
        "engine": "Prov-IVM",
        "query_or_tx": "Q4d_Clock_Calibration",
        "noisy_score": round(idf1_clk_ivm, 4),
        "err_detection": len(cal_snaps),
        "err_glare_hub": touched_clk,
        "err_viewpoint": round(nbr_agr_clk, 2),
        "err_clock_kin": round(pv_exact_cal_ms, 2),
        "err_algorithm": round(mcf_cal_ms, 2),
        "bh_amp_factor": round(sp_msh_clk, 1),
        "latency_ms": round(ivm_clk_ms, 3),
        "speedup_vs_prov_exact": round(sp_ex_clk, 1),
        "speedup_vs_mcf": round(sp_mcf_clk, 1),
    })

    if csv_rows:
        with open(RESULTS_DIR / "table3_error_attribution_ivm.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    return {
        "exp5a_params": {"rho_occ": 0.18, "alpha_hub": 0.02, "sigma_tau": 4.5, "q2_k": 5},
        "waterfall": waterfall,
        "stage_attribution": attribution,
        "q4_ivm": ivm_results,
    }


def run_exp6_ablation_and_sensitivity(device: torch.device, tuned_params: Dict[str, float]) -> Dict:
    """Exp 6: Out-of-Sample Synthesizer Module Ablation against Held-Out Real-CityFlow & Val Sensitivity."""
    print("\n================ [Exp 6] Out-of-Sample Synthesizer Module Ablation (vs. Held-Out Real-CityFlow) ================")
    seeds = [42, 43, 44]
    engine_names = [e.name for e in get_all_db_engines(tuned_params)]

    variants = [
        ("Portolan (Full)", {"use_vmf": True, "use_blackhole": True, "use_congestion_wave": True, "use_hard_negatives": True}),
        ("w/o vMF & SO(d) Manifold", {"use_vmf": False, "use_blackhole": True, "use_congestion_wave": True, "use_hard_negatives": True}),
        ("w/o Black-Hole Injection", {"use_vmf": True, "use_blackhole": False, "use_congestion_wave": True, "use_hard_negatives": True}),
        ("w/o Congestion Wave", {"use_vmf": True, "use_blackhole": True, "use_congestion_wave": False, "use_hard_negatives": True}),
        ("w/o Hard-Negative Cones", {"use_vmf": True, "use_blackhole": True, "use_congestion_wave": False if False else True, "use_hard_negatives": False}),
    ]

    cal_ds = load_real_benchmark_stream("CityFlow-Seed", fold_idx=1, split="cal")
    cal_profile, _ = calibrate_seed_profile_from_stream(cal_ds, name="CityFlow-Seed")
    real_ds = load_real_benchmark_stream(
        "CityFlow-Seed", fold_idx=1, split="eval", seed_profile_override=cal_profile
    )
    d_real = extract_stream_distributions(real_ds, device)
    cg_real = SpatioTemporalCandidateGraph(real_ds, device)
    real_scores = {}
    for eng in get_all_db_engines(tuned_params):
        tr, _, ls = eng.recover_trajectories(cg_real)
        mt = compute_mtmc_metrics(tr, real_ds["snap_veh"])
        mt["latency_ms"] = ls * 1000.0
        real_scores[eng.name] = mt

    m18_real = np.array(
        [float(compute_mtmc_metrics(ec.recover_trajectories(cg_real)[0], real_ds["snap_veh"])["idf1"]) for _, ec in build_m18_engine_configs(tuned_params)],
        dtype=np.float64,
    )

    ablation_acc = {
        v_name: {
            "margin_W1": [],
            "hub_W1": [],
            "tt_W1": [],
            "mean_W1": [],
            "mean_JSD": [],
            "zipf": [],
            "spearman_rho": [],
            "idf1_spearman_rho": [],
            "m18_spearman_rho": [],
            "idf1_mae": [],
        }
        for v_name, _ in variants
    }

    topo = real_ds["topology"]
    for sd in seeds:
        spec = copy.deepcopy(real_ds["spec"])
        spec.seed = sd

        for v_name, v_flags in variants:
            ds_v = PortolanSynthesizer(cal_profile, spec, topo, device, **v_flags).synthesize(generate_workloads=False)
            d_v = extract_stream_distributions(ds_v, device)
            fid_v = compute_fidelity_metrics(d_real, d_v)

            cg_v = SpatioTemporalCandidateGraph(ds_v, device)
            v_scores = {}
            for eng in get_all_db_engines(tuned_params):
                tr, _, ls = eng.recover_trajectories(cg_v)
                mt = compute_mtmc_metrics(tr, ds_v["snap_veh"])
                mt["latency_ms"] = ls * 1000.0
                v_scores[eng.name] = mt

            m18_v = np.array(
                [float(compute_mtmc_metrics(ec.recover_trajectories(cg_v)[0], ds_v["snap_veh"])["idf1"]) for _, ec in build_m18_engine_configs(tuned_params)],
                dtype=np.float64,
            )
            m18_st = compute_multiconfig_ranking_stats(m18_real, m18_v, n_perm=20000, seed=sd)
            rc_v = compute_ranking_consistency(real_scores, v_scores, engine_names)

            ablation_acc[v_name]["margin_W1"].append(fid_v["cosine_margin"]["W1"])
            ablation_acc[v_name]["hub_W1"].append(fid_v["hubness_tail"]["W1"])
            ablation_acc[v_name]["tt_W1"].append(fid_v["travel_time"]["W1"])
            ablation_acc[v_name]["mean_W1"].append(fid_v["mean_W1"])
            ablation_acc[v_name]["mean_JSD"].append(fid_v["mean_JSD"])
            ablation_acc[v_name]["zipf"].append(fid_v["hubness_tail"]["zipf_gamma"])
            ablation_acc[v_name]["spearman_rho"].append(rc_v["idf1_spearman_rho"])
            ablation_acc[v_name]["idf1_spearman_rho"].append(rc_v["idf1_spearman_rho"])
            ablation_acc[v_name]["m18_spearman_rho"].append(m18_st["spearman_rho"])
            ablation_acc[v_name]["idf1_mae"].append(rc_v["idf1_mae"])

    ablation_summary = {}
    csv_rows = []
    for v_name, _ in variants:
        acc = ablation_acc[v_name]
        ablation_summary[v_name] = {k: _mean_std(vals) for k, vals in acc.items()}
        print(
            f"[Exp6 vs Held-Out Real-CityFlow] {v_name:25s}: mean_W1={ablation_summary[v_name]['mean_W1']['mean']:.4f}, "
            f"hub_W1={ablation_summary[v_name]['hub_W1']['mean']:.4f}, "
            f"tt_W1={ablation_summary[v_name]['tt_W1']['mean']:.4f}, "
            f"IDF1_rho={ablation_summary[v_name]['spearman_rho']['mean']:.3f} (M18={ablation_summary[v_name]['m18_spearman_rho']['mean']:.3f}), "
            f"IDF1_MAE={ablation_summary[v_name]['idf1_mae']['mean']:.4f}"
        )
        csv_rows.append({
            "variant": v_name,
            "margin_W1": round(ablation_summary[v_name]["margin_W1"]["mean"], 4),
            "hub_W1": round(ablation_summary[v_name]["hub_W1"]["mean"], 4),
            "tt_W1": round(ablation_summary[v_name]["tt_W1"]["mean"], 4),
            "mean_W1": round(ablation_summary[v_name]["mean_W1"]["mean"], 4),
            "mean_JSD": round(ablation_summary[v_name]["mean_JSD"]["mean"], 4),
            "zipf_gamma": round(ablation_summary[v_name]["zipf"]["mean"], 2),
            "idf1_spearman_rho": round(ablation_summary[v_name]["spearman_rho"]["mean"], 3),
            "m18_spearman_rho": round(ablation_summary[v_name]["m18_spearman_rho"]["mean"], 3),
            "idf1_mae": round(ablation_summary[v_name]["idf1_mae"]["mean"], 4),
        })

    # Declarative Knob Sensitivity on Held-Out Validation Cohort (val_seed=101)
    val_seed = 101
    base_spec = get_benchmark_spec("Portolan-Tiny", seed=val_seed)
    topo_val = RoadCameraTopology(base_spec.n_cam, seed=val_seed)
    occ_grid = [0.00, 0.10, 0.20, 0.30, 0.40]
    clk_grid = [0.0, 2.5, 7.5, 15.0, 30.0]
    sens_engines = ["Prov-IVM", "MCF-Sharded", "MCF-Tracker", "Global-BM", "Prob-Path"]

    occ_sens = {"validation_seed": val_seed, "rho_occ": occ_grid, "idf1": {en: [] for en in sens_engines}}
    for ro in occ_grid:
        sp_o = copy.deepcopy(base_spec)
        sp_o.rho_occ = ro
        ds_o = PortolanSynthesizer(cal_profile, sp_o, topo_val, device).synthesize(generate_workloads=False)
        cg_o = SpatioTemporalCandidateGraph(ds_o, device)
        e_map = {e.name: e for e in get_all_db_engines(tuned_params)}
        e_map["MCF-Sharded"] = get_mcf_sharded_engine(tuned_params)
        for en in sens_engines:
            tr, _, _ = e_map[en].recover_trajectories(cg_o)
            occ_sens["idf1"][en].append(float(compute_mtmc_metrics(tr, ds_o["snap_veh"])["idf1"]))

    clk_sens = {"validation_seed": val_seed, "sigma_tau": clk_grid, "idf1": {en: [] for en in sens_engines}}
    for st in clk_grid:
        sp_c = copy.deepcopy(base_spec)
        sp_c.sigma_tau = st
        ds_c = PortolanSynthesizer(cal_profile, sp_c, topo_val, device).synthesize(generate_workloads=False)
        cg_c = SpatioTemporalCandidateGraph(ds_c, device)
        e_map = {e.name: e for e in get_all_db_engines(tuned_params)}
        e_map["MCF-Sharded"] = get_mcf_sharded_engine(tuned_params)
        for en in sens_engines:
            tr, _, _ = e_map[en].recover_trajectories(cg_c)
            clk_sens["idf1"][en].append(float(compute_mtmc_metrics(tr, ds_c["snap_veh"])["idf1"]))

    with open(RESULTS_DIR / "table4_ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    return {
        "ablation": ablation_summary,
        "sensitivity_occlusion": occ_sens,
        "sensitivity_clock_drift": clk_sens,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset: tune,exp1,exp23,exp3b,exp3c,exp4,exp5,exp5b,exp6",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto' (default, GPU with max free VRAM or CPU), 'cpu', 'cuda', or 'cuda:N'",
    )
    parser.add_argument(
        "--sla-timeout",
        type=int,
        default=600,
        help="Per-solver serial wall-clock timeout in seconds for Exp 3c (default: 600s)",
    )
    args = parser.parse_args()
    only_set = {s.strip() for s in args.only.split(",") if s.strip()}

    t_global = time.perf_counter()
    device = select_best_device() if args.device == "auto" else torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Running Portolan EAB Benchmark Suite on {device} ({gpu_name})")

    metrics_path = RESULTS_DIR / "metrics.json"
    if only_set and metrics_path.exists():
        with open(metrics_path, "r") as f:
            suite_metrics = json.load(f)
    else:
        suite_metrics = {}

    def _ckpt() -> None:
        with open(metrics_path, "w") as _f:
            json.dump(suite_metrics, _f, indent=2)
        sys.stdout.flush()

    if not only_set or "tune" in only_set or "validation_tuning" not in suite_metrics:
        suite_metrics["validation_tuning"] = tune_engines_equal_budget(device, val_seed=101)
        _ckpt()
    cf_tuned = suite_metrics["validation_tuning"]["profiles"]["CityFlow-Seed"]["selected_thresholds"]

    if not only_set or "exp1" in only_set:
        suite_metrics["exp1_scalability"] = run_exp1_scalability(device)
        _ckpt()
    if not only_set or "exp23" in only_set:
        exp2_3 = run_exp2_and_exp3(device, suite_metrics["validation_tuning"])
        suite_metrics["exp2_fidelity"] = exp2_3["exp2_fidelity"]
        suite_metrics["exp3_ranking_consistency"] = exp2_3["exp3_ranking"]
        _ckpt()
    if not only_set or "exp3b" in only_set:
        suite_metrics["exp3b_district_scale"] = run_exp3b_district_scale(device, cf_tuned)
        _ckpt()
    if not only_set or "exp3c" in only_set or "exp3c_med" in only_set or "exp3c_city" in only_set:
        t_filt = None
        if "exp3c_med" in only_set and "exp3c_city" not in only_set and "exp3c" not in only_set:
            t_filt = ["Portolan-Medium"]
        elif "exp3c_city" in only_set and "exp3c_med" not in only_set and "exp3c" not in only_set:
            t_filt = ["Portolan-CityScale"]

        def _ckpt_3c(partial_3c: Dict) -> None:
            suite_metrics["exp3c_metropolitan_scaling"] = partial_3c
            _ckpt()

        suite_metrics["exp3c_metropolitan_scaling"] = run_exp3c_metropolitan_scaling(
            device,
            cf_tuned,
            sla_timeout_sec=int(args.sla_timeout),
            tiers_filter=t_filt,
            existing_3c=suite_metrics.get("exp3c_metropolitan_scaling"),
            tier_ckpt_hook=_ckpt_3c,
        )
        _ckpt()
    if not only_set or "exp4" in only_set:
        suite_metrics["exp4_stress_testing"] = run_exp4_stress_testing(device, cf_tuned)
        _ckpt()
    if not only_set or "exp5" in only_set:
        suite_metrics["exp5_error_attribution_and_ivm"] = run_exp5_error_attribution_and_ivm(device, cf_tuned)
        _ckpt()
    elif "exp5b" in only_set:
        suite_metrics["exp5_error_attribution_and_ivm"] = run_exp5_error_attribution_and_ivm(
            device, cf_tuned, existing_5a=suite_metrics.get("exp5_error_attribution_and_ivm")
        )
        _ckpt()
    if not only_set or "exp6" in only_set:
        suite_metrics["exp6_ablation_and_sensitivity"] = run_exp6_ablation_and_sensitivity(device, cf_tuned)
        _ckpt()

    total_runtime_sec = time.perf_counter() - t_global
    suite_metrics["metadata"] = {
        "benchmark": "Portolan ICDE 2027 EAB Suite",
        "device": str(device),
        "gpu_name": gpu_name,
        "pytorch_version": torch.__version__,
        "last_run_sec": round(total_runtime_sec, 2),
    }

    with open(metrics_path, "w") as f:
        json.dump(suite_metrics, f, indent=2)

    print(f"\nBENCHMARK EXECUTION COMPLETED SUCCESSFULLY in {total_runtime_sec:.1f}s!")
    print(f"Saved suite metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
