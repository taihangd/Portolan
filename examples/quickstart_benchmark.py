"""
Portolan Standalone Quickstart Benchmark (~2-3 seconds on CPU or GPU).

Demonstrates:
  1. Synthesizing a realistic cross-camera trajectory stream with hyperspherical vMF embeddings,
     macroscopic congestion waves, and controllable Embedding Black-Hole (hubness) defects.
  2. Generating multi-modal declarative query workloads (Q1, Q2, Q3, Q4) with exact ground truth
     and exporting the 4 standardized Output-Layer tables.
  3. Profiling custom ReID feature tensors via SeedProfile.from_feature_tensors().
  4. Benchmarking all 6 reference trajectory & database engines (IDF1, MOTA, Q1, Q2, Q3)
     plus Bounded Cascading Incremental View Maintenance (Q4 privacy unlearning).
"""

import argparse
import sys
from pathlib import Path
import torch

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from portolan import (
    RoadCameraTopology,
    SeedProfile,
    SpatioTemporalCandidateGraph,
    PortolanSynthesizer,
    WorkloadSpec,
    compute_mtmc_metrics,
    evaluate_q1_from_trajs,
    evaluate_q2_eager_from_trajs,
    evaluate_q3_from_trajs,
    export_standardized_tables,
    get_seed_profile,
    select_best_device,
)
from global_bm import EagerHungarianEngine
from mcf_tracker import MinCostNetFlowEngine
from online_clust import StreamGreedyClustEngine
from vec_filter import VecDBFilteredANNEngine
from prob_path import TopoMarkovMAPEngine
from prov_ivm import ProvenanceCascadingDBEngine


def get_all_db_engines():
    return [
        EagerHungarianEngine(),
        MinCostNetFlowEngine(),
        StreamGreedyClustEngine(),
        VecDBFilteredANNEngine(),
        TopoMarkovMAPEngine(),
        ProvenanceCascadingDBEngine(),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Portolan Standalone Quickstart Benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto' (default), 'cpu', 'cuda', or 'cuda:N'",
    )
    args = parser.parse_args()
    device = select_best_device() if args.device == "auto" else torch.device(args.device)
    print(f"[Portolan Quickstart] Selected compute device: {device}")

    # 1. Configure a fast 30-camera, 800-vehicle urban district benchmark
    profile = get_seed_profile("CityFlow-Seed")
    spec = WorkloadSpec(
        scale_name="Quickstart-District",
        n_cam=30,
        n_veh=800,
        target_snapshots=8500,
        embed_dim=256,
        rho_occ=0.15,
        sigma_tau=2.5,
        alpha_hub=0.015,
        q1_num_queries=50,
        q2_min_hops=4,
        seed=42,
    )
    topology = RoadCameraTopology(spec.n_cam, seed=spec.seed)

    synth = PortolanSynthesizer(profile, spec, topology, device)
    ds = synth.synthesize(generate_workloads=True)
    print(
        f"[1/4] Synthesized {ds['total_snapshots']:,} snapshots across {spec.n_cam} cameras "
        f"in {ds['elapsed_sec']*1000.0:.1f} ms "
        f"({ds['total_snapshots']/max(1e-6, ds['elapsed_sec']):,.0f} snaps/s)."
    )

    # 2. Custom tensor profiling demonstration
    custom_profile = SeedProfile.from_feature_tensors(
        ds["embeddings"][:1500],
        ds["snap_veh"][:1500],
        name="Profiled-From-Stream",
    )
    print(
        f"[2/4] Custom Tensor Calibration -> intra_cos={custom_profile.intra_cos_mean:.3f}, "
        f"inter_cos={custom_profile.inter_cos_mean:.3f}, kappa_est={custom_profile.vmf_kappa_mean:.1f}"
    )

    # 3. Benchmark all 6 reference trajectory & database paradigms
    cgraph = SpatioTemporalCandidateGraph(ds, device)
    wl = ds["workloads"]
    k_hops = wl["Q2"]["default_k"]
    gt_q2_pairs = wl["Q2"]["gt_pairs_by_k"][k_hops]

    print(f"\n[3/4] Benchmarking 6 Reference Engines (Q2 hops k={k_hops}, |GT pairs|={len(gt_q2_pairs)}):")
    print(f"{'Engine':<16} {'IDF1':>7} {'MOTA':>7} {'Q1-F1':>7} {'Q2-F1':>7} {'Q3-Acc':>8} {'Time(ms)':>10}")
    print("-" * 68)

    prov_engine = None
    mcf_latency_ms = 0.0
    for eng in get_all_db_engines():
        trajs, _, elapsed = eng.recover_trajectories(cgraph)
        mtmc = compute_mtmc_metrics(trajs, ds["snap_veh"])
        q1_f1 = evaluate_q1_from_trajs(trajs, cgraph.n_snaps, wl["Q1"])
        if isinstance(eng, ProvenanceCascadingDBEngine):
            prov_engine = eng
            q2_res, q2_sec = eng.execute_coupled_q2_join(
                cgraph, k_hops, wl["Q2"]["delta_t_sec"], gt_q2_pairs
            )
            elapsed += q2_sec
        else:
            q2_res = evaluate_q2_eager_from_trajs(
                trajs, ds["snap_cam"], ds["snap_time"], ds["snap_veh"],
                k_hops, wl["Q2"]["delta_t_sec"], gt_q2_pairs,
            )
        q3_res = evaluate_q3_from_trajs(trajs, ds["snap_cam"], wl["Q3"])
        lat_ms = elapsed * 1000.0
        if isinstance(eng, MinCostNetFlowEngine):
            mcf_latency_ms = lat_ms
        print(
            f"{eng.name:<16} {mtmc['idf1']:>7.3f} {mtmc['mota']:>7.3f} "
            f"{q1_f1:>7.3f} {q2_res['f1']:>7.3f} {q3_res['q3_score']:>8.3f} {lat_ms:>10.1f}"
        )

    # 4. Export standardized Output-Layer tables & run Q4 Bounded Cascading IVM Repair
    assert prov_engine is not None
    tables = export_standardized_tables(ds, cpg_out=prov_engine.cpg_out)
    unlearn_vids = set(wl["Q4"]["unlearn_vehicle_ids"][:5])
    removed_nodes = {
        sid for sid, vid in enumerate(ds["snap_veh"].tolist()) if int(vid) in unlearn_vids
    }
    _, touched_nodes, ivm_sec = prov_engine.execute_bounded_ivm_repair(cgraph, removed_nodes)
    ivm_ms = ivm_sec * 1000.0
    speedup = mcf_latency_ms / max(1e-4, ivm_ms)
    print(
        f"\n[4/4] Exported {len(tables)} standardized tables "
        f"(CPG edges={len(tables['Competitive_Provenance_Graph']['top_k_edges']):,}).\n"
        f"      Q4 Privacy Unlearning ({len(unlearn_vids)} vehicles, {len(removed_nodes)} snapshots): "
        f"Prov-IVM touched {touched_nodes} nodes in {ivm_ms:.2f} ms "
        f"({speedup:.1f}x faster than full MCF recomputation)."
    )


if __name__ == "__main__":
    main()
