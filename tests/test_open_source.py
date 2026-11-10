"""Standalone unit tests for the Portolan open-source synthesizer and six-engine testbed."""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from portolan import (
    RoadCameraTopology,
    SeedProfile,
    SpatioTemporalCandidateGraph,
    PortolanSynthesizer,
    WorkloadSpec,
    compute_fidelity_metrics,
    compute_mtmc_metrics,
    evaluate_q1_from_trajs,
    evaluate_q2_eager_from_trajs,
    evaluate_q2_track_pairs,
    evaluate_q3_from_trajs,
    export_standardized_tables,
    extract_stream_distributions,
    get_benchmark_spec,
    get_seed_profile,
    load_real_benchmark_stream,
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


def get_all_db_engines():
    return [
        EagerHungarianEngine(),
        MinCostNetFlowEngine(),
        StreamGreedyClustEngine(),
        VecDBFilteredANNEngine(),
        TopoMarkovMAPEngine(),
        ProvenanceCascadingDBEngine(),
    ]


def test_yaml_configs_and_declarative_loading():
    for fname in ("cityflow_benchmark.yaml", "default_4090.yaml"):
        cfg_path = PKG_ROOT / "configs" / fname
        assert cfg_path.exists(), f"Missing config: {cfg_path}"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict) and len(cfg) > 0

    profile, spec = WorkloadSpec.from_yaml(PKG_ROOT / "configs" / "cityflow_benchmark.yaml")
    assert profile.name == "CityFlow-Seed"
    assert spec.n_cam == 40 and spec.n_veh == 2000
    assert spec.follower_pairs == 15 and spec.q2_min_hops == 4

    for scale in ("Portolan-Tiny", "Portolan-Small", "Portolan-Medium", "Portolan-CityScale"):
        sp = get_benchmark_spec(scale)
        assert sp.n_cam >= 40 and sp.n_veh >= 2000


def test_synthesis_baselines():
    device = torch.device("cpu")
    profile = get_seed_profile("CityFlow-Seed")
    spec = WorkloadSpec(
        scale_name="OS-BaselineTest",
        n_cam=16,
        n_veh=80,
        embed_dim=128,
        seed=2027,
    )
    topo = RoadCameraTopology(spec.n_cam, seed=2027)

    ngw_ds = NormalizedGaussianWalkSynthesizer(profile, spec, topo, device).synthesize(generate_workloads=True)
    lgmm_ds = LatentGMMSynthesizer(profile, spec, topo, device).synthesize(generate_workloads=True)
    assert ngw_ds["total_snapshots"] > 0 and lgmm_ds["total_snapshots"] > 0
    assert np.allclose(np.linalg.norm(ngw_ds["embeddings"].numpy(), axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(lgmm_ds["embeddings"].numpy(), axis=1), 1.0, atol=1e-4)

    sim_proxy = ThreeDSimPipelineProxy(device=device, embed_dim=128)
    sim_proxy.profile_hardware_throughput(n_sample_snapshots=2, frames_per_snapshot=1)
    est = sim_proxy.estimate_scale(spec)
    assert est["elapsed_sec"] > 0.0 and est["peak_gpu_mb"] > 0.0


def test_end_to_end_synthesis_tables_and_six_reference_engines():
    device = torch.device("cpu")
    profile = get_seed_profile("CityFlow-Seed")
    spec = WorkloadSpec(
        scale_name="OS-UnitTest",
        n_cam=20,
        n_veh=150,
        embed_dim=128,
        rho_occ=0.12,
        sigma_tau=3.0,
        alpha_hub=0.02,
        sigma_dp=0.04,
        seed=2027,
    )
    topo = RoadCameraTopology(spec.n_cam, seed=2027)
    synth = PortolanSynthesizer(profile, spec, topo, device)
    ds = synth.synthesize(generate_workloads=True)

    norms = np.linalg.norm(ds["embeddings"].numpy(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    cal = SeedProfile.from_feature_tensors(ds["embeddings"][:500], ds["snap_veh"][:500])
    assert 0.15 < cal.intra_cos_mean < 1.0

    cgraph = SpatioTemporalCandidateGraph(ds, device)
    wl = ds["workloads"]
    k_hops = wl["Q2"]["default_k"]
    gt_q2 = wl["Q2"]["gt_pairs_by_k"][k_hops]

    prov = None
    for eng in get_all_db_engines():
        trajs, next_ptr, _ = eng.recover_trajectories(cgraph)
        assert len(next_ptr) == cgraph.n_snaps
        mt = compute_mtmc_metrics(trajs, ds["snap_veh"])
        q1 = evaluate_q1_from_trajs(trajs, cgraph.n_snaps, wl["Q1"])
        q2 = evaluate_q2_eager_from_trajs(
            trajs, ds["snap_cam"], ds["snap_time"], ds["snap_veh"],
            k_hops, wl["Q2"]["delta_t_sec"], gt_q2,
        )
        q3 = evaluate_q3_from_trajs(trajs, ds["snap_cam"], wl["Q3"])
        assert 0.0 <= mt["idf1"] <= 1.0
        assert -5.0 <= mt["mota"] <= 1.0
        assert 0.0 <= q1 <= 1.0
        assert 0.0 <= q2["f1"] <= 1.0
        assert 0.0 <= q3["q3_score"] <= 1.0
        if isinstance(eng, ProvenanceCascadingDBEngine):
            prov = eng

    # Verify time-windowed Global-BM variant as well
    bm_tw = EagerHungarianEngine(window_sec=45.0, overlap_sec=35.0)
    trajs_tw, ptr_tw, _ = bm_tw.recover_trajectories(cgraph)
    assert len(ptr_tw) == cgraph.n_snaps and 0.0 <= compute_mtmc_metrics(trajs_tw, ds["snap_veh"])["idf1"] <= 1.0

    assert prov is not None
    q2_coupled, _ = prov.execute_coupled_q2_join(cgraph, k_hops, wl["Q2"]["delta_t_sec"], gt_q2)
    assert 0.0 <= q2_coupled["f1"] <= 1.0
    assert evaluate_q2_track_pairs([], np.zeros(4, dtype=np.int32), np.zeros(4, dtype=np.int32), 2, set())["f1"] == 0.0

    dp_topo = prov.evaluate_privacy_trellis_recovery(cgraph, wl["Q1"], k_hops=3, topology_free=False)
    dp_free = prov.evaluate_privacy_trellis_recovery(cgraph, wl["Q1"], k_hops=3, topology_free=True)
    assert 0.0 <= dp_topo["reid_accuracy"] <= 1.0 and 0.0 <= dp_free["reid_accuracy"] <= 1.0

    tables = export_standardized_tables(ds, cpg_out=prov.cpg_out)
    assert set(tables["Snapshots_Stream"].keys()) == {"sid", "cam_id", "t_obs", "emb"}
    assert "is_bh" in tables["GroundTruth_Trajectories"]
    assert len(tables["Competitive_Provenance_Graph"]["top_k_edges"]) > 0
    assert len(tables["Competitive_Provenance_Graph"]["src_sid"]) == len(
        tables["Competitive_Provenance_Graph"]["top_k_edges"]
    )

    unlearn_set = set(wl["Q4"]["unlearn_vehicle_ids"][:2])
    removed = {i for i, v in enumerate(ds["snap_veh"].tolist()) if int(v) in unlearn_set}
    _, touched, ivm_sec = prov.execute_bounded_ivm_repair(cgraph, removed)
    assert touched >= len(removed) and ivm_sec > 0.0

    cal_cams = set(wl["Q4"]["clock_calibration_cams"])
    cal_cg, cal_snaps, aff_sources = cgraph.clone_with_calibrated_clocks(cal_cams)
    _, touched_clk, clk_sec = prov.execute_clock_calibration_ivm(cal_cg, aff_sources)
    assert len(cal_snaps) > 0 and touched_clk >= 0 and clk_sec > 0.0

    # Verify bundled genuine Real-CityFlow and Real-VeRi out-of-sample splits (0% cal/eval overlap)
    cal_cf = load_real_benchmark_stream("CityFlow-Seed", fold_idx=1, split="cal")
    real_cf = load_real_benchmark_stream("CityFlow-Seed", fold_idx=1, split="eval")
    cal_vr = load_real_benchmark_stream("VeRi-Seed", fold_idx=1, split="cal")
    real_vr = load_real_benchmark_stream("VeRi-Seed", fold_idx=1, split="eval")
    assert cal_cf["total_snapshots"] == 859 and cal_cf["topology"].n_cam == 9
    assert real_cf["total_snapshots"] == 2178 and real_cf["topology"].n_cam == 31
    assert len(set(cal_cf["raw_veh_ids"].tolist()) & set(real_cf["raw_veh_ids"].tolist())) == 0
    assert cal_vr["total_snapshots"] == 958 and cal_vr["topology"].n_cam == 20
    assert real_vr["total_snapshots"] == 1010 and real_vr["topology"].n_cam == 20
    assert len(set(cal_vr["raw_veh_ids"].tolist()) & set(real_vr["raw_veh_ids"].tolist())) == 0

    # Verify MCF-Sharded paradigm variant
    mcf_sh = MinCostNetFlowEngine(sharded=True)
    tr_sh, ptr_sh, _ = mcf_sh.recover_trajectories(cgraph)
    assert mcf_sh.name == "MCF-Sharded" and len(ptr_sh) == cgraph.n_snaps
    assert 0.0 <= compute_mtmc_metrics(tr_sh, ds["snap_veh"])["idf1"] <= 1.0

    syn_cf = PortolanSynthesizer(real_cf["profile"], real_cf["spec"], real_cf["topology"], device).synthesize(
        generate_workloads=False
    )
    fid = compute_fidelity_metrics(
        extract_stream_distributions(real_cf, device, sample_size=600),
        extract_stream_distributions(syn_cf, device, sample_size=600),
    )
    assert fid["mean_W1"] < 0.30


def test_d512_calibration_and_epsilon_dp_budget():
    import math

    device = torch.device("cpu")
    profile = get_seed_profile("CityFlow-Seed")
    topo = RoadCameraTopology(n_cam=20, seed=55)

    spec_512 = WorkloadSpec(
        scale_name="OS-d512",
        n_cam=20,
        n_veh=140,
        embed_dim=512,
        alpha_hub=0.0,
        seed=55,
    )
    ds_512 = PortolanSynthesizer(profile, spec_512, topo, device).synthesize(generate_workloads=False)
    emb_512 = ds_512["embeddings"].numpy()
    pos_cos = [
        float(emb_512[sids[0]] @ emb_512[sids[1]])
        for sids in ds_512["gt_trajectories"].values()
        if len(sids) >= 2 and ds_512["snap_cam"][sids[0]] != ds_512["snap_cam"][sids[1]]
    ]
    assert abs(float(np.mean(pos_cos)) - profile.intra_cos_mean) < 0.12

    spec_dp = WorkloadSpec(
        scale_name="OS-EpsDP",
        n_cam=16,
        n_veh=100,
        target_snapshots=1200,
        embed_dim=128,
        epsilon_dp=140.0,
        sigma_dp=0.0,
        seed=55,
    )
    ds_dp = PortolanSynthesizer(profile, spec_dp, RoadCameraTopology(16, seed=55), device).synthesize(
        generate_workloads=True
    )
    expected_sigma = (2.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5))) / 140.0
    assert abs(ds_dp["effective_sigma_dp"] - expected_sigma) < 1e-6
    cgraph_dp = SpatioTemporalCandidateGraph(ds_dp, device)
    prov_dp = ProvenanceCascadingDBEngine()
    prov_dp.recover_trajectories(cgraph_dp)
    res_dp = prov_dp.evaluate_privacy_trellis_recovery(cgraph_dp, ds_dp["workloads"]["Q1"], k_hops=3)
    assert 0.0 <= res_dp["reid_accuracy"] <= 1.0

    spec_sig = WorkloadSpec(
        scale_name="OS-SigDP",
        n_cam=16,
        n_veh=100,
        target_snapshots=1200,
        embed_dim=128,
        epsilon_dp=0.0,
        sigma_dp=expected_sigma,
        seed=55,
    )
    ds_sig = PortolanSynthesizer(profile, spec_sig, RoadCameraTopology(16, seed=55), device).synthesize(
        generate_workloads=True
    )
    cgraph_sig = SpatioTemporalCandidateGraph(ds_sig, device)
    prov_sig = ProvenanceCascadingDBEngine()
    prov_sig.recover_trajectories(cgraph_sig)
    res_sig = prov_sig.evaluate_privacy_trellis_recovery(cgraph_sig, ds_sig["workloads"]["Q1"], k_hops=3)
    assert abs(res_dp["reid_accuracy"] - res_sig["reid_accuracy"]) < 1e-9


def test_oracle_waterfall_veri_seed_and_baseline_table_exports():
    device = torch.device("cpu")
    veri_profile = get_seed_profile("VeRi-Seed")
    spec = WorkloadSpec(
        scale_name="OS-Waterfall",
        n_cam=16,
        n_veh=90,
        embed_dim=128,
        rho_occ=0.18,
        alpha_hub=0.04,
        sigma_tau=3.5,
        q1_num_queries=15,
        seed=2027,
    )
    topo = RoadCameraTopology(spec.n_cam, seed=2027)

    # 1. Shared-tape counterfactual waterfall (S0 -> S4) on VeRi-Seed
    wf = PortolanSynthesizer(veri_profile, spec, topo, device).synthesize_oracle_waterfall(
        generate_workloads=True
    )
    assert set(wf.keys()) == {
        "S0_Full_Noisy",
        "S1_Oracle_Detection",
        "S2_Oracle_DeGlare",
        "S3_Oracle_Viewpoint",
        "S4_Oracle_Clock_Kin",
    }
    s0, s1, s2, s3, s4 = (
        wf["S0_Full_Noisy"],
        wf["S1_Oracle_Detection"],
        wf["S2_Oracle_DeGlare"],
        wf["S3_Oracle_Viewpoint"],
        wf["S4_Oracle_Clock_Kin"],
    )
    assert s0["total_snapshots"] < s1["total_snapshots"]
    assert s1["total_snapshots"] == s2["total_snapshots"] == s3["total_snapshots"] == s4["total_snapshots"]
    assert int(np.sum(s1["is_blackhole"])) > 0
    assert int(np.sum(s2["is_blackhole"])) == 0 and int(np.sum(s3["is_blackhole"])) == 0 and int(np.sum(s4["is_blackhole"])) == 0

    # 2. Standardized Output-Layer table export across NGW, L-GMM, and Real-VeRi (zero GT leak)
    ngw_ds = NormalizedGaussianWalkSynthesizer(veri_profile, spec, topo, device).synthesize(generate_workloads=True)
    lgmm_ds = LatentGMMSynthesizer(veri_profile, spec, topo, device).synthesize(generate_workloads=True)
    real_vr = load_real_benchmark_stream("VeRi-Seed", fold_idx=0)
    for ds_item in (s0, ngw_ds, lgmm_ds, real_vr):
        tbls = export_standardized_tables(ds_item)
        assert set(tbls["Snapshots_Stream"].keys()) == {"sid", "cam_id", "t_obs", "emb"}
        assert "veh_id" not in tbls["Snapshots_Stream"]
        assert "is_bh" not in tbls["Snapshots_Stream"]
        assert len(tbls["GroundTruth_Trajectories"]["veh_id"]) == len(ds_item["snap_veh"])


if __name__ == "__main__":
    for _fn in (
        test_yaml_configs_and_declarative_loading,
        test_synthesis_baselines,
        test_end_to_end_synthesis_tables_and_six_reference_engines,
        test_d512_calibration_and_epsilon_dp_budget,
        test_oracle_waterfall_veri_seed_and_baseline_table_exports,
    ):
        _fn()
        print(f"  [PASS] {_fn.__name__}")
