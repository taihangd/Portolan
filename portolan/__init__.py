"""Portolan: Simulation-Free Cross-Camera Trajectory Stream & Multi-Modal Workload Synthesizer."""

from .profiles_and_topology import (
    RoadCameraTopology,
    SeedProfile,
    WorkloadSpec,
    calibrate_seed_profile_from_stream,
    compute_empirical_profile_stats,
    get_benchmark_spec,
    get_seed_profile,
    load_real_benchmark_stream,
    select_best_device,
)
from .synthesizer import (
    PortolanSynthesizer,
    build_multimodal_query_workloads,
    export_standardized_tables,
    sample_vmf_radial_w,
)
from .candidate_graph import SpatioTemporalCandidateGraph, find_arrival_troughs
from .query_eval import (
    BaseDBEngine,
    evaluate_q1_from_trajs,
    evaluate_q2_eager_from_trajs,
    evaluate_q2_track_pairs,
    evaluate_q3_from_trajs,
    links_to_trajectories,
)
from .evaluation import (
    compute_fidelity_metrics,
    compute_mtmc_metrics,
    compute_multiconfig_ranking_stats,
    compute_ranking_consistency,
    compute_vehicle_block_bootstrap_se,
    extract_stream_distributions,
)

__version__ = "1.0.0"
__all__ = [
    "RoadCameraTopology",
    "SeedProfile",
    "WorkloadSpec",
    "calibrate_seed_profile_from_stream",
    "compute_empirical_profile_stats",
    "load_real_benchmark_stream",
    "get_benchmark_spec",
    "get_seed_profile",
    "select_best_device",
    "PortolanSynthesizer",
    "build_multimodal_query_workloads",
    "export_standardized_tables",
    "sample_vmf_radial_w",
    "SpatioTemporalCandidateGraph",
    "find_arrival_troughs",
    "BaseDBEngine",
    "links_to_trajectories",
    "evaluate_q1_from_trajs",
    "evaluate_q2_eager_from_trajs",
    "evaluate_q2_track_pairs",
    "evaluate_q3_from_trajs",
    "compute_fidelity_metrics",
    "compute_mtmc_metrics",
    "compute_multiconfig_ranking_stats",
    "compute_ranking_consistency",
    "compute_vehicle_block_bootstrap_se",
    "extract_stream_distributions",
]
