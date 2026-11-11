# Portolan: A Tail-Calibrated Benchmark for City-Scale Cross-Camera Trajectory Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](requirements.txt)

Open-source benchmark and evaluation suite for **Portolan**.

Portolan is a GPU-accelerated, tail-calibrated stream and query workload synthesizer for evaluating cross-camera trajectory recovery, vector similarity search, and provenance-enabled incremental view maintenance (IVM) at metropolitan scale. Instead of rendering 3D city assets and multi-camera video frames, Portolan couples arterial signal-wave mobility with $O(Nd)$ Givens $SO(d)$ camera rotations, von Mises-Fisher (vMF) appearance dispersion, fleet identity-twin cones, and glare-induced **Embedding Black-Holes** directly on the unit hypersphere $\mathbb{S}^{d-1}$.

---

## 1. Repository Structure

The repository is organized into modular packages for the core synthesizer, three synthesis baselines, and six evaluated trajectory/query packages (seven solver configurations including `MCF-Sharded`):

```text
Portolan/
├── portolan/                        # Core Portolan synthesizer, candidate index & Q1-Q4 harness
│   ├── profiles_and_topology.py     # Road-camera graph G=(C,E), SeedProfile, WorkloadSpec & stream loader
│   ├── synthesizer.py               # Coupled mobility, SO(d)+vMF+Black-Hole manifold & S0->S4 lineage tapes
│   ├── candidate_graph.py           # Shared GPU spatio-temporal candidate index
│   ├── query_eval.py                # Unified BaseDBEngine interface & Q1-Q4 workload evaluators
│   └── evaluation.py                # Distributional distances (W1, JSD, MMD), IDF1/MOTA & rank metrics
├── data/                            # Bundled anonymized 256-D real-world feature streams (11.6 MiB)
│   ├── real_cityflow_stream.npz     # Real-CityFlow cross-camera sightings & road topology
│   ├── real_veri776_stream.npz      # Real-VeRi cross-camera sightings & ring-road topology
│   └── real_seed_profiles.json      # Measured CityFlow-Seed and VeRi-Seed calibration profiles
├── scripts/
│   └── extract_real_benchmarks.py   # Feature extraction script from raw CityFlowV2 / VeRi-776 imagery
├── sim3d_pipeline/                  # [Synthesis Baseline 1] 3D-Sim-Pipeline (Stage-profiled 3D simulation model)
├── ngw/                             # [Synthesis Baseline 2] NGW (Normalized isotropic Gaussian walk)
├── lgmm/                            # [Synthesis Baseline 3] L-GMM (Low-rank Euclidean Gaussian mixture)
├── global_bm/                       # [Evaluated System 1] Global-BM (Bipartite Hungarian matching)
├── mcf_tracker/                     # [Evaluated System 2] MCF-Tracker & MCF-Sharded (Min-cost DAG network flow)
├── online_clust/                    # [Evaluated System 3] Online-Clust (Streaming threshold clustering)
├── vec_filter/                      # [Evaluated System 4] Vec-Filter (Spatio-temporal filtered vector search)
├── prob_path/                       # [Evaluated System 5] Prob-Path (Road-network Markov transition DP)
├── prov_ivm/                        # [Evaluated System 6] Prov-IVM (Top-K Competitive Provenance Graph & IVM)
├── experiments/
│   └── run_all_experiments.py       # Benchmark driver (outputs JSON & CSV evaluation tables)
├── configs/
│   ├── default_4090.yaml            # Benchmark scale presets (Tiny, Small, Medium, CityScale)
│   └── cityflow_benchmark.yaml      # Declarative single-file workload configuration
├── examples/
│   └── quickstart_benchmark.py      # Minimal end-to-end synthesis & 6-system evaluation example
├── tests/
│   └── test_open_source.py          # Automated unit test suite
├── reproduce_all.sh                 # One-command verification and experiment runner
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Python dependencies
└── LICENSE                          # MIT License
```

---

## 2. Bundled Real-World Feature Streams

To support out-of-the-box evaluation without redistributing raw licensed surveillance videos, `data/` includes anonymized 256-D unit-sphere ReID embeddings, camera timestamps, and road-graph coordinates (`11.6 MiB` total) with disjoint calibration (`cal`) and evaluation (`eval`) splits:

- **`Real-CityFlow`** (`data/real_cityflow_stream.npz`): `3,037` vehicle passes across `40` cameras (`S01`–`S05` from CityFlowV2), split into upstream calibration corridors (`S01+S02`) and disjoint downstream evaluation corridors (`S03–S05`).
- **`Real-VeRi`** (`data/real_veri776_stream.npz`): `1,968` vehicle passes across `20` cameras (`273` trips from held-out `VeRi-776` test vehicles), split into disjoint calibration and evaluation trip sets.

*(Optional)* Users with licensed access to raw CityFlowV2 and VeRi-776 imagery can re-extract the `.npz` streams via `scripts/extract_real_benchmarks.py`.

---

## 3. Quickstart & Evaluation

### Requirements
- Python $\ge 3.10$, PyTorch $\ge 2.0$, NumPy $\ge 1.24$, SciPy $\ge 1.10$, PyYAML $\ge 6.0$, Pytest $\ge 7.0$.
- Supports CPU or any CUDA GPU (`--device auto|cpu|cuda|cuda:N`). Wave-streamed synthesis bounds `Portolan-CityScale` ($5.10\text{M}$ sightings, $d=256$) to $\sim 3.0\text{ GB}$ peak GPU memory.

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Step 1: Fast Verification & Quickstart Demo (~4s on CPU/GPU)
Runs the unit test suite across all modules and executes a self-contained synthesis + $Q_1$–$Q_4$ evaluation pass:
```bash
bash reproduce_all.sh --quick
```

### Step 2: Run Benchmark Experiments
Run the full experimental suite and generate JSON metrics and CSV summary tables (`experiments/results/*.csv`):
```bash
CUDA_VISIBLE_DEVICES=0 bash reproduce_all.sh
```
Individual experiment groups can also be executed modularly:
```bash
python experiments/run_all_experiments.py --only exp1 --device cuda:0
python experiments/run_all_experiments.py --only exp23,exp3b,exp3c,exp4,exp5,exp6 --device cuda:0
```

---

## 4. Minimal Python API

```python
from portolan import (
    PortolanSynthesizer,
    RoadCameraTopology,
    SpatioTemporalCandidateGraph,
    WorkloadSpec,
    compute_mtmc_metrics,
    export_standardized_tables,
    load_real_benchmark_stream,
    select_best_device,
)
from global_bm import EagerHungarianEngine
from mcf_tracker import MinCostNetFlowEngine
from online_clust import StreamGreedyClustEngine
from vec_filter import VecDBFilteredANNEngine
from prob_path import TopoMarkovMAPEngine
from prov_ivm import ProvenanceCascadingDBEngine

device = select_best_device()
profile, spec = WorkloadSpec.from_yaml("configs/cityflow_benchmark.yaml")
topology = RoadCameraTopology(n_cam=spec.n_cam, seed=spec.seed)
dataset = PortolanSynthesizer(profile, spec, topology, device).synthesize(generate_workloads=True)

cgraph = SpatioTemporalCandidateGraph(dataset, device)
for engine in [
    EagerHungarianEngine(),
    MinCostNetFlowEngine(),
    StreamGreedyClustEngine(),
    VecDBFilteredANNEngine(),
    TopoMarkovMAPEngine(),
    ProvenanceCascadingDBEngine(),
]:
    trajs, _, elapsed_sec = engine.recover_trajectories(cgraph)
    m = compute_mtmc_metrics(trajs, dataset["snap_veh"])
    print(f"{engine.name:14s} | IDF1={m['idf1']:.3f} | Time={elapsed_sec:.2f}s")

real_cf = load_real_benchmark_stream("CityFlow-Seed", fold_idx=1, split="eval")
tables = export_standardized_tables(dataset)
```

---

## 5. License

Released under the [MIT License](LICENSE). When using the bundled `Real-CityFlow` and `Real-VeRi` feature streams, please also cite the original CityFlow (Tang et al., CVPR 2019) and VeRi-776 (Liu et al., ECCV 2016) datasets.
