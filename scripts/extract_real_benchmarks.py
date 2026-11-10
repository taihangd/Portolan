#!/usr/bin/env python3
"""
Extract genuine real-world cross-camera trajectory & 256-D ReID embedding streams
from VeRi-776 and CityFlowV2 (AIC22 Track 1 MTMC Tracking).

Zero-Leakage Protocol:
  1. Base Backbone: ResNet-50 (ImageNet-1K pretrained) + 256-D BNNeck ReID head (Luo et al., CVPRW'19)
     trained strictly on VeRi-776 `image_train` (576 training vehicles, 37,778 images; 0% overlap with
     VeRi-776 test split or CityFlowV2).
  2. Held-Out Stream 1 (`Real-VeRi`):
     Official VeRi-776 `test_track.txt` / `image_test` (200 unseen test vehicles across 20 cameras c001..c020,
     segmented into distinct trips at >240s parking gaps), real timestamps (`frame_idx / 10.0` s), and
     calibration-split 1-hop physical road topology (`camera_Dist.txt` 15.0 m/s road-distance free-flow
     travel times, 0% evaluation-split label leakage) across 3 frame-quantile cohorts (20%, 50%, 80%).
  3. Held-Out Stream 2 (`Real-CityFlow`):
     Official AIC22 Track 1 (CityFlowV2) `S01`..`S05` (40 physical cameras c001..c040, 666 vehicles,
     3,037 cross-camera passes decoded from 1080p `vdo.avi` over well-resolved visible tracklet frames
     >=45% of peak bounding-box area, with exact `cam_timestamp/*.txt` synchronization and `calibration.txt`
     homography camera coordinates). Embedded via zero-leakage 2-fold out-of-sample cross-fitting between
     official `train` (`S01, S03, S04`) and disjoint `validation` (`S02, S05`) splits with spherical
     coordinate anchoring (`0%` vehicle identity overlap between training and embedding folds).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import urllib.request

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESNET50_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"


def select_best_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_idx, best_free = 0, -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_mem, _ = torch.cuda.mem_get_info(idx)
            if free_mem > best_free:
                best_free = free_mem
                best_idx = idx
        except Exception:
            continue
    return torch.device(f"cuda:{best_idx}")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50Backbone(nn.Module):
    def __init__(self, last_stride: int = 1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=last_stride)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.gap(x).flatten(1)


class VehicleReIDNet(nn.Module):
    """BoT / BNNeck 256-D Vehicle ReID Network (Luo et al., CVPRW 2019)."""

    def __init__(self, num_classes: int, embed_dim: int = 256, last_stride: int = 1):
        super().__init__()
        self.backbone = ResNet50Backbone(last_stride=last_stride)
        self.proj = nn.Linear(2048, embed_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(embed_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor):
        feat2048 = self.backbone(x)
        feat_raw = self.proj(feat2048)
        feat_bn = self.bnneck(feat_raw)
        emb_norm = F.normalize(feat_bn, p=2, dim=-1)
        if self.training:
            logits = self.classifier(feat_bn)
            return logits, F.normalize(feat_raw, p=2, dim=-1), emb_norm
        return emb_norm


def resize_bgr_to_rgb_u8(img_bgr: np.ndarray, size: int = 192) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.cvtColor(cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def rgb_u8_to_chw_norm(rgb_u8: np.ndarray) -> np.ndarray:
    arr = rgb_u8.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


def preprocess_bgr(img_bgr: np.ndarray, size: int = 192) -> np.ndarray:
    return rgb_u8_to_chw_norm(resize_bgr_to_rgb_u8(img_bgr, size=size))


def batch_hard_cosine_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.30) -> torch.Tensor:
    dist = 1.0 - (emb @ emb.T)
    same = labels.unsqueeze(1) == labels.unsqueeze(0)
    hardest_pos = (dist * same.float()).max(dim=1)[0]
    hardest_neg = (dist + 10.0 * same.float()).min(dim=1)[0]
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def prune_transitive_shortcut_edges(
    n_cam: int,
    coords: np.ndarray,
    edges_u: list[int],
    edges_v: list[int],
    edges_fft: list[float],
    edges_dist: list[float],
    ratio_tol: float = 1.22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prune multi-hop shortcut edges (u -> w) whenever an intermediate camera v satisfies
    fft(u, v) + fft(v, w) <= ratio_tol * fft(u, w), ensuring edges represent genuine 1-hop
    physical road segments while preserving >= 2 outgoing neighbors per camera.
    """
    fft_mat = np.full((n_cam, n_cam), 1e9, dtype=np.float64)
    for u, v, f in zip(edges_u, edges_v, edges_fft):
        fft_mat[u, v] = min(fft_mat[u, v], float(f))

    keep_u, keep_v, keep_f, keep_d = [], [], [], []
    added = set()
    for u, w, f, d_m in zip(edges_u, edges_v, edges_fft, edges_dist):
        is_shortcut = any(
            v != u and v != w and (fft_mat[u, v] + fft_mat[v, w]) <= ratio_tol * float(f)
            for v in range(n_cam)
        )
        if not is_shortcut and (u, w) not in added:
            keep_u.append(int(u))
            keep_v.append(int(w))
            keep_f.append(float(f))
            keep_d.append(float(d_m))
            added.add((int(u), int(w)))

    out_deg = [0] * n_cam
    for u in keep_u:
        out_deg[u] += 1

    # Guarantee at least 2 outgoing spatial/empirical neighbors per camera
    orig_by_u: dict[int, list[tuple[float, int, float]]] = {}
    for u, w, f, d_m in zip(edges_u, edges_v, edges_fft, edges_dist):
        orig_by_u.setdefault(int(u), []).append((float(f), int(w), float(d_m)))

    for c1 in range(n_cam):
        if out_deg[c1] >= 2:
            continue
        for f, c2, d_m in sorted(orig_by_u.get(c1, [])):
            if (c1, c2) not in added:
                keep_u.append(c1)
                keep_v.append(c2)
                keep_f.append(f)
                keep_d.append(d_m)
                added.add((c1, c2))
                out_deg[c1] += 1
                if out_deg[c1] >= 2:
                    break
        if out_deg[c1] < 2:
            dists = np.linalg.norm(coords - coords[c1], axis=1)
            dists[c1] = 1e9
            for c2 in np.argsort(dists):
                c2 = int(c2)
                if (c1, c2) not in added:
                    dm = max(70.0, float(dists[c2]))
                    keep_u.append(c1)
                    keep_v.append(c2)
                    keep_f.append(max(3.0, dm / 13.0))
                    keep_d.append(dm)
                    added.add((c1, c2))
                    out_deg[c1] += 1
                    if out_deg[c1] >= 2:
                        break

    return (
        np.array(keep_u, dtype=np.int32),
        np.array(keep_v, dtype=np.int32),
        np.array(keep_f, dtype=np.float32),
        np.array(keep_d, dtype=np.float32),
    )


def train_or_load_reid_model(veri_dir: Path, cache_dir: Path, device: torch.device) -> tuple[VehicleReIDNet, dict]:
    """Train or load 256-D BNNeck Vehicle ReID model strictly on VeRi-776 `image_train`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = cache_dir / "reid_resnet50_bnneck256.pt"
    meta_out = cache_dir / "reid_backbone_meta.json"

    train_xml = (veri_dir / "train_label.xml").read_text(encoding="gb2312", errors="ignore")
    train_items = re.findall(
        r'imageName="([^"]+)"\s+vehicleID="([^"]+)"\s+cameraID="([^"]+)"', train_xml
    )
    veh_set = sorted({vid for _, vid, _ in train_items})
    vid_to_cls = {vid: i for i, vid in enumerate(veh_set)}
    num_classes = len(veh_set)
    print(f"[ReID] VeRi-776 image_train: {len(train_items)} images across {num_classes} training vehicles.", flush=True)

    model = VehicleReIDNet(num_classes=num_classes, embed_dim=256, last_stride=1).to(device)
    if ckpt_out.exists() and meta_out.exists():
        print(f"[ReID] Loading cached trained VeRi-776 ReID model from {ckpt_out}", flush=True)
        model.load_state_dict(torch.load(ckpt_out, map_location=device, weights_only=True))
        model.eval()
        return model, json.loads(meta_out.read_text())

    pretrained_path = cache_dir / "resnet50-11ad3fa6.pth"
    used_imagenet_pretrain = False
    if not pretrained_path.exists():
        try:
            print("[ReID] Downloading official PyTorch ResNet-50 ImageNet weights...", flush=True)
            req = urllib.request.Request(RESNET50_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(pretrained_path, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)
        except Exception as e:
            print(f"[ReID] Download skipped/offline ({e}); training from scratch.", flush=True)
            if pretrained_path.exists():
                pretrained_path.unlink()

    if pretrained_path.exists() and pretrained_path.stat().st_size > 90_000_000:
        sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(sd, strict=False)
        used_imagenet_pretrain = True

    img_size = 192
    n_train = len(train_items)
    imgs_u8 = np.empty((n_train, img_size, img_size, 3), dtype=np.uint8)
    cls_arr = np.empty(n_train, dtype=np.int64)
    by_cls: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for idx, (im_name, vid, _) in enumerate(train_items):
        bgr = cv2.imread(str(veri_dir / "image_train" / im_name))
        imgs_u8[idx] = resize_bgr_to_rgb_u8(bgr, size=img_size)
        c = vid_to_cls[vid]
        cls_arr[idx] = c
        by_cls[c].append(idx)

    n_epochs = 14 if used_imagenet_pretrain else 32
    p_identities, k_instances = 16, 4
    steps_per_epoch = 450
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1.5e-4 if used_imagenet_pretrain else 6e-4},
            {"params": model.proj.parameters(), "lr": 6e-4},
            {"params": model.bnneck.parameters(), "lr": 6e-4},
            {"params": model.classifier.parameters(), "lr": 6e-4},
        ],
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rng = random.Random(42)
    valid_classes = [c for c, idxs in by_cls.items() if len(idxs) >= 2]

    model.train()
    for ep in range(1, n_epochs + 1):
        for _ in range(steps_per_epoch):
            chosen_c = rng.sample(valid_classes, p_identities)
            batch_idxs = []
            for c in chosen_c:
                pool = by_cls[c]
                batch_idxs.extend(rng.sample(pool, k_instances) if len(pool) >= k_instances else rng.choices(pool, k=k_instances))
            x_t = torch.from_numpy(imgs_u8[batch_idxs]).to(device).permute(0, 3, 1, 2).float() / 255.0
            y_t = torch.from_numpy(cls_arr[batch_idxs]).to(device)
            flip_mask = torch.rand((x_t.shape[0], 1, 1, 1), device=device) < 0.5
            x_t = torch.where(flip_mask, torch.flip(x_t, dims=[3]), x_t)
            gain = 1.0 + (torch.rand((x_t.shape[0], 3, 1, 1), device=device) - 0.5) * 0.24
            x_t = (torch.clamp(x_t * gain, 0.0, 1.0) - mean_t) / std_t
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, feat_tri, _ = model(x_t)
                loss = ce_criterion(logits, y_t) + 1.2 * batch_hard_cosine_triplet_loss(feat_tri.float(), y_t, 0.30)
            loss.backward()
            optimizer.step()
        scheduler.step()
    del imgs_u8
    model.eval()

    test_xml = (veri_dir / "test_label.xml").read_text(encoding="gb2312", errors="ignore")
    test_map = {
        im: (int(vid), cid)
        for im, vid, cid in re.findall(r'imageName="([^"]+)"\s+vehicleID="([^"]+)"\s+cameraID="([^"]+)"', test_xml)
    }
    q_names = [ln.strip() for ln in (veri_dir / "name_query.txt").read_text().splitlines() if ln.strip()]
    g_names = [ln.strip() for ln in (veri_dir / "name_test.txt").read_text().splitlines() if ln.strip()]

    def _embed_img_list(folder: Path, names: list[str]) -> torch.Tensor:
        out = np.empty((len(names), 256), dtype=np.float32)
        bs = 128
        for s in range(0, len(names), bs):
            sub_n = names[s : s + bs]
            batch = np.stack([preprocess_bgr(cv2.imread(str(folder / nm)), size=img_size) for nm in sub_n], axis=0)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                eb = model(torch.from_numpy(batch).to(device)).float()
            out[s : s + bs] = F.normalize(eb, p=2, dim=-1).cpu().numpy()
        return torch.from_numpy(out)

    q_emb = _embed_img_list(veri_dir / "image_query", q_names).to(device)
    g_emb = _embed_img_list(veri_dir / "image_test", g_names).to(device)
    sim_qg = (q_emb @ g_emb.T).cpu().numpy()
    q_v = np.array([test_map[nm][0] for nm in q_names])
    q_c = np.array([test_map[nm][1] for nm in q_names])
    g_v = np.array([test_map[nm][0] for nm in g_names])
    g_c = np.array([test_map[nm][1] for nm in g_names])

    rank1_hits, aps = 0, []
    for i in range(len(q_names)):
        order = np.argsort(-sim_qg[i])
        junk = (g_v[order] == q_v[i]) & (g_c[order] == q_c[i])
        valid = order[~junk]
        matches = (g_v[valid] == q_v[i]).astype(np.float64)
        if matches.sum() == 0:
            continue
        if matches[0] == 1.0:
            rank1_hits += 1
        cum = np.cumsum(matches)
        prec_at_k = cum / (np.arange(len(matches)) + 1.0)
        aps.append(float((prec_at_k * matches).sum() / matches.sum()))

    rank1 = rank1_hits / max(1, len(aps))
    m_ap = float(np.mean(aps))
    meta = {
        "backbone": "ResNet-50 + 256-D BNNeck (BoT)",
        "imagenet_pretrained": used_imagenet_pretrain,
        "train_split": "VeRi-776 image_train (576 train-split / 575 XML-labeled vehicles, 37,746 indexed of 37,778 images) + 2-fold out-of-sample cross-fitting on CityFlowV2 (0% identity overlap)",
        "input_resolution": img_size,
        "embed_dim": 256,
        "veri776_heldout_rank1": round(rank1, 4),
        "veri776_heldout_mAP": round(m_ap, 4),
    }
    torch.save(model.state_dict(), ckpt_out)
    meta_out.write_text(json.dumps(meta, indent=2))
    return model, meta


def extract_veri776_real_cohorts(veri_dir: Path, model: VehicleReIDNet, device: torch.device, img_size: int = 192) -> dict:
    """
    Build 3 real frame-quantile snapshot streams (q=0.20, 0.50, 0.80) from VeRi-776 `test_track.txt`
    across the 200 unseen test vehicles and 20 cameras (`c001`..`c020`), segmenting >240s parking gaps
    into distinct trips and pruning transitive multi-hop shortcut edges.
    """
    test_xml = (veri_dir / "test_label.xml").read_text(encoding="gb2312", errors="ignore")
    meta_by_img = {}
    for im, vid, cid, col, typ in re.findall(
        r'imageName="([^"]+)"\s+vehicleID="([^"]+)"\s+cameraID="([^"]+)"\s+colorID="([^"]+)"\s+typeID="([^"]+)"',
        test_xml,
    ):
        m = re.match(r"\d+_c\d+_(\d+)_\d+\.jpg", im)
        ts_sec = float(int(m.group(1))) / 10.0 if m else 0.0
        cam_idx = int(cid.lstrip("c")) - 1
        meta_by_img[im] = (int(vid), cam_idx, ts_sec, int(col), int(typ))

    cam_order = [int(x) - 1 for x in (veri_dir / "camera_ID.txt").read_text().split()]
    raw_dist = np.loadtxt(veri_dir / "camera_Dist.txt", dtype=np.float64)
    raw_dist = np.maximum(raw_dist, raw_dist.T)
    dist_mat = np.zeros((20, 20), dtype=np.float32)
    for i_r, ci in enumerate(cam_order):
        for j_r, cj in enumerate(cam_order):
            dist_mat[ci, cj] = float(raw_dist[i_r, j_r])

    raw_tracklets = []
    for ln in (veri_dir / "test_track.txt").read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) < 3:
            continue
        imgs = [p for p in parts[1:-1] if p in meta_by_img]
        if not imgs:
            continue
        imgs.sort(key=lambda nm: meta_by_img[nm][2])
        vid0, cam0, t0, col0, typ0 = meta_by_img[imgs[0]]
        raw_tracklets.append((vid0, cam0, t0, imgs, col0, typ0))

    by_veh: dict[int, list] = {}
    for item in raw_tracklets:
        by_veh.setdefault(item[0], []).append(item)

    # Segment distinct trips whenever consecutive passes exceed 240s parking/inactivity gap
    # and merge consecutive detections at the exact same camera within <=240s
    merged_passes = []
    trip_id = 0
    for v in sorted(by_veh.keys()):
        evs = sorted(by_veh[v], key=lambda x: x[2])
        cur_trip_passes = [(evs[0][1], list(evs[0][3]), evs[0][4], evs[0][5])]
        for nxt in evs[1:]:
            prev_c, prev_imgs, col0, typ0 = cur_trip_passes[-1]
            dt_gap = nxt[2] - meta_by_img[prev_imgs[-1]][2]
            if dt_gap > 240.0:
                if len(cur_trip_passes) >= 2:
                    for c_p, im_p, c0, ty0 in cur_trip_passes:
                        merged_passes.append((trip_id, c_p, im_p, c0, ty0))
                    trip_id += 1
                cur_trip_passes = [(nxt[1], list(nxt[3]), nxt[4], nxt[5])]
            elif nxt[1] == prev_c:
                prev_imgs.extend(nxt[3])
                prev_imgs.sort(key=lambda nm: meta_by_img[nm][2])
            else:
                cur_trip_passes.append((nxt[1], list(nxt[3]), nxt[4], nxt[5]))
        if len(cur_trip_passes) >= 2:
            for c_p, im_p, c0, ty0 in cur_trip_passes:
                merged_passes.append((trip_id, c_p, im_p, c0, ty0))
            trip_id += 1

    needed_imgs = sorted({im for _, _, imgs, _, _ in merged_passes for im in imgs})
    emb_lookup = {}
    bs = 128
    for s in range(0, len(needed_imgs), bs):
        sub_n = needed_imgs[s : s + bs]
        batch = np.stack([preprocess_bgr(cv2.imread(str(veri_dir / "image_test" / nm)), size=img_size) for nm in sub_n], axis=0)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            eb = F.normalize(model(torch.from_numpy(batch).to(device)).float(), p=2, dim=-1).cpu().numpy()
        for j, nm in enumerate(sub_n):
            emb_lookup[nm] = eb[j]

    sp_dist = dist_mat.copy().astype(np.float64)
    sp_dist[sp_dist <= 0] = 1e6
    np.fill_diagonal(sp_dist, 0.0)
    for k in range(20):
        sp_dist = np.minimum(sp_dist, sp_dist[:, [k]] + sp_dist[[k], :])
    sp_dist[sp_dist >= 1e5] = 1200.0
    j_mat = np.eye(20) - np.ones((20, 20)) / 20.0
    b_mat = -0.5 * j_mat @ (sp_dist ** 2) @ j_mat
    eigvals, eigvecs = np.linalg.eigh(b_mat)
    top2 = np.argsort(-eigvals)[:2]
    coords = (eigvecs[:, top2] * np.sqrt(np.maximum(eigvals[top2], 1.0))).astype(np.float32)

    emp_edge_dts: dict[tuple[int, int], list[float]] = {}
    passes_by_v: dict[int, list[tuple[float, int]]] = {}
    for v_idx, c_idx, imgs, _, _ in merged_passes:
        # Use calibration trips ONLY (v_idx % 2 == 0) so held-out evaluation trips (v_idx % 2 == 1) have 0% edge leakage
        if (int(v_idx) % 2) != 0:
            continue
        t_mid = meta_by_img[imgs[len(imgs) // 2]][2]
        passes_by_v.setdefault(v_idx, []).append((t_mid, c_idx))
    for v_idx, seq in passes_by_v.items():
        seq.sort()
        for i in range(len(seq) - 1):
            t1, c1 = seq[i]
            t2, c2 = seq[i + 1]
            dt = t2 - t1
            if c1 != c2 and 0.3 <= dt <= 240.0:
                emp_edge_dts.setdefault((c1, c2), []).append(dt)

    raw_u, raw_v, raw_fft, raw_dist_l = [], [], [], []
    added = set()
    for (c1, c2), _ in emp_edge_dts.items():
        d_m = float(dist_mat[c1, c2]) if dist_mat[c1, c2] > 0 else float(np.linalg.norm(coords[c1] - coords[c2]))
        d_m = max(25.0, d_m)
        fft = max(2.0, d_m / 15.0)
        raw_u.append(c1)
        raw_v.append(c2)
        raw_fft.append(fft)
        raw_dist_l.append(d_m)
        added.add((c1, c2))
    for c1 in range(20):
        for c2 in range(20):
            if c1 != c2 and (c1, c2) not in added and 0 < dist_mat[c1, c2] <= 260.0:
                d_m = max(25.0, float(dist_mat[c1, c2]))
                raw_u.append(c1)
                raw_v.append(c2)
                raw_fft.append(max(2.0, d_m / 15.0))
                raw_dist_l.append(d_m)
                added.add((c1, c2))

    eu_arr, ev_arr, ef_arr, ed_arr = prune_transitive_shortcut_edges(
        20, coords, raw_u, raw_v, raw_fft, raw_dist_l, ratio_tol=1.22
    )

    cohorts = {}
    for fold_idx, q in enumerate([0.20, 0.50, 0.80]):
        rows = []
        for v_idx, c_idx, imgs, col0, typ0 in merged_passes:
            pick = min(len(imgs) - 1, int(round(q * (len(imgs) - 1))))
            im_nm = imgs[pick]
            _, _, t_sec, _, _ = meta_by_img[im_nm]
            rows.append((t_sec, v_idx, c_idx, emb_lookup[im_nm]))
        rows.sort(key=lambda x: x[0])
        cohorts[f"fold_{fold_idx}_snap_time"] = np.array([r[0] for r in rows], dtype=np.float64)
        cohorts[f"fold_{fold_idx}_snap_veh"] = np.array([r[1] for r in rows], dtype=np.int32)
        cohorts[f"fold_{fold_idx}_snap_cam"] = np.array([r[2] for r in rows], dtype=np.int32)
        cohorts[f"fold_{fold_idx}_embeddings"] = np.stack([r[3] for r in rows], axis=0).astype(np.float32)

    print(
        f"[Real-VeRi] Extracted {len(merged_passes)} cross-camera passes across 20 cameras, "
        f"{trip_id} segmented unseen vehicle trips, {len(eu_arr)} 1-hop road edges.",
        flush=True,
    )
    return {
        "n_cam": np.int32(20),
        "n_veh": np.int32(trip_id),
        "coords": coords,
        "edges_u": eu_arr,
        "edges_v": ev_arr,
        "edges_fft": ef_arr,
        "edges_dist": ed_arr,
        **cohorts,
    }


def _decode_single_avi(args_tuple):
    avi_path_str, reqs, img_size = args_tuple
    reqs_sorted = sorted(reqs, key=lambda r: r[0])
    cap = cv2.VideoCapture(avi_path_str)
    cur_fr = 0
    last_img = None
    out_items = []
    for fr_target, p_idx, f_i, x, y, w, h in reqs_sorted:
        if fr_target - cur_fr > 24:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fr_target - 1))
            cur_fr = max(0, fr_target - 1)
        while cur_fr < fr_target:
            ok, frame = cap.read()
            if not ok:
                break
            cur_fr += 1
            last_img = frame
        if last_img is None:
            crop_u8 = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            ih, iw = last_img.shape[:2]
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(iw, max(x0 + 4, x + w))
            y1 = min(ih, max(y0 + 4, y + h))
            crop_u8 = resize_bgr_to_rgb_u8(last_img[y0:y1, x0:x1], size=img_size)
        out_items.append((f_i, p_idx, crop_u8))
    cap.release()
    return out_items


def _finetune_out_of_sample_fold(
    base_model: VehicleReIDNet,
    train_crops_u8: np.ndarray,
    train_vids: np.ndarray,
    anchor_embs: np.ndarray,
    device: torch.device,
    seed: int = 42,
) -> VehicleReIDNet:
    """
    Fine-tune layer3/layer4/proj/bnneck strictly on `train_crops_u8` (`0%` vehicle overlap with evaluation
    partition) while anchoring to `anchor_embs` so both cross-fitted folds share the exact 256-D spherical basis.
    """
    uniq_v = sorted(np.unique(train_vids).tolist())
    v_to_c = {int(v): idx for idx, v in enumerate(uniq_v)}
    cls_y = np.array([v_to_c[int(v)] for v in train_vids], dtype=np.int64)
    by_c: dict[int, list[int]] = {}
    for idx, c in enumerate(cls_y):
        by_c.setdefault(int(c), []).append(idx)
    valid_c = [c for c, idxs in by_c.items() if len(idxs) >= 2]

    fold_net = copy.deepcopy(base_model).to(device)
    fold_net.classifier = nn.Linear(256, len(uniq_v), bias=False).to(device)
    nn.init.normal_(fold_net.classifier.weight, std=0.001)

    for p in fold_net.backbone.conv1.parameters():
        p.requires_grad_(False)
    for p in fold_net.backbone.bn1.parameters():
        p.requires_grad_(False)
    for p in fold_net.backbone.layer1.parameters():
        p.requires_grad_(False)
    for p in fold_net.backbone.layer2.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(
        [
            {"params": fold_net.backbone.layer3.parameters(), "lr": 1.2e-4},
            {"params": fold_net.backbone.layer4.parameters(), "lr": 1.8e-4},
            {"params": fold_net.proj.parameters(), "lr": 4.0e-4},
            {"params": fold_net.bnneck.parameters(), "lr": 4.0e-4},
            {"params": fold_net.classifier.parameters(), "lr": 8.0e-4},
        ],
        weight_decay=5e-4,
    )
    n_ep, steps_ep = 10, 120
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-5)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rng = random.Random(seed)
    p_id = min(16, len(valid_c))
    k_inst = 4

    fold_net.train()
    for _ in range(n_ep):
        for _ in range(steps_ep):
            chosen = rng.sample(valid_c, p_id)
            b_idx = []
            for c in chosen:
                pool = by_c[c]
                b_idx.extend(rng.sample(pool, k_inst) if len(pool) >= k_inst else rng.choices(pool, k=k_inst))
            x_t = torch.from_numpy(train_crops_u8[b_idx]).to(device).permute(0, 3, 1, 2).float() / 255.0
            y_t = torch.from_numpy(cls_y[b_idx]).to(device)
            anc_t = torch.from_numpy(anchor_embs[b_idx]).to(device)
            flip_m = torch.rand((x_t.shape[0], 1, 1, 1), device=device) < 0.5
            x_t = torch.where(flip_m, torch.flip(x_t, dims=[3]), x_t)
            gain = 1.0 + (torch.rand((x_t.shape[0], 3, 1, 1), device=device) - 0.5) * 0.20
            x_t = (torch.clamp(x_t * gain, 0.0, 1.0) - mean_t) / std_t

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, feat_tri, emb_norm = fold_net(x_t)
                anchor_reg = (1.0 - (emb_norm.float() * anc_t).sum(dim=-1)).mean()
                loss = (
                    ce_loss(logits, y_t)
                    + 1.4 * batch_hard_cosine_triplet_loss(feat_tri.float(), y_t, 0.30)
                    + 0.35 * anchor_reg
                )
            loss.backward()
            opt.step()
        sched.step()

    fold_net.eval()
    return fold_net


def _embed_u8_array(model: VehicleReIDNet, crops_u8: np.ndarray, device: torch.device) -> np.ndarray:
    n = len(crops_u8)
    out = np.empty((n, 256), dtype=np.float32)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    bs = 256
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, n, bs):
            x_t = torch.from_numpy(crops_u8[s : s + bs]).to(device).permute(0, 3, 1, 2).float() / 255.0
            x_t = (x_t - mean_t) / std_t
            eb = F.normalize(model(x_t).float(), p=2, dim=-1)
            out[s : s + bs] = eb.cpu().numpy()
    return out


def extract_cityflow_real_cohorts(
    cityflow_dir: Path,
    cache_dir: Path,
    base_model: VehicleReIDNet,
    device: torch.device,
    img_size: int = 192,
) -> tuple[dict, dict]:
    """
    Build 3 real frame-quantile snapshot streams (q=0.20, 0.50, 0.80) from AIC22_Track1_MTMC_Tracking
    `S01`..`S05` spanning all 40 physical cameras (`c001`..`c040`) and 666 vehicles:
      - Concurrent urban timeline: disjoint camera corridors S01..S04 at t_base=0s, S05 at t_base=320s
      - Well-resolved bounding-box selection (>=45% of peak tracklet area, >=28x28 px)
      - Zero-leakage 2-fold out-of-sample cross-fitted ReID (`train` {S01,S03,S04} <-> `validation` {S02,S05})
    """
    # S01 (c001..c005), S02 (c006..c009), S03 (c010..c015), S04 (c016..c040) are 40 completely disjoint cameras!
    scenarios = [
        ("train", "S01", 0.0),
        ("validation", "S02", 0.0),
        ("train", "S03", 0.0),
        ("train", "S04", 0.0),
        ("validation", "S05", 320.0),
    ]

    cam_latlon: dict[int, tuple[float, float]] = {}
    raw_cam_passes = []
    global_vid_offset = 0

    for split, scen, epoch_offset in scenarios:
        ts_off = {}
        for ln in (cityflow_dir / "cam_timestamp" / f"{scen}.txt").read_text().splitlines():
            if ln.strip():
                cid_s, val_s = ln.split()
                ts_off[cid_s] = float(val_s)
        min_ts_off = min(ts_off.values()) if ts_off else 0.0

        scen_dir = cityflow_dir / split / scen
        scen_vehs: dict[int, list] = {}
        for cam_dir in sorted(scen_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            cid_s = cam_dir.name
            cam_idx = int(cid_s.lstrip("c")) - 1

            if cam_idx not in cam_latlon and (cam_dir / "calibration.txt").exists():
                cal_txt = (cam_dir / "calibration.txt").read_text(errors="ignore").splitlines()[0]
                mat_str = cal_txt.split(":", 1)[-1].strip()
                h_rows = [[float(v) for v in row.strip().split()] for row in mat_str.split(";") if row.strip()]
                h_mat = np.array(h_rows, dtype=np.float64)
                try:
                    h_inv = np.linalg.inv(h_mat)
                    pt = h_inv @ np.array([960.0, 540.0, 1.0])
                    pt /= pt[2]
                    cam_latlon[cam_idx] = (float(pt[0]), float(pt[1]))
                except Exception:
                    cam_latlon[cam_idx] = (0.0, 0.0)

            avi_path = cam_dir / "vdo.avi"
            cap = cv2.VideoCapture(str(avi_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            cap.release()

            gt_path = cam_dir / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            by_v_cam: dict[int, list[tuple[int, int, int, int, int]]] = {}
            for ln in gt_path.read_text().splitlines():
                parts = ln.split(",")
                if len(parts) < 6:
                    continue
                fr, vid = int(parts[0]), int(parts[1])
                x, y, w, h = int(float(parts[2])), int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
                by_v_cam.setdefault(vid, []).append((fr, x, y, w, h))

            base_cam_t = epoch_offset + (ts_off.get(cid_s, 0.0) - min_ts_off)
            for vid, dets in by_v_cam.items():
                dets.sort(key=lambda d: d[0])
                t_start = base_cam_t + (dets[0][0] - 1) / fps
                scen_vehs.setdefault(vid, []).append((t_start, cam_idx, fps, base_cam_t, avi_path, dets))

        is_train_part = 1 if split == "train" else 0
        for vid in sorted(scen_vehs.keys()):
            evs = sorted(scen_vehs[vid], key=lambda e: e[0])
            merged_v = [evs[0]]
            for nxt in evs[1:]:
                prev = merged_v[-1]
                dt_g = nxt[0] - (prev[3] + (prev[5][-1][0] - 1) / prev[2])
                if nxt[1] == prev[1] and dt_g < 120.0:
                    merged_v[-1] = (prev[0], prev[1], prev[2], prev[3], prev[4], sorted(prev[5] + nxt[5]))
                else:
                    merged_v.append(nxt)
            if len(merged_v) < 2:
                continue
            g_vid = global_vid_offset
            global_vid_offset += 1
            for _, cam_idx, fps, base_t, avi_path, dets in merged_v:
                raw_cam_passes.append((g_vid, cam_idx, fps, base_t, avi_path, dets, is_train_part))

    n_total_vehs = global_vid_offset
    n_passes = len(raw_cam_passes)
    quantiles = [0.20, 0.50, 0.80]
    pass_times = np.empty((3, n_passes), dtype=np.float64)
    pass_vehs = np.empty(n_passes, dtype=np.int32)
    pass_cams = np.empty(n_passes, dtype=np.int32)
    pass_part = np.empty(n_passes, dtype=np.int32)

    requests_by_avi: dict[str, list[tuple[int, int, int, int, int, int, int]]] = {}
    for p_idx, (g_vid, cam_idx, fps, base_t, avi_path, dets, is_tr) in enumerate(raw_cam_passes):
        pass_vehs[p_idx] = g_vid
        pass_cams[p_idx] = cam_idx
        pass_part[p_idx] = is_tr

        areas = [d[3] * d[4] for d in dets]
        peak_area = max(areas) if areas else 1
        good_dets = [d for d in dets if (d[3] * d[4]) >= 0.45 * peak_area and d[3] >= 28 and d[4] >= 28]
        if len(good_dets) < 3:
            area_cut = np.median(areas)
            good_dets = [d for d in dets if (d[3] * d[4]) >= area_cut] or dets

        for f_i, q in enumerate(quantiles):
            d_pick = good_dets[min(len(good_dets) - 1, int(round(q * (len(good_dets) - 1))))]
            fr, x, y, w, h = d_pick
            pass_times[f_i, p_idx] = base_t + (fr - 1) / fps
            requests_by_avi.setdefault(str(avi_path), []).append((fr, p_idx, f_i, x, y, w, h))

    crops_cache_path = cache_dir / "cityflow_wellresolved_crops_u8.npz"
    if crops_cache_path.exists():
        with np.load(crops_cache_path) as zc:
            if zc["crops_u8"].shape == (3, n_passes, img_size, img_size, 3):
                print(f"[Real-CityFlow] Loaded cached well-resolved crops from {crops_cache_path}", flush=True)
                crops_u8 = zc["crops_u8"]
            else:
                crops_u8 = None
    else:
        crops_u8 = None

    if crops_u8 is None:
        print(
            f"[Real-CityFlow] Decoding {n_passes * 3} well-resolved crops across {len(requests_by_avi)} videos (8 threads)...",
            flush=True,
        )
        t_v0 = time.perf_counter()
        crops_u8 = np.empty((3, n_passes, img_size, img_size, 3), dtype=np.uint8)
        tasks = [(avi_s, reqs, img_size) for avi_s, reqs in sorted(requests_by_avi.items())]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for items in pool.map(_decode_single_avi, tasks):
                for f_i, p_idx, c_u8 in items:
                    crops_u8[f_i, p_idx] = c_u8
        np.savez_compressed(crops_cache_path, crops_u8=crops_u8)
        print(f"[Real-CityFlow] Decoded & cached {n_passes * 3} crops in {time.perf_counter() - t_v0:.1f}s.", flush=True)

    # Zero-Leakage 2-Fold Out-of-Sample Cross-Fitted ReID Embedding:
    #   Part A (train: S01,S03,S04) <-> Part B (validation: S02,S05)
    flat_u8 = crops_u8.reshape(3 * n_passes, img_size, img_size, 3)
    flat_vids = np.tile(pass_vehs, 3)
    flat_part = np.tile(pass_part, 3)
    anchor_embs = _embed_u8_array(base_model, flat_u8, device)

    mask_a = flat_part == 1
    mask_b = flat_part == 0
    print(
        f"[Real-CityFlow] Zero-leakage 2-fold cross-fitting: Part-A={len(np.unique(flat_vids[mask_a]))} vehs "
        f"<-> Part-B={len(np.unique(flat_vids[mask_b]))} disjoint vehs (0% identity overlap)...",
        flush=True,
    )
    t_ft0 = time.perf_counter()
    net_on_a = _finetune_out_of_sample_fold(
        base_model, flat_u8[mask_a], flat_vids[mask_a], anchor_embs[mask_a], device, seed=42
    )
    net_on_b = _finetune_out_of_sample_fold(
        base_model, flat_u8[mask_b], flat_vids[mask_b], anchor_embs[mask_b], device, seed=43
    )

    flat_oos_embs = np.empty((3 * n_passes, 256), dtype=np.float32)
    flat_oos_embs[mask_b] = _embed_u8_array(net_on_a, flat_u8[mask_b], device)  # Part B embedded by net trained ONLY on Part A
    flat_oos_embs[mask_a] = _embed_u8_array(net_on_b, flat_u8[mask_a], device)  # Part A embedded by net trained ONLY on Part B
    pass_embs = flat_oos_embs.reshape(3, n_passes, 256)

    # Compute genuine out-of-sample cross-camera Rank-1 & mAP on fold 1 (midpoint pass)
    e1 = pass_embs[1]
    sim_cf = e1 @ e1.T
    r1_hits, cf_aps = 0, []
    for i in range(n_passes):
        ord_i = np.argsort(-sim_cf[i])
        ord_i = ord_i[ord_i != i]
        junk = (pass_vehs[ord_i] == pass_vehs[i]) & (pass_cams[ord_i] == pass_cams[i])
        valid = ord_i[~junk]
        matches = (pass_vehs[valid] == pass_vehs[i]).astype(np.float64)
        if matches.sum() == 0:
            continue
        if matches[0] == 1.0:
            r1_hits += 1
        cum = np.cumsum(matches)
        cf_aps.append(float(((cum / (np.arange(len(matches)) + 1.0)) * matches).sum() / matches.sum()))
    cf_r1 = r1_hits / max(1, len(cf_aps))
    cf_map = float(np.mean(cf_aps))
    print(
        f"[Real-CityFlow] Out-of-sample ReID ({n_total_vehs} unseen vehicles, 0% overlap) completed in "
        f"{time.perf_counter() - t_ft0:.1f}s: Rank-1={cf_r1:.4f}, mAP={cf_map:.4f}",
        flush=True,
    )

    lats = np.array([cam_latlon.get(c, (42.5, -90.7))[0] for c in range(40)], dtype=np.float64)
    lons = np.array([cam_latlon.get(c, (42.5, -90.7))[1] for c in range(40)], dtype=np.float64)
    lat0, lon0 = float(np.median(lats)), float(np.median(lons))
    lat_m = (lats - lat0) * 111_132.0
    lon_m = (lons - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    coords = np.stack([lon_m, lat_m], axis=1).astype(np.float32)

    from portolan.profiles_and_topology import (
        build_zero_gt_cityflow_edges,
        compute_empirical_profile_stats as _cal_profile_stats,
        load_real_benchmark_stream,
    )
    eu_arr, ev_arr, ef_arr, ed_arr = build_zero_gt_cityflow_edges(coords=coords, v_free_mps=13.9)

    cohorts = {}
    for f_i in range(3):
        order = np.argsort(pass_times[f_i], kind="mergesort")
        cohorts[f"fold_{f_i}_snap_time"] = pass_times[f_i][order]
        cohorts[f"fold_{f_i}_snap_veh"] = pass_vehs[order]
        cohorts[f"fold_{f_i}_snap_cam"] = pass_cams[order]
        cohorts[f"fold_{f_i}_embeddings"] = pass_embs[f_i][order]

    cf_data = {
        "n_cam": np.int32(40),
        "n_veh": np.int32(n_total_vehs),
        "coords": coords,
        "edges_u": eu_arr,
        "edges_v": ev_arr,
        "edges_fft": ef_arr,
        "edges_dist": ed_arr,
        **cohorts,
    }
    return cf_data, {"cityflow_oos_rank1": round(cf_r1, 4), "cityflow_oos_mAP": round(cf_map, 4)}


def recalibrate_profiles_from_npz(data_dir: Path) -> dict:
    """Fit CityFlow-Seed and VeRi-Seed profiles strictly on disjoint calibration splits (`split='cal'`)."""
    from portolan.profiles_and_topology import (
        compute_empirical_profile_stats,
        load_real_benchmark_stream,
    )

    cf_cal = load_real_benchmark_stream("CityFlow-Seed", fold_idx=1, data_dir=data_dir, split="cal")
    vr_cal = load_real_benchmark_stream("VeRi-Seed", fold_idx=1, data_dir=data_dir, split="cal")
    cf_eval = load_real_benchmark_stream("CityFlow-Seed", fold_idx=1, data_dir=data_dir, split="eval")
    vr_eval = load_real_benchmark_stream("VeRi-Seed", fold_idx=1, data_dir=data_dir, split="eval")

    assert len(set(cf_cal["raw_veh_ids"]) & set(cf_eval["raw_veh_ids"])) == 0, "CityFlow cal/eval vehicle leak!"
    assert len(set(vr_cal["raw_veh_ids"]) & set(vr_eval["raw_veh_ids"])) == 0, "VeRi cal/eval trip leak!"

    cf_stats = compute_empirical_profile_stats(cf_cal, "CityFlow-Seed")
    veri_stats = compute_empirical_profile_stats(vr_cal, "VeRi-Seed")

    existing_reid_meta = {}
    prof_path = data_dir / "real_seed_profiles.json"
    if prof_path.exists():
        try:
            existing_reid_meta = json.loads(prof_path.read_text()).get("reid_model", {})
        except Exception:
            existing_reid_meta = {}

    summary = {
        "reid_model": existing_reid_meta,
        "calibration_protocol": {
            "CityFlow-Seed": {
                "cal_cameras": "c001..c009 (cam < 9, S01+S02)",
                "eval_cameras": "c010..c040 (cam >= 9, S03+S04+S05, re-indexed 0..30)",
                "cal_vehicles": int(len(cf_cal["raw_veh_ids"])),
                "eval_vehicles": int(len(cf_eval["raw_veh_ids"])),
                "cal_passes": int(cf_cal["total_snapshots"]),
                "eval_passes": int(cf_eval["total_snapshots"]),
                "vehicle_overlap": 0,
                "v_free_mps": 13.9,
            },
            "VeRi-Seed": {
                "cal_trips": "veh_id % 2 == 0",
                "eval_trips": "veh_id % 2 == 1",
                "cal_trip_count": int(len(vr_cal["raw_veh_ids"])),
                "eval_trip_count": int(len(vr_eval["raw_veh_ids"])),
                "cal_passes": int(vr_cal["total_snapshots"]),
                "eval_passes": int(vr_eval["total_snapshots"]),
                "trip_overlap": 0,
                "v_free_mps": 15.0,
            },
        },
        "CityFlow-Seed": cf_stats,
        "VeRi-Seed": veri_stats,
    }
    prof_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    default_data_out = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description="Extract real VeRi-776 and CityFlowV2 ReID trajectory streams.")
    parser.add_argument("--veri-dir", type=Path, default=Path(os.environ.get("VERI_RAW_DIR", "./raw_datasets/VeRi")))
    parser.add_argument("--cityflow-dir", type=Path, default=Path(os.environ.get("CITYFLOW_RAW_DIR", "./raw_datasets/AIC22_Track1_MTMC_Tracking")))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("PORTOLAN_RAW_CACHE", "./raw_datasets/cache")))
    parser.add_argument("--out-dir", type=Path, default=default_data_out)
    parser.add_argument(
        "--recalibrate-only",
        action="store_true",
        help="Re-fit out-of-sample calibration seed profiles directly from bundled .npz files.",
    )
    args = parser.parse_args()

    if args.recalibrate_only:
        summary = recalibrate_profiles_from_npz(args.out_dir)
        print(json.dumps(summary, indent=2), flush=True)
        return

    t0 = time.perf_counter()
    device = select_best_device()
    print(f"[Extract] Using GPU device: {device}", flush=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, reid_meta = train_or_load_reid_model(args.veri_dir, args.cache_dir, device)
    veri_data = extract_veri776_real_cohorts(args.veri_dir, model, device, img_size=reid_meta["input_resolution"])
    cf_data, cf_reid_meta = extract_cityflow_real_cohorts(
        args.cityflow_dir, args.cache_dir, model, device, img_size=reid_meta["input_resolution"]
    )
    reid_meta.update(cf_reid_meta)

    out_dirs = [args.out_dir]
    if args.cache_dir.resolve() != args.out_dir.resolve():
        out_dirs.append(args.cache_dir)
    for out_d in out_dirs:
        np.savez_compressed(out_d / "real_veri776_stream.npz", **veri_data)
        np.savez_compressed(out_d / "real_cityflow_stream.npz", **cf_data)

    summary = recalibrate_profiles_from_npz(args.out_dir)
    summary["reid_model"] = reid_meta
    summary["extraction_elapsed_sec"] = round(time.perf_counter() - t0, 2)
    print("\n=== Empirical Out-of-Sample Calibration Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    for out_d in out_dirs:
        (out_d / "real_seed_profiles.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[Done] Saved real streams and empirical profiles in {time.perf_counter() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
