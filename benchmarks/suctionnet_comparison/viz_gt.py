"""Visualize Seg2Grasp vs SuctionNet suctions colored by GROUND-TRUTH quality.

Each shown suction is scored with the official suctionnetAPI sub-scores against
the posed GT models (smoothness/seal x wrench, 0 if colliding). Points are
colored red->green by that GT product score (red=fail, green=good seal), so you
see WHICH predicted suctions the benchmark actually considers successful. Top-1
is starred and outlined by whether it passes threshold 0.2.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import scipy.io as scio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
# suctionnetAPI must be pip-installed (see README) — no sys.path hack.
from suctionnetAPI.suctionnet_eval import SuctionNetEval
from suctionnetAPI.utils.eval_utils import (
    get_suction_score, transform_points, create_table_points,
    collision_detection, compute_closest_points, voxel_sample_points)

API_ROOT = config.GRASPNET_ROOT      # for SuctionNetEval (expects root/scenes,models,...)
ROOT = config.SCENES_DIR             # for direct rgb/meta reads (root/scenes/scene_XXXX)
PREDS = config.PREDS_ROOT
OUT = os.path.join(config.VIZ_ROOT, "viz_gt")
CAM = config.CAMERA
THR = 0.2

_model_cache = {}


def intrinsics(scene_id, frame):
    m = scio.loadmat(os.path.join(ROOT, f"scene_{scene_id:04d}", CAM, "meta", f"{frame:04d}.mat"))
    K = m["intrinsic_matrix"]
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def project(pts_m, fx, fy, cx, cy):
    z = np.clip(pts_m[:, 2], 1e-6, None)
    return np.stack([pts_m[:, 0] * fx / z + cx, pts_m[:, 1] * fy / z + cy], axis=-1)


def scene_models(ev, scene_id):
    if scene_id not in _model_cache:
        models, dense, objs = ev.get_scene_models(scene_id, ann_id=0)
        sampled = [voxel_sample_points(m, 0.005) for m in models]
        _model_cache[scene_id] = (dict(zip(objs, sampled)), dict(zip(objs, dense)))
    return _model_cache[scene_id]


def gt_scores(ev, scene_id, frame, arr):
    """Return prod score per suction row (seal*wrench, 0 if colliding)."""
    sampled_by_obj, dense_by_obj = scene_models(ev, scene_id)
    obj_list, pose_list, camera_pose, align_mat = ev.get_model_poses(scene_id, frame)

    model_trans, dense_trans, seg = [], [], []
    for i, obj in enumerate(obj_list):
        mt = transform_points(sampled_by_obj[obj], pose_list[i])
        model_trans.append(mt)
        dense_trans.append(transform_points(dense_by_obj[obj], pose_list[i]))
        seg.append(i * np.ones(len(mt), np.int32))
    scene = np.concatenate(model_trans); seg = np.concatenate(seg)
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.01)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
    scene_tab = np.concatenate([scene, table_trans])

    pts = arr[:, 4:7].astype(np.float64)
    assign = seg[compute_closest_points(pts, scene)]
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        obj = int(assign[i])
        row = arr[i].astype(np.float64)
        coll = collision_detection([row[np.newaxis, :]], [model_trans[obj]], scene_tab, outlier=0.05)[0]
        if len(coll) and bool(coll[0]):
            continue
        s, w = get_suction_score(row, dense_trans[obj], align_mat, camera_pose)
        out[i] = float(s) * float(w)
    return out


def gt_color(prod):
    t = float(np.clip(prod / 0.4, 0, 1))
    return (0, int(255 * t), int(255 * (1 - t)))  # BGR red->green


def draw_panel(rgb, arr, prods, fx, fy, cx, cy, label, topk, top1_is_row0):
    img = rgb.copy()
    order = np.argsort(-arr[:, 0])
    top1 = 0 if top1_is_row0 else int(order[0])
    keep = order[:topk]
    uv = project(arr[:, 4:7], fx, fy, cx, cy)
    for i in keep:
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 6, gt_color(prods[i]), -1)
            cv2.circle(img, (u, v), 6, (25, 25, 25), 1)
    u0, v0 = project(arr[top1, 4:7][np.newaxis], fx, fy, cx, cy)[0]
    tip = project((arr[top1, 4:7] + arr[top1, 1:4] * 0.04)[np.newaxis], fx, fy, cx, cy)[0]
    u0, v0 = int(round(u0)), int(round(v0))
    passed = prods[top1] >= THR
    if 0 <= u0 < img.shape[1] and 0 <= v0 < img.shape[0]:
        cv2.arrowedLine(img, (u0, v0), (int(round(tip[0])), int(round(tip[1]))), (0, 255, 255), 3, tipLength=0.3)
        cv2.drawMarker(img, (u0, v0), (255, 255, 255), cv2.MARKER_STAR, 36, 4)
        cv2.drawMarker(img, (u0, v0), (0, 200, 0) if passed else (0, 0, 220), cv2.MARKER_STAR, 36, 2)
    npass = int((prods[keep] >= THR).sum())
    bar = np.full((46, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, f"{label}  top-1 GT={prods[top1]:.3f} {'PASS' if passed else 'FAIL'}  |  {npass}/{len(keep)} shown pass thr{THR}",
                (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (240, 240, 240), 2)
    return np.vstack([bar, img])


def compare_frame(ev, scene_id, frame, topk):
    rgb = cv2.imread(os.path.join(ROOT, f"scene_{scene_id:04d}", CAM, "rgb", f"{frame:04d}.png"))
    fx, fy, cx, cy = intrinsics(scene_id, frame)
    ns = np.load(os.path.join(PREDS, "normal_std", "test_seen", f"scene_{scene_id:04d}", CAM, "suction", f"{frame:04d}.npz"))["arr_0"]
    sg = np.load(os.path.join(PREDS, "seg2grasp_committed", "test_seen", f"scene_{scene_id:04d}", CAM, "suction", f"{frame:04d}.npz"))["arr_0"]
    # score only the top-K we draw (cheaper)
    ns_k = np.argsort(-ns[:, 0])[:topk]; sg_k = np.argsort(-sg[:, 0])[:max(topk, 1)]
    ns_k = np.union1d(ns_k, [int(np.argmax(ns[:, 0]))])
    sg_k = np.union1d(sg_k, [0])
    ns_p = np.zeros(len(ns)); ns_p[ns_k] = gt_scores(ev, scene_id, frame, ns[ns_k])
    sg_p = np.zeros(len(sg)); sg_p[sg_k] = gt_scores(ev, scene_id, frame, sg[sg_k])
    left = draw_panel(rgb, ns, ns_p, fx, fy, cx, cy, "SuctionNet normal_std", topk, False)
    right = draw_panel(rgb, sg, sg_p, fx, fy, cx, cy, "Seg2Grasp (committed)", topk, True)
    gap = np.full((left.shape[0], 8, 3), 30, np.uint8)
    combo = np.hstack([left, gap, right])
    hdr = np.full((40, combo.shape[1], 3), 20, np.uint8)
    cv2.putText(hdr, f"scene_{scene_id:04d} frame {frame:04d}  |  GT quality: red=fail  green=good seal (seal x wrench); star=top-1",
                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1)
    return np.vstack([hdr, combo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="100:0,105:0,110:0,115:0,120:0,101:2")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ev = SuctionNetEval(root=API_ROOT, camera=CAM)
    tiles = []
    for pair in args.pairs.split(","):
        s, f = pair.split(":")
        img = compare_frame(ev, int(s), int(f), args.topk)
        path = os.path.join(OUT, f"gt_scene{int(s):04d}_f{int(f):04d}.png")
        cv2.imwrite(path, img); print("wrote", path, flush=True)
        tiles.append(img)
    w = min(t.shape[1] for t in tiles)
    tiles = [cv2.resize(t, (w, int(t.shape[0] * w / t.shape[1]))) for t in tiles]
    cv2.imwrite(os.path.join(OUT, "gallery_gt.png"), np.vstack(tiles))
    print("wrote", os.path.join(OUT, "gallery_gt.png"))


if __name__ == "__main__":
    main()
