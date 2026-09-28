"""Side-by-side visualization of Seg2Grasp vs SuctionNet suctions on GraspNet RGB.

For a scene/frame it overlays, per method, the top-K predicted suction points
projected onto the RGB image (colored by the method's own score) and highlights
the top-1 grasp (white star + projected surface-normal arrow). Panels are
stitched left(SuctionNet normal_std) | right(Seg2Grasp committed) with labels.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import scipy.io as scio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

ROOT = config.SCENES_DIR              # direct rgb/meta reads (root/scenes/scene_XXXX)
PREDS = config.PREDS_ROOT
OUT = os.path.join(config.VIZ_ROOT, "viz")
CAM = config.CAMERA


def intrinsics(scene_id, frame):
    m = scio.loadmat(os.path.join(ROOT, f"scene_{scene_id:04d}", CAM, "meta", f"{frame:04d}.mat"))
    K = m["intrinsic_matrix"]
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def project(pts_m, fx, fy, cx, cy):
    """(N,3) camera-frame points in metres -> (N,2) pixel coords."""
    z = np.clip(pts_m[:, 2], 1e-6, None)
    u = pts_m[:, 0] * fx / z + cx
    v = pts_m[:, 1] * fy / z + cy
    return np.stack([u, v], axis=-1)


def score_color(s, smin, smax):
    """Score -> BGR (blue=low, red=high) via a simple viridis-ish ramp."""
    t = 0.0 if smax <= smin else (s - smin) / (smax - smin)
    t = float(np.clip(t, 0, 1))
    # blue(255,0,0 BGR) -> green -> red(0,0,255 BGR)
    if t < 0.5:
        r = 0; g = int(510 * t); b = int(255 * (1 - 2 * t))
    else:
        r = int(255 * (2 * t - 1)); g = int(510 * (1 - t)); b = 0
    return (b, g, r)


def draw_panel(rgb, arr, fx, fy, cx, cy, label, topk, top1_is_row0):
    img = rgb.copy()
    scores = arr[:, 0]
    order = np.argsort(-scores)
    top1_idx = 0 if top1_is_row0 else int(order[0])
    keep = order[:topk]
    smin, smax = float(scores[keep].min()), float(scores[keep].max())

    uv = project(arr[:, 4:7], fx, fy, cx, cy)
    for i in keep:
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 5, score_color(scores[i], smin, smax), -1)
            cv2.circle(img, (u, v), 5, (30, 30, 30), 1)

    # top-1 star + normal arrow
    p = arr[top1_idx, 4:7]
    d = arr[top1_idx, 1:4]
    u0, v0 = project(p[np.newaxis], fx, fy, cx, cy)[0]
    tip = project((p + d * 0.04)[np.newaxis], fx, fy, cx, cy)[0]
    u0, v0 = int(round(u0)), int(round(v0))
    if 0 <= u0 < img.shape[1] and 0 <= v0 < img.shape[0]:
        cv2.arrowedLine(img, (u0, v0), (int(round(tip[0])), int(round(tip[1]))),
                        (0, 255, 255), 3, tipLength=0.3)
        cv2.drawMarker(img, (u0, v0), (255, 255, 255), cv2.MARKER_STAR, 34, 3)
        cv2.drawMarker(img, (u0, v0), (0, 0, 255), cv2.MARKER_STAR, 34, 1)

    bar = np.full((46, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, f"{label}  (top-1 score={arr[top1_idx,0]:.3f}, {len(arr)} cands, showing top {min(topk,len(arr))})",
                (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    return np.vstack([bar, img])


def compare_frame(scene_id, frame, topk):
    rgb = cv2.imread(os.path.join(ROOT, f"scene_{scene_id:04d}", CAM, "rgb", f"{frame:04d}.png"))
    fx, fy, cx, cy = intrinsics(scene_id, frame)
    ns = np.load(os.path.join(PREDS, "normal_std", "test_seen", f"scene_{scene_id:04d}", CAM, "suction", f"{frame:04d}.npz"))["arr_0"]
    sg = np.load(os.path.join(PREDS, "seg2grasp_committed", "test_seen", f"scene_{scene_id:04d}", CAM, "suction", f"{frame:04d}.npz"))["arr_0"]
    left = draw_panel(rgb, ns, fx, fy, cx, cy, "SuctionNet normal_std", topk, top1_is_row0=False)
    right = draw_panel(rgb, sg, fx, fy, cx, cy, "Seg2Grasp (committed)", topk, top1_is_row0=True)
    gap = np.full((left.shape[0], 8, 3), 30, np.uint8)
    combo = np.hstack([left, gap, right])
    hdr = np.full((40, combo.shape[1], 3), 20, np.uint8)
    cv2.putText(hdr, f"scene_{scene_id:04d}  frame {frame:04d}  (realsense)  |  yellow arrow = surface normal, star = top-1",
                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 255), 1)
    return np.vstack([hdr, combo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="100:0,105:0,110:0,115:0,120:0,101:2",
                    help="comma list of scene:frame")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tiles = []
    for pair in args.pairs.split(","):
        s, f = pair.split(":")
        img = compare_frame(int(s), int(f), args.topk)
        path = os.path.join(OUT, f"cmp_scene{int(s):04d}_f{int(f):04d}.png")
        cv2.imwrite(path, img)
        print("wrote", path)
        tiles.append(img)
    w = min(t.shape[1] for t in tiles)
    tiles = [cv2.resize(t, (w, int(t.shape[0] * w / t.shape[1]))) for t in tiles]
    gallery = np.vstack(tiles)
    gpath = os.path.join(OUT, "gallery_compare.png")
    cv2.imwrite(gpath, gallery)
    print("wrote", gpath)


if __name__ == "__main__":
    main()
