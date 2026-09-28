"""3-panel GT-colored comparison: normal_std | Seg2Grasp committed | bestscore.

Same GT scoring as viz_gt.py (official suctionnetAPI seal x wrench, 0 if
colliding). Adds the Seg2Grasp 'bestscore' variant whose top-1 is the planner's
highest-scored candidate anywhere (seal-aware), next to its real 'committed'
policy (most-elevated target) and the SuctionNet baseline.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_gt import (SuctionNetEval, API_ROOT, ROOT, PREDS, CAM, THR, intrinsics, project,
                    gt_scores, draw_panel)

import config
OUT = os.path.join(config.VIZ_ROOT, "viz_gt3")


def load_pred(variant, scene_id, frame):
    return np.load(os.path.join(PREDS, variant, "test_seen", f"scene_{scene_id:04d}",
                                CAM, "suction", f"{frame:04d}.npz"))["arr_0"]


def scored_panel(ev, rgb, arr, scene_id, frame, fx, fy, cx, cy, label, topk, top1_row0):
    keep = np.argsort(-arr[:, 0])[:topk]
    keep = np.union1d(keep, [0 if top1_row0 else int(np.argmax(arr[:, 0]))])
    prods = np.zeros(len(arr))
    prods[keep] = gt_scores(ev, scene_id, frame, arr[keep])
    return draw_panel(rgb, arr, prods, fx, fy, cx, cy, label, topk, top1_row0)


def compare_frame(ev, scene_id, frame, topk):
    rgb = cv2.imread(os.path.join(ROOT, f"scene_{scene_id:04d}", CAM, "rgb", f"{frame:04d}.png"))
    fx, fy, cx, cy = intrinsics(scene_id, frame)
    panels = [
        scored_panel(ev, rgb, load_pred("normal_std", scene_id, frame), scene_id, frame,
                     fx, fy, cx, cy, "SuctionNet normal_std", topk, False),
        scored_panel(ev, rgb, load_pred("seg2grasp_committed", scene_id, frame), scene_id, frame,
                     fx, fy, cx, cy, "Seg2Grasp committed (elevated target)", topk, True),
        scored_panel(ev, rgb, load_pred("seg2grasp_bestscore", scene_id, frame), scene_id, frame,
                     fx, fy, cx, cy, "Seg2Grasp bestscore (top planner cand)", topk, False),
    ]
    gap = np.full((panels[0].shape[0], 8, 3), 30, np.uint8)
    combo = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    hdr = np.full((40, combo.shape[1], 3), 20, np.uint8)
    cv2.putText(hdr, f"scene_{scene_id:04d} frame {frame:04d}  |  GT quality: red=fail  green=good seal; star=top-1 (green=pass thr{THR})",
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
        path = os.path.join(OUT, f"gt3_scene{int(s):04d}_f{int(f):04d}.png")
        cv2.imwrite(path, img); print("wrote", path, flush=True)
        tiles.append(img)
    w = min(t.shape[1] for t in tiles)
    tiles = [cv2.resize(t, (w, int(t.shape[0] * w / t.shape[1]))) for t in tiles]
    cv2.imwrite(os.path.join(OUT, "gallery_gt3.png"), np.vstack(tiles))
    print("wrote", os.path.join(OUT, "gallery_gt3.png"))


if __name__ == "__main__":
    main()
