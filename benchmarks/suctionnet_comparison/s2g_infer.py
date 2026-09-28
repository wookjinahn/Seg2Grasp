"""Run Seg2Grasp on GraspNet frames and dump predictions in suctionnetAPI format.

For each frame we produce a scene-wide set of scored suction candidates so the
official metric (precision over top-k) is meaningful:

  - rank #1  = Seg2Grasp's ACTUAL committed grasp (its real policy: pick the most
               elevated object via select_target, then the best suction point via
               estimate_suction_point). Its score is boosted above all others so
               AP_top1 reflects exactly the grasp the robot would execute.
  - ranks 2+ = every other scored candidate from suction_candidates()+
               score_candidates() run over EACH segmented object. This gives the
               top-k richness the metric rewards, using Seg2Grasp's own scoring.

Output per frame: <save_dir>/<split>/scene_%04d/<camera>/suction/%04d.npz with
key 'arr_0' = float32 [N,7] = [score, dir(3), translation_m(3)], translation in
METRES, camera frame; direction = unit surface normal facing the camera (z<0).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                            # noqa: E402
sys.path.insert(0, config.REPO_ROOT)  # import the Seg2Grasp package from this repo

from graspnet_loader import load_frame                                   # noqa: E402
from seg2grasp.segmentation.segmenter import Segmenter                   # noqa: E402
from seg2grasp.grasping.target_selection import select_target           # noqa: E402
from seg2grasp.grasping.suction_planner import (                        # noqa: E402
    estimate_suction_point, suction_candidates, score_candidates)


def split_for_scene(scene_id):
    if 100 <= scene_id < 130:
        return "test_seen"
    if 130 <= scene_id < 160:
        return "test_similar"
    return "test_novel"


def object_candidates(surface_mm, vacuum_radius, z_near, z_far, threshold,
                      downsample, w_angle, w_dist, w_count):
    """All scored Seg2Grasp candidates for one object's masked surface (mm)."""
    surface = surface_mm[(surface_mm[:, 2] > z_near) & (surface_mm[:, 2] < z_far)]
    surface = surface.reshape(-1, 3)
    if len(surface) < 10:
        return []
    surface = surface[::max(int(downsample), 1)]
    centroid = surface.mean(axis=0)
    cands = suction_candidates(surface, vacuum_radius=vacuum_radius, flat_tol=threshold)
    return score_candidates(cands, centroid, w_angle, w_dist, w_count)


_DUMMY = np.array([[0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5]], dtype=np.float32)


def _rows_to_arr(rows):
    arr = np.zeros((len(rows), 7), dtype=np.float32)
    for i, (score, nrm, pt) in enumerate(rows):
        arr[i] = [score, nrm[0], nrm[1], nrm[2], pt[0], pt[1], pt[2]]
    return arr


def infer_frame(segmenter, rgb, depth_mm, pc_mm, args):
    """Return (committed_arr, bestscore_arr) for one frame.

    Both share the same scene-wide candidate pool (so AP_top50 is ~identical);
    they differ only in rank-1:
      - committed_arr : row0 = Seg2Grasp's actual selected-target grasp, score
                        boosted above all candidates (real execution policy).
      - bestscore_arr : candidates only, so top-1 = the highest Seg2Grasp-scored
                        candidate anywhere (seal-aware; isolates planner quality).
    """
    masks, bboxes, _ = segmenter.segment(rgb, depth_mm)

    # committed grasp: most-elevated target + its best suction point
    committed = None
    if masks is not None and len(masks) > 0:
        target = select_target(masks, bboxes, pc_mm, rgb_img=rgb)
        if target is not None:
            pose = estimate_suction_point(
                target.pc_crop, mask=target.mask_crop,
                vacuum_radius=args.vacuum_radius, z_near=args.z_near,
                z_far=args.z_far, threshold=args.threshold,
                rng=np.random.default_rng(0))
            if pose is not None:
                committed = (pose.point / 1000.0, pose.normal)

    # scene-wide candidates over every object
    rows = []
    if masks is not None:
        for m in masks:
            surface = pc_mm[np.asarray(m, dtype=bool)]
            for c in object_candidates(
                    surface, args.vacuum_radius, args.z_near, args.z_far,
                    args.threshold, args.downsample,
                    args.w_angle, args.w_dist, args.w_count):
                rows.append((c["score"], c["normal"], c["point"] / 1000.0))

    bestscore_arr = _rows_to_arr(rows) if rows else _DUMMY.copy()

    if committed is not None:
        pt, nrm = committed
        max_score = max((r[0] for r in rows), default=0.0)
        top = np.array([[max_score + 1.0, nrm[0], nrm[1], nrm[2], pt[0], pt[1], pt[2]]],
                       dtype=np.float32)
        committed_arr = np.concatenate([top, bestscore_arr], axis=0) if rows else top
    else:
        committed_arr = bestscore_arr.copy()

    return committed_arr, bestscore_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=config.GRASPNET_ROOT)
    ap.add_argument("--save-dir", required=True,
                    help="base dump dir; writes '<save-dir>_committed' and '<save-dir>_bestscore'")
    ap.add_argument("--camera", default=config.CAMERA)
    ap.add_argument("--scenes", default="100,101", help="comma list of scene ids")
    ap.add_argument("--frames", default="all", help="'all' (0-255) or comma list")
    ap.add_argument("--vacuum-radius", type=float, default=30.0)
    ap.add_argument("--z-near", type=float, default=100.0)
    ap.add_argument("--z-far", type=float, default=2000.0)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--w-angle", type=float, default=0.5)
    ap.add_argument("--w-dist", type=float, default=0.2)
    ap.add_argument("--w-count", type=float, default=0.3)
    args = ap.parse_args()

    scenes = [int(s) for s in args.scenes.split(",")]
    frames = range(256) if args.frames == "all" else [int(f) for f in args.frames.split(",")]

    segmenter = Segmenter()
    for scene_id in scenes:
        split = split_for_scene(scene_id)
        out_dirs = {}
        for variant in ("committed", "bestscore"):
            d = os.path.join(f"{args.save_dir}_{variant}", split,
                             f"scene_{scene_id:04d}", args.camera, "suction")
            os.makedirs(d, exist_ok=True)
            out_dirs[variant] = d
        for frame in frames:
            rgb, depth_mm, pc_mm = load_frame(args.dataset_root, scene_id, args.camera, frame)
            committed_arr, bestscore_arr = infer_frame(segmenter, rgb, depth_mm, pc_mm, args)
            np.savez(os.path.join(out_dirs["committed"], f"{frame:04d}.npz"), committed_arr)
            np.savez(os.path.join(out_dirs["bestscore"], f"{frame:04d}.npz"), bestscore_arr)
            print(f"scene {scene_id} frame {frame:04d}: {len(bestscore_arr)} candidates", flush=True)


if __name__ == "__main__":
    sys.exit(main())
