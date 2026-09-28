"""Diagnose why Seg2Grasp's committed grasp scores AP_top1=0 on scene 100.

For each frame we take row 0 of the Seg2Grasp prediction (the committed grasp),
assign it to the nearest GT object, and print the exact eval sub-scores:
smoothness(seal), wrench, collision, and distance from the grasp point to the
nearest GT surface point (off-surface check). We contrast scene 100 vs 101 and,
for reference, score normal_std's own top-1 on the same frames.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
# suctionnetAPI must be pip-installed (see README) — no sys.path hack.
from suctionnetAPI.suctionnet_eval import SuctionNetEval
from suctionnetAPI.utils.eval_utils import (
    get_suction_score, transform_points, create_table_points,
    collision_detection, compute_closest_points, voxel_sample_points)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
ROOT = config.GRASPNET_ROOT           # passed to SuctionNetEval (expects root/scenes,...)
PREDS = config.PREDS_ROOT
CAM = config.CAMERA


def score_one(ev, scene_id, ann_id, suction_row):
    """Return dict of sub-scores for a single suction [score,dir(3),trans_m(3)]."""
    models, dense_models, _ = ev.get_scene_models(scene_id, ann_id=0)
    _, pose_list, camera_pose, align_mat = ev.get_model_poses(scene_id, ann_id)

    model_trans = [transform_points(m, pose_list[i]) for i, m in enumerate(models)]
    dense_trans = [transform_points(m, pose_list[i]) for i, m in enumerate(dense_models)]
    scene = np.concatenate(model_trans, axis=0)
    seg = np.concatenate([i * np.ones(len(m), np.int32) for i, m in enumerate(model_trans)])

    pt = suction_row[4:7]
    obj = int(seg[compute_closest_points(pt[np.newaxis, :], scene)[0]])

    # off-surface distance to the assigned object's dense surface
    d_surf = float(np.linalg.norm(dense_trans[obj] - pt[np.newaxis, :], axis=1).min())

    # collision against scene+table (same as eval)
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.01)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
    scene_tab = np.concatenate([scene, table_trans])
    coll = collision_detection([suction_row[np.newaxis, :]], [model_trans[obj]], scene_tab, outlier=0.05)[0]
    collided = bool(coll[0]) if len(coll) else False

    if collided:
        smooth, wrench = 0.0, 0.0
    else:
        smooth, wrench = get_suction_score(suction_row, dense_trans[obj], align_mat, camera_pose)
    return dict(obj=obj, d_surf_mm=d_surf * 1000, collided=collided,
                smooth=float(smooth), wrench=float(wrench), prod=float(smooth * wrench))


def main():
    ev = SuctionNetEval(root=ROOT, camera=CAM)
    for scene_id in (100, 101):
        print(f"\n===== scene {scene_id} =====")
        for method in ("seg2grasp", "normal_std"):
            split = "test_seen"
            print(f"  --- {method} (row0=committed for seg2grasp; top-by-score for normal_std) ---")
            passes = 0
            for ann in range(16):
                p = os.path.join(PREDS, method, split, f"scene_{scene_id:04d}", CAM, "suction", f"{ann:04d}.npz")
                arr = np.load(p)["arr_0"]
                row = arr[0] if method == "seg2grasp" else arr[int(np.argmax(arr[:, 0]))]
                r = score_one(ev, scene_id, ann, row.astype(np.float64))
                passes += int(r["prod"] >= 0.2)
                if ann < 6:
                    print(f"    f{ann:02d} obj={r['obj']:2d} dsurf={r['d_surf_mm']:5.1f}mm "
                          f"coll={int(r['collided'])} smooth={r['smooth']:.3f} "
                          f"wrench={r['wrench']:.3f} prod={r['prod']:.3f}")
            print(f"    -> frames passing thr0.2: {passes}/16")


if __name__ == "__main__":
    main()
