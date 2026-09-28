"""Pilot evaluation with the official suctionnetAPI metric over a frame subset.

suctionnetAPI's eval_scene hardcodes range(256); we subclass it to evaluate an
arbitrary frame list (pilot = 0..15) with the body otherwise IDENTICAL to the
stock implementation. suction_nms is unavailable, so SuctionGroup.nms is
monkey-patched to a faithful numpy translation-only NMS (the API calls
nms(0.02, 181deg); 181deg means rotation never filters -> pure 2cm dedup).

Aggregation matches eval_seen: res shape (n_scene, n_frame, 50, 4);
AP_top50 = mean(res[...,:50,:]), AP_top1 = mean(res[...,:1,:]).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from suctionnetAPI import SuctionNetEval
from suctionnetAPI.suction import SuctionGroup
from suctionnetAPI.utils.eval_utils import (create_table_points, transform_points,
                                            voxel_sample_points, eval_suction)


def nms_numpy(arr, translation_thresh=0.02):
    """Greedy score-descending NMS, suppress neighbours within translation_thresh (m)."""
    if len(arr) == 0:
        return arr
    trans = arr[:, 4:7]
    order = np.argsort(-arr[:, 0])
    suppressed = np.zeros(len(arr), dtype=bool)
    keep = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        d = np.linalg.norm(trans - trans[idx], axis=1)
        suppressed |= (d < translation_thresh)
    return arr[keep]


# faithful fallback for the missing suction_nms compiled op
SuctionGroup.nms = lambda self, t=0.02, r=None: SuctionGroup(nms_numpy(self.suction_group_array, t))


class PilotEval(SuctionNetEval):
    def eval_scene_frames(self, scene_id, split, dump_folder, frames):
        """Body identical to stock eval_scene, but iterates `frames` not range(256)."""
        threshold_list = [0.2, 0.4, 0.6, 0.8]
        TOP_K = 50
        model_list, dense_model_list, _ = self.get_scene_models(scene_id, ann_id=0)
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.01)
        model_sampled_list = [voxel_sample_points(m, 0.005) for m in model_list]

        scene_accuracy = []
        for ann_id in frames:
            suction_group = SuctionGroup().from_npy(
                os.path.join(dump_folder, split, 'scene_%04d' % scene_id, self.camera,
                             'suction', '%04d.npz' % ann_id))
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            suction_list, smooth, wrench, coll = eval_suction(
                suction_group, model_sampled_list, dense_model_list, pose_list,
                align_mat, camera_pose, table=table_trans)
            suction_list = [x for x in suction_list if len(x[0]) != 0]
            smooth = [x for x in smooth if len(x) != 0]
            wrench = [x for x in wrench if len(x) != 0]
            coll = [x for x in coll if len(x) != 0]
            if len(suction_list) == 0:
                scene_accuracy.append(np.zeros((TOP_K, len(threshold_list))))
                continue
            suction_list = np.concatenate(suction_list)
            smooth = np.concatenate(smooth)
            wrench = np.concatenate(wrench)
            coll = np.concatenate(coll)

            indices = np.argsort(-suction_list[:, 0])
            smooth, wrench = smooth[indices], wrench[indices]

            acc = np.zeros((TOP_K, len(threshold_list)))
            for ti, thr in enumerate(threshold_list):
                for k in range(TOP_K):
                    n = min(k + 1, len(wrench))
                    acc[k, ti] = np.sum(((wrench[:n] * smooth[:n]) >= thr).astype(np.float32)) / (k + 1)
            scene_accuracy.append(acc)
        return np.array(scene_accuracy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=config.GRASPNET_ROOT)
    ap.add_argument("--camera", default=config.CAMERA)
    ap.add_argument("--dump", required=True, help="prediction dump_folder")
    ap.add_argument("--scenes", default="100,101")
    ap.add_argument("--frames", default="all")
    ap.add_argument("--split", default="test_seen")
    args = ap.parse_args()
    scenes = [int(s) for s in args.scenes.split(",")]
    frames = list(range(256)) if args.frames == "all" else [int(f) for f in args.frames.split(",")]

    ev = PilotEval(root=args.root, camera=args.camera)
    per_scene = []
    for s in scenes:
        acc = ev.eval_scene_frames(s, args.split, args.dump, frames)
        per_scene.append(acc)
        print("scene %d: AP_top50=%.4f AP_top1=%.4f" % (
            s, float(np.mean(acc[:, :50, :])), float(np.mean(acc[:, :1, :]))), flush=True)
    res = np.array(per_scene)  # (n_scene, n_frame, 50, 4)
    ap_top50 = float(np.mean(res[:, :, :50, :]))
    ap_top1 = float(np.mean(res[:, :, :1, :]))
    t1 = [float(np.mean(res[:, :, :1, i])) for i in range(4)]
    print("\n==== %s ====" % os.path.basename(args.dump.rstrip("/")))
    print("AP_top50 = %.4f" % ap_top50)
    print("AP_top1  = %.4f" % ap_top1)
    print("AP_top1 per-threshold [0.2,0.4,0.6,0.8] = [%s]" % ", ".join("%.4f" % x for x in t1))


if __name__ == "__main__":
    main()
