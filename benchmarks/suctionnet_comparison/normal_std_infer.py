"""SuctionNet normal_std baseline inference on GraspNet frames (pilot-parametrized).

Faithful port of suctionnet-baseline/normal_std/inference.py: same estimate_suction
policy (policy.py), same top-1024 grid sampling, same dump format
(save_root/<split>/scene_%04d/<camera>/suction/%04d.npz, key arr_0 = [N,7] =
score,dir(3),trans_m(3)). Only the scene/frame loop and dataset paths are
parametrized so we can run a small pilot; numpy legacy aliases already patched.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from policy import estimate_suction


class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width, self.height = width, height
        self.fx, self.fy, self.cx, self.cy, self.scale = fx, fy, cx, cy, scale


def uniform_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    return kernel / kernel_size**2


def grid_sample(pred_score_map, down_rate=20, topk=512):
    num_row = pred_score_map.shape[0] // down_rate
    num_col = pred_score_map.shape[1] // down_rate
    idx_list = []
    for i in range(num_row):
        for j in range(num_col):
            grid = pred_score_map[i*down_rate:(i+1)*down_rate, j*down_rate:(j+1)*down_rate]
            max_idx = np.argmax(grid)
            max_idx = np.array([max_idx // down_rate, max_idx % down_rate]).astype(np.int32)
            max_idx[0] += i*down_rate
            max_idx[1] += j*down_rate
            idx_list.append(max_idx[np.newaxis, ...])
    idx = np.concatenate(idx_list, axis=0)
    suction_scores = pred_score_map[idx[:, 0], idx[:, 1]]
    sort_idx = np.argsort(suction_scores)[::-1][:topk]
    return suction_scores[sort_idx], idx[:, 0][sort_idx], idx[:, 1][sort_idx]


def split_for_scene(scene_id):
    if 100 <= scene_id < 130:
        return "test_seen"
    if 130 <= scene_id < 160:
        return "test_similar"
    return "test_novel"


def inference(dataset_root, save_root, camera, scene_idx, frames):
    split = split_for_scene(scene_idx)
    for anno_idx in frames:
        base = os.path.join(dataset_root, "scenes", "scene_{:04d}".format(scene_idx), camera)
        depth = cv2.imread(os.path.join(base, "depth", "{:04d}.png".format(anno_idx)),
                           cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        seg_mask = cv2.imread(os.path.join(base, "label", "{:04d}.png".format(anno_idx)),
                              cv2.IMREAD_UNCHANGED).astype(bool)
        meta = scio.loadmat(os.path.join(base, "meta", "{:04d}.mat".format(anno_idx)))
        K = meta["intrinsic_matrix"]
        camera_info = CameraInfo(1280, 720, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1000.0)

        heatmap, normals, point_cloud = estimate_suction(depth, seg_mask, camera_info)

        k_size = 15
        kernel = torch.from_numpy(uniform_kernel(k_size)).unsqueeze(0).unsqueeze(0)
        heatmap = np.pad(heatmap, k_size // 2)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap = F.conv2d(heatmap, kernel).squeeze().numpy()

        suction_scores, idx0, idx1 = grid_sample(heatmap, down_rate=10, topk=1024)
        suction_directions = normals[idx0, idx1, :]
        suction_translations = point_cloud[idx0, idx1, :]
        suction_arr = np.concatenate(
            [suction_scores[..., np.newaxis], suction_directions, suction_translations], axis=-1)

        out_dir = os.path.join(save_root, split, "scene_%04d" % scene_idx, camera, "suction")
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "%04d.npz" % anno_idx), suction_arr)
        print("normal_std scene %d frame %04d: %d suctions" % (scene_idx, anno_idx, len(suction_arr)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=config.GRASPNET_ROOT)
    ap.add_argument("--save-root", required=True)
    ap.add_argument("--camera", default=config.CAMERA)
    ap.add_argument("--scenes", default="100,101")
    ap.add_argument("--frames", default="all")
    args = ap.parse_args()
    scenes = [int(s) for s in args.scenes.split(",")]
    frames = range(256) if args.frames == "all" else [int(f) for f in args.frames.split(",")]
    for scene_idx in scenes:
        inference(args.dataset_root, args.save_root, args.camera, scene_idx, frames)


if __name__ == "__main__":
    main()
