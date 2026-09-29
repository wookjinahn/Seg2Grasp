"""SuctionNet learned-model (DeepLabV3+/ConvNet RGB-D) inference on GraspNet frames.

Portable rewrite of suctionnet-baseline/neural_network/inference.py: same model
zoo, preprocessing (RGB + depth -> 4-channel input for deeplabv3plus_resnet101),
heatmap grid-sampling and normal estimation, same dump format
(save_root/<split>/scene_%04d/<camera>/suction/%04d.npz, key arr_0 = [N,7] =
score,dir(3),trans_m(3)) as normal_std_infer.py, so it drops into the same
full_eval.py. Only paths/CLI are parametrized (via config.py) and dead code
(interactive log-dir prompts) removed; the model code itself is vendored
unmodified (bar the torch.hub import fix) under third_party/suctionnet_neural_network/
— see that directory's README for attribution.

Needs a checkpoint (NOT committed — see ../../.ai/DECISIONS.md 2026-09-28 entry
for the Google-Drive id) and the `suctionnet_eval` conda env (adds open3d).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

sys.path.insert(0, config.NN_THIRD_PARTY)
import DeepLabV3Plus.network as network
import ConvNet

MODEL_MAP = {
    "deeplabv3_resnet50": network.deeplabv3_resnet50,
    "deeplabv3plus_resnet50": network.deeplabv3plus_resnet50,
    "deeplabv3_resnet101": network.deeplabv3_resnet101,
    "deeplabv3plus_resnet101": network.deeplabv3plus_resnet101,
    "deeplabv3_mobilenet": network.deeplabv3_mobilenet,
    "deeplabv3plus_mobilenet": network.deeplabv3plus_mobilenet,
    "convnet_resnet101": ConvNet.convnet_resnet101,
    "deeplabv3plus_resnet101_depth": network.deeplabv3plus_resnet101_depth,
}


class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width, self.height = width, height
        self.fx, self.fy, self.cx, self.cy, self.scale = fx, fy, cx, cy, scale


def uniform_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    return kernel / kernel_size**2


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert depth.shape[0] == camera.height and depth.shape[1] == camera.width
    xmap, ymap = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    points_z = depth
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    return cloud.reshape([-1, 3]) if not organized else cloud


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


def get_suction_from_heatmap(depth_img, heatmap, camera_info):
    suction_scores, idx0, idx1 = grid_sample(heatmap, down_rate=10, topk=1024)
    if depth_img.ndim == 3:
        depth_img = depth_img[..., 0]
    point_cloud = create_point_cloud_from_depth_image(depth_img, camera_info)
    suction_points = point_cloud[idx0, idx1, :]

    point_cloud = point_cloud.reshape(-1, 3)
    pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
    pc_voxel_sampled = pc_o3d.voxel_down_sample(0.003)
    points_sampled = np.array(pc_voxel_sampled.points).astype(np.float32)
    points_sampled = np.concatenate([suction_points, points_sampled], axis=0)
    pc_voxel_sampled.points = o3d.utility.Vector3dVector(points_sampled)
    pc_voxel_sampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
    pc_voxel_sampled.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    pc_voxel_sampled.normalize_normals()
    pc_normals = np.array(pc_voxel_sampled.normals).astype(np.float32)
    suction_normals = pc_normals[:suction_points.shape[0], :]

    suction_arr = np.concatenate(
        [suction_scores[..., np.newaxis], suction_normals, suction_points], axis=-1)
    return suction_arr


def inference_one_view(net, device, model_name, rgb_file, depth_file, meta_file,
                        save_root, split, camera, scene_idx, anno_idx):
    meta = scio.loadmat(meta_file)
    K = meta["intrinsic_matrix"]
    camera_info = CameraInfo(1280, 720, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1000.0)

    rgb = cv2.imread(rgb_file).astype(np.float32) / 255.0
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    rgb, depth = torch.from_numpy(rgb), torch.from_numpy(depth)
    depth = torch.clamp(depth, 0, 1)

    if model_name == "convnet_resnet101":
        depth_ch = depth.unsqueeze(-1).repeat([1, 1, 3])
        rgbd = torch.cat([rgb, depth_ch], dim=-1).unsqueeze(0)
    elif "depth" in model_name:
        rgbd = depth.unsqueeze(-1).unsqueeze(0)
    else:
        rgbd = torch.cat([rgb, depth.unsqueeze(-1)], dim=-1).unsqueeze(0)
    rgbd = rgbd.permute(0, 3, 1, 2).to(device)

    net.eval()
    tic = time.time()
    with torch.no_grad():
        pred = net(rgbd).clamp(0, 1)
    print("inference time:", time.time() - tic, flush=True)

    heatmap = (pred[0, 0] * pred[0, 1]).cpu().unsqueeze(0).unsqueeze(0)
    k_size = 15
    kernel = torch.from_numpy(uniform_kernel(k_size)).unsqueeze(0).unsqueeze(0)
    heatmap = F.conv2d(heatmap, kernel, padding=k_size // 2).squeeze().numpy()

    suctions = get_suction_from_heatmap(depth.numpy(), heatmap, camera_info)

    out_dir = os.path.join(save_root, split, "scene_%04d" % scene_idx, camera, "suction")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "%04d.npz" % anno_idx), suctions)
    print("nn scene %d frame %04d: %d suctions" % (scene_idx, anno_idx, len(suctions)), flush=True)


def split_for_scene(scene_id):
    if 100 <= scene_id < 130:
        return "test_seen"
    if 130 <= scene_id < 160:
        return "test_similar"
    return "test_novel"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deeplabv3plus_resnet101", choices=list(MODEL_MAP))
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--output-stride", type=int, default=16, choices=[8, 16])
    ap.add_argument("--checkpoint-path", default=config.NN_CHECKPOINT)
    ap.add_argument("--dataset-root", default=config.GRASPNET_ROOT)
    ap.add_argument("--save-root", required=True)
    ap.add_argument("--camera", default=config.CAMERA)
    ap.add_argument("--scenes", default="100,101")
    ap.add_argument("--frames", default="all")
    args = ap.parse_args()

    net = MODEL_MAP[args.model](num_classes=args.num_classes, output_stride=args.output_stride)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = nn.DataParallel(net).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded checkpoint epoch:", checkpoint.get("epoch"), flush=True)

    scenes = [int(s) for s in args.scenes.split(",")]
    frames = range(256) if args.frames == "all" else [int(f) for f in args.frames.split(",")]
    for scene_idx in scenes:
        split = split_for_scene(scene_idx)
        base = os.path.join(args.dataset_root, "scenes", "scene_%04d" % scene_idx, args.camera)
        for anno_idx in frames:
            inference_one_view(
                net, device, args.model,
                rgb_file=os.path.join(base, "rgb", "%04d.png" % anno_idx),
                depth_file=os.path.join(base, "depth", "%04d.png" % anno_idx),
                meta_file=os.path.join(base, "meta", "%04d.mat" % anno_idx),
                save_root=args.save_root, split=split, camera=args.camera,
                scene_idx=scene_idx, anno_idx=anno_idx)


if __name__ == "__main__":
    main()
