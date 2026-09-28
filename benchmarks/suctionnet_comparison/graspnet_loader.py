"""Load a GraspNet/SuctionNet-1Billion frame into what Seg2Grasp expects.

GraspNet gives depth as a 16-bit PNG in millimetres (factor_depth=1000) plus a
camera intrinsic matrix per frame (meta .mat / camK.npy). Seg2Grasp wants:
    rgb   [H,W,3] BGR uint8
    depth [H,W]   metric depth (mm)
    pc    [H,W,3] organized point cloud (mm)

so we back-project the depth into an organized cloud (kept in mm, NOT metres,
because Seg2Grasp's whole planner is tuned in mm). The eval later needs metres,
which the inference script converts at dump time.
"""
import os

import cv2
import numpy as np
import scipy.io as scio


def load_intrinsics(scene_dir, camera, frame):
    """Return (fx, fy, cx, cy) for a frame from meta/*.mat (fallback camK.npy)."""
    meta_path = os.path.join(scene_dir, camera, "meta", f"{frame:04d}.mat")
    if os.path.exists(meta_path):
        K = scio.loadmat(meta_path)["intrinsic_matrix"]
    else:
        K = np.load(os.path.join(scene_dir, camera, "camK.npy"))
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def organized_cloud_mm(depth_mm, fx, fy, cx, cy):
    """Back-project a metric depth map (mm) to an organized cloud [H,W,3] (mm)."""
    h, w = depth_mm.shape
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_mm.astype(np.float32)
    x = (xmap - cx) * z / fx
    y = (ymap - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def load_frame(dataset_root, scene_id, camera, frame):
    """Load one GraspNet frame.

    Returns (rgb_bgr [H,W,3] uint8, depth_mm [H,W] float32, pc_mm [H,W,3] float32).
    ``dataset_root`` is a standard GraspNet root containing ``scenes/scene_XXXX/``.
    """
    scene_dir = os.path.join(dataset_root, "scenes", f"scene_{scene_id:04d}")
    rgb = cv2.imread(os.path.join(scene_dir, camera, "rgb", f"{frame:04d}.png"))
    depth = cv2.imread(os.path.join(scene_dir, camera, "depth", f"{frame:04d}.png"),
                       cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError(f"missing rgb/depth for scene {scene_id} frame {frame}")
    depth_mm = depth.astype(np.float32)  # factor_depth=1000 -> raw png value IS mm
    fx, fy, cx, cy = load_intrinsics(scene_dir, camera, frame)
    pc_mm = organized_cloud_mm(depth_mm, fx, fy, cx, cy)
    return rgb, depth_mm, pc_mm
