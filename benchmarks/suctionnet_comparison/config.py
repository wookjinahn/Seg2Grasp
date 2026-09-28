"""Portable paths for the SuctionNet-comparison benchmark.

Everything resolves relative to this file / the repo, or from environment
variables, so the benchmark moves between machines without code edits:

    GRASPNET_ROOT   standard GraspNet/SuctionNet root that CONTAINS
                    scenes/, models/, dense_point_clouds/  (default: <bench>/graspnet_dataset)
    S2G_BENCH_PREDS where prediction dumps are written/read  (default: <bench>/preds_full)
    S2G_BENCH_VIZ   where rendered figures go                (default: <bench>/viz_out)
    GRASPNET_CAMERA realsense | kinect                       (default: realsense)

The GraspNet dataset itself is large and NOT committed — point GRASPNET_ROOT at
wherever it lives on the current machine (see README). The Seg2Grasp package is
imported from this repo (two levels up), so no PYTHONPATH juggling is needed.
"""
import os

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BENCH_DIR))  # <repo>/benchmarks/suctionnet_comparison -> <repo>

GRASPNET_ROOT = os.environ.get("GRASPNET_ROOT", os.path.join(BENCH_DIR, "graspnet_dataset"))
SCENES_DIR = os.path.join(GRASPNET_ROOT, "scenes")       # standard GraspNet layout
PREDS_ROOT = os.environ.get("S2G_BENCH_PREDS", os.path.join(BENCH_DIR, "preds_full"))
VIZ_ROOT = os.environ.get("S2G_BENCH_VIZ", os.path.join(BENCH_DIR, "viz_out"))
CAMERA = os.environ.get("GRASPNET_CAMERA", "realsense")
