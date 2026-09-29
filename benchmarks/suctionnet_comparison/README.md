# Seg2Grasp vs SuctionNet — GraspNet benchmark

Quantitative + qualitative comparison of **Seg2Grasp's analytic suction planner**
against two SuctionNet baselines — the training-free **`normal_std`** analytic
method and the supervised **DeepLabV3+ RGB-D** learned model — on the **GraspNet /
SuctionNet-1Billion** dataset, scored by the **official `suctionnetAPI`** metric
(AP@top-k over seal × wrench, collision-checked).

Everything here is self-contained and path-portable (see `config.py`): clone the
repo on any machine, point `GRASPNET_ROOT` at the dataset, and run.

## Results (test_seen, realsense, 30 scenes × 256 frames)

| Method | AP_top1 | AP_top50 | AP_top1 per-thr [0.2/0.4/0.6/0.8] |
|---|---|---|---|
| SuctionNet normal_std (analytic) | 0.1540 | 0.1010 | 0.315 / 0.188 / 0.095 / 0.018 |
| **SuctionNet DeepLabV3+ (learned RGB-D)** | **0.4625** | **0.2826** | 0.707 / 0.629 / 0.422 / 0.092 |
| **Seg2Grasp (committed)** | **0.2055** | **0.1204** | 0.342 / 0.269 / 0.169 / 0.042 |
| Seg2Grasp (bestscore) | 0.1503 | 0.1188 | 0.251 / 0.203 / 0.119 / 0.029 |

- **Seg2Grasp (committed)** = its real policy: pick the most-elevated object, then
  its best suction point (single executed grasp; the honest AP_top1).
- **Seg2Grasp (bestscore)** = "seal-aware top-1": rank-1 is the planner's
  highest-scored candidate anywhere (isolates planner quality from target choice).
- **normal_std** = SuctionNet's training-free analytic baseline (surface-normal
  smoothness heatmap), its top-1024 candidates.
- **DeepLabV3+ (learned RGB-D)** = SuctionNet's supervised model, pretrained
  realsense weights, evaluated on its own dataset's *seen* objects — the setting
  most favorable to it.

Seg2Grasp (committed) beats normal_std (both training-free) on **both**
metrics; its target selection also beats its own bestscore variant on top-1 →
target selection is an asset. The learned DeepLabV3+ model roughly **doubles**
Seg2Grasp on this in-distribution split, as expected for a supervised model
scored on seen objects — the open question (not yet run) is whether that edge
survives on `test_novel`, which is Seg2Grasp's claimed generalization strength.
Galleries in `results/` show the training-free pair overlaid on the RGB scenes
(GT-colored: red=fail, green=good seal). Full rationale in `../../.ai/DECISIONS.md`.

**Caveats.** (1) `suction_nms` (a compiled op the API imports) is unavailable
upstream, so a pure-numpy 2 cm translation-only NMS is used — exact for the API's
`nms(0.02, 181°)` call (181°>180° disables the rotation gate). (2) Only the
**realsense** camera and **test_seen** split were run — `test_similar`/`test_novel`
are not yet downloaded/evaluated. (3) GraspNet depth PNGs are millimetres
(factor_depth=1000); prediction translations are metres, camera frame.

## Layout

```
config.py               portable paths (GRASPNET_ROOT env var etc.)
graspnet_loader.py      GraspNet depth -> organized mm cloud + intrinsics
s2g_infer.py            Seg2Grasp -> suctionnetAPI dumps (committed + bestscore)
normal_std_infer.py     SuctionNet normal_std baseline inference
policy.py               normal_std suction policy (vendored from suctionnet-baseline)
suctionnet_nn_infer.py  SuctionNet learned model (DeepLabV3+/ConvNet RGB-D) inference
full_eval.py            official eval over full test_seen (eval_seen)
eval_pilot.py           official eval over a frame subset (pilot)
diag_scene100.py        per-frame GT sub-score diagnostic
viz_compare.py          overlay top-k suctions, colored by each method's own score
viz_gt.py               overlay colored by GROUND-TRUTH quality (2-panel)
viz_gt3.py              GT-colored 3-panel (normal_std | committed | bestscore)
run_all_evals.sh        convenience driver (training-free pair only)
envs/                   conda specs (s2g_seg, suctionnet_eval) + pip freezes
third_party/suctionnetAPI/              vendored + numpy-patched official API (see below)
third_party/suctionnet_neural_network/  vendored DeepLabV3Plus/ConvNet model code (see its README)
results/                representative galleries (downscaled)
```

## Setup on a new machine

1. **Clone this repo** (the benchmark lives at `benchmarks/suctionnet_comparison/`).

2. **Create the two conda envs** from the pinned specs:
   ```bash
   conda env create -f envs/s2g_seg.yml           # Seg2Grasp: detectron2 + MSDeformAttn
   conda env create -f envs/suctionnet_eval.yml   # normal_std + official eval
   ```
   `s2g_seg` compiles native CUDA ops (detectron2 `_C`, MSDeformAttn). The `.yml`
   captures package versions, but a fresh machine must rebuild those ops against
   its own CUDA toolchain / GPU arch — see `../../scripts/setup_seg_env.sh` (uv,
   Blackwell) or `../../environment.yml` (the conda/sm_89 build actually used) and
   set `TORCH_CUDA_ARCH_LIST` for your GPU.

3. **Install the vendored suctionnetAPI** (numpy-alias-patched; the numbers above
   also depend on the numpy-NMS fallback baked into `full_eval.py`/`eval_pilot.py`):
   ```bash
   conda run -n suctionnet_eval pip install -e third_party/suctionnetAPI
   ```

4. **Get the dataset** (GraspNet/SuctionNet-1Billion) — `test_seen` scenes,
   `models/`, `dense_point_clouds/`. Arrange as a standard GraspNet root:
   ```
   $GRASPNET_ROOT/scenes/scene_0100 … scene_0129/{realsense,kinect}/{rgb,depth,label,meta,annotations,camera_poses.npy,cam0_wrt_table.npy}
   $GRASPNET_ROOT/models/%03d/nontextured.ply
   $GRASPNET_ROOT/dense_point_clouds/%03d.npz
   ```
   If your scenes are extracted flat (`scene_0100/…` directly), make the `scenes/`
   dir with symlinks: `mkdir scenes && ln -s ../scene_0* scenes/`.
   Then: `export GRASPNET_ROOT=/path/to/graspnet` (default is `./graspnet_dataset`).

5. **Get the Seg2Grasp segmentation checkpoint** (for `s2g_infer.py`):
   `bash ../../scripts/download_weights.sh`.

6. **(Optional) get the learned-model checkpoint** (for `suctionnet_nn_infer.py`) —
   the SuctionNet DeepLabV3+ RGB-D pretrained realsense weights, Google-Drive id
   `18TbctdhpNXEKLYDWFzI9cT1Wnhe-tn9h` (see `../../.ai/DECISIONS.md`). Download and
   place at `checkpoints/realsense-deeplabplus-RGBD`, or point
   `SUCTIONNET_NN_CHECKPOINT` at it. Not committed (~700 MB). Inference also needs
   `open3d` (already in `envs/suctionnet_eval.yml`).

## Run

```bash
# 1) SuctionNet normal_std predictions (env: suctionnet_eval)
conda run -n suctionnet_eval python normal_std_infer.py \
    --save-root preds_full/normal_std --scenes $(seq -s, 100 129) --frames all

# 2) Seg2Grasp predictions — one GPU pass writes _committed and _bestscore (env: s2g_seg)
conda run -n s2g_seg python s2g_infer.py \
    --save-dir preds_full/seg2grasp --scenes $(seq -s, 100 129) --frames all

# 2b) SuctionNet learned model (DeepLabV3+ RGB-D) predictions — needs a GPU + the
#     checkpoint from step 6 (env: suctionnet_eval, has open3d + torch)
conda run -n suctionnet_eval python suctionnet_nn_infer.py \
    --save-root preds_full/suctionnet_nn --scenes $(seq -s, 100 129) --frames all

# 3) Official eval for each dump (env: suctionnet_eval)
for d in normal_std seg2grasp_committed seg2grasp_bestscore suctionnet_nn; do
  conda run -n suctionnet_eval python full_eval.py --dump preds_full/$d --proc 12
done

# 4) Visualize (env: suctionnet_eval; training-free pair only)
conda run -n suctionnet_eval python viz_gt3.py --pairs 100:0,105:0,101:2
```

A quick pilot (a few scenes/frames) uses `eval_pilot.py --frames 0,1,...,15`.

## Attribution

`third_party/suctionnetAPI/` and `policy.py` derive from
[graspnet/suctionnetAPI](https://github.com/graspnet/suctionnetAPI) and
[graspnet/suctionnet-baseline](https://github.com/graspnet/suctionnet-baseline).
Local changes: numpy-3 alias fixes (`np.float/np.int/np.bool`) and the numpy NMS
fallback for the missing `suction_nms` op. `third_party/suctionnet_neural_network/`
and `suctionnet_nn_infer.py` also derive from `graspnet/suctionnet-baseline`
(`neural_network/`) — see that directory's own README for its local patch. See
those projects for their licenses.
