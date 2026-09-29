# Decisions

Append-only log of notable decisions made while working on this repo — the
"why," not the "what" (the "what" belongs in commit messages and diffs). Add a
new entry whenever a non-obvious choice is made that a future agent could
otherwise second-guess or accidentally reverse. Do not edit or delete past
entries; if a decision is later reversed, add a new entry that supersedes it.

Format per entry:

```
## YYYY-MM-DD — Short title

**Context:** what prompted the decision.
**Decision:** what was decided.
**Why:** the reasoning / trade-off.
**Alternatives considered:** (optional) what else was on the table and why it lost.
```

---

## 2026-09-26 — Adopt `.ai/HANDOFF.md` + `.ai/DECISIONS.md` for agent handoff

**Context:** No project convention existed for persisting state or rationale
across agent sessions; each new session had to re-derive context from git
history and code alone.

**Decision:** Track live/working state in `.ai/HANDOFF.md` (current state, next
steps, a checkpoint log) and durable rationale in `.ai/DECISIONS.md` (this
file). Agents should read both before starting work and update `HANDOFF.md` at
every work boundary.

**Why:** Keeps ephemeral "what's in flight" separate from durable "why we did
X," so `DECISIONS.md` stays a stable append-only record while `HANDOFF.md` can
be freely rewritten as state changes.

**Alternatives considered:** A single combined `AGENTS.md`/`NOTES.md` file —
rejected because mixing frequently-rewritten state with an append-only
decision log makes the log harder to trust (edits to fix stale state risk
clobbering old rationale).

---

## 2026-09-26 — Local segmentation env built with conda + sm_89, not the repo's uv + sm_120 script

**Context:** User wanted to test the pipeline and compare it against SuctionNet,
starting by getting Seg2Grasp itself running. This workstation has no `uv`, no
CUDA toolkit/nvcc installed, and an RTX 4060 Ti (Ada Lovelace, compute
capability sm_89) — not the Blackwell (sm_120) GPU the repo's
`scripts/setup_seg_env.sh` is written for. The user explicitly asked to set
this up with conda (conda/miniconda was already present on the machine).

**Decision:** Reimplemented the same install sequence as
`scripts/setup_seg_env.sh` (torch → build deps → detectron2 from source →
MSDeformAttn CUDA op → editable install) using conda instead of uv, with
`TORCH_CUDA_ARCH_LIST=8.9` instead of the script's `12.0`, into a new env named
`s2g_seg`. Along the way, fixed three toolchain issues purely at the
env/package level (no repo code touched):
1. `pytorch-cuda=12.1` pulled in `mkl==2025.0.0`, which breaks
   `libtorch_cpu.so` (`undefined symbol: iJIT_NotifyEvent`) → pinned
   `mkl<2025`.
2. `cuda-toolkit=12.1` under-pinned `cuda-nvcc`/`cuda-cccl`, which resolved to
   13.3 and mismatched the installed PyTorch's CUDA build → pinned
   `cuda-nvcc=12.1`, `cuda-nvcc_linux-64=12.1`, `cuda-cccl=12.1` explicitly.
3. The system's default gcc was too new for nvcc 12.1 (nvcc rejects gcc > 12)
   → installed `gcc_linux-64=12 gxx_linux-64=12` from conda-forge and pointed
   `CC`/`CXX`/`NVCC_PREPEND_FLAGS` at it for the build steps.

Result: `demo/run.py` (+ `--grasp-steps`) ran clean over all 24 bundled
samples. The repo's own warning (vendored `third_party/mask2former/` may need
API patches against detectron2 `main`) did not materialize — detectron2 0.6
built from `main` at build time worked with the vendored code as-is.

**Why:** Matching the actual GPU's compute capability (sm_89) rather than the
script's sm_120 default is required for the CUDA kernels to run at all (or run
efficiently) on this hardware; conda over uv was a direct user preference and
also let the env pull a matching gcc/cuda-nvcc from conda-forge/nvidia
channels in one place rather than requiring a system-level toolchain install.

**Alternatives considered:** Installing a system-wide CUDA toolkit (apt/runfile)
matching a specific torch cu-index build — rejected as more invasive (system
package changes vs. an isolated conda env) and slower to get right than
pinning versions inside conda.

**Scope note:** This env (`s2g_seg`) only covers segmentation + grasping. The
classification (Qwen) half was explicitly out of scope for this pass, and the
README's default 35B model would not fit this GPU's 16GB VRAM anyway (the
smaller FP8 alternative is flagged in the README as broken/slow on non-Blackwell
GPUs) — full three-module classification runs are not currently feasible on
this machine without a smaller/quantized model choice, which hasn't been
decided yet.

---

## 2026-09-27 — Comparison target: Seg2Grasp analytic planner vs SuctionNet `normal_std`, scored by official suctionnetAPI

**Context:** User wants to benchmark Seg2Grasp's analytic suction planner against
SuctionNet on the GraspNet/SuctionNet-1Billion dataset (zips at
`~/Desktop/Codes/graspnet_dataset/`, SuctionNet baseline code at
`~/Desktop/Codes/suctionnet-baseline/`).

**Decision:** Compare against SuctionNet's **`normal_std`** baseline (the
training-free analytic method) first — chosen by the user — and evaluate BOTH
methods with the **official `suctionnetAPI`** metric, not a custom proxy. The
learned `neural_network` baseline (DeepLabV3+/ConvNet, needs Google-Drive
weights + old torch) is deferred.

**Why:** `normal_std` is training-free like Seg2Grasp's planner, so it is the
most apples-to-apples opponent and needs no checkpoint download. The official
API removes any doubt about metric fairness.

---

## 2026-09-27 — suctionnetAPI eval contract (hard-won; reference for both methods' output)

Reverse-engineered from the cloned `suctionnetAPI` source. Both methods MUST
produce predictions in exactly this shape or the eval is meaningless.

**Prediction dump layout:** `dump_folder/<split>/scene_%04d/<camera>/suction/%04d.npz`
where `<split>` = `test_seen` for scenes 100–129, `<camera>` = `realsense` or
`kinect`. Loaded via `SuctionGroup().from_npy()` which reads key `arr_0`.

**Prediction array:** float32 `[N, ≥7]`, columns
`[score, dir_x, dir_y, dir_z, t_x, t_y, t_z(, object_id)]`.
- `object_id` (col 7) is NOT used by eval — objects are assigned geometrically
  by nearest GT model point — so a 7-column array is sufficient (the API's
  `SUCTION_ARRAY_LEN=8` matters only for `.object_ids()`, never called in eval).
- **Translation is in METERS, camera frame** (normal_std builds the cloud from
  `depth_png / 1000`). Direction = unit surface normal in camera frame, oriented
  toward the camera (normal_std uses `orient_normals_to_align_with_direction([0,0,-1])`).
  ⚠️ Seg2Grasp works in **mm** internally, so its adapter MUST divide
  translations by 1000 before dumping.

**Eval root layout** (`SuctionNetEval(root, camera)`):
`root/{scenes/scene_%04d/<camera>/{annotations/*.xml,camera_poses.npy,cam0_wrt_table.npy},
models/%03d/nontextured.ply, dense_point_clouds/%03d.npz(key 'points')}`.
Our extracted `scene_0100..0129/` must be reachable under `root/scenes/`
(symlink), plus `models/` and `dense_point_clouds/` from their zips, plus the
`seal_label/`, `suction_collision_label/`, `wrench_label/` already extracted.

**Metric:** per frame — nms(2cm) → assign each suction to nearest GT object →
top-10 per object → collision check (gripper cylinder vs scene+table) →
`smoothness(seal) × wrench` score → precision@k over the scene-sorted top-50,
at thresholds [0.2,0.4,0.6,0.8]. Reports **AP_top50** and **AP_top1**.

**Key consequence for the comparison:** the metric is precision-over-top-k, i.e.
it rewards emitting MANY well-ranked candidates. Seg2Grasp natively outputs ONE
suction (one chosen target), so:
- **AP_top1 is the fair, apples-to-apples number** for Seg2Grasp (its single
  best grasp vs the opponent's single best).
- For a meaningful **AP_top50**, the Seg2Grasp adapter must emit a scene-wide
  ranked candidate set — feasible by running its `suction_candidates()` +
  `score_candidates()` (suction_planner.py) over ALL segmented objects, not just
  the selected target, then formatting to the [N,7] meters array. Report AP_top50
  with this caveat noted.

**Known blockers:** `suctionnetAPI` install pulls `suction_nms` (a compiled
extension) and `point_cloud_utils`; `transforms3d==0.3.1`. `suction_nms` does
not `pip download` cleanly and may need building from source. `normal_std`
inference.py uses `np.bool` (removed in numpy≥1.24; s2g_seg has numpy 1.26) so
it needs a one-line patch, and expects data under `dataset_root/scenes/...`.

---

## 2026-09-27 — Pilot results + root cause of Seg2Grasp's low AP_top1 (NOT a bug)

**Pilot** (scenes 100–101, realsense, 16 frames, official metric):

| Method | AP_top1 | AP_top50 |
|---|---|---|
| SuctionNet normal_std | 0.102 | 0.074 |
| Seg2Grasp | 0.039 | 0.082 |

Seg2Grasp wins AP_top50, loses AP_top1. `suction_nms` was genuinely
unavailable (broken PyPI sdist, no source in graspnet repos) so the eval used an
**authorized pure-numpy translation-only NMS fallback** — faithful because the
API calls `nms(0.02, 181°)` and 181°>180° disables the rotation gate, leaving a
2cm translation dedup. Flag if publishing numbers.

**Root cause of Seg2Grasp AP_top1=0 on scene 100** (diagnosed by scoring the
committed grasp per-frame with the eval's own sub-scores — see
`s2g_suctionnet_bench/diag_scene100.py`): the committed point is ON-surface
(1.5–6.5 mm), not colliding, and has good wrench (0.57–0.90) — but the
**smoothness/seal score is 0 every frame**. The API's seal test requires a 1 cm
suction-cup ring to seal against a continuous, sufficiently large flat patch.
Seg2Grasp's `select_target` picks the MOST-ELEVATED object (nearest camera),
which in scene 100 is a small/curved object (obj 3/4) that a 1 cm cup cannot
seal; it then commits even though no truly sealable patch exists. normal_std
instead picks the globally smoothest point, landing on the big flat object
(obj 8) and passing the seal test.

**Conclusion (a real finding, not an adapter bug — confirmed because both methods
pass/fail on the same eval and Seg2Grasp DOES pass on other frames):**
Seg2Grasp's weakness under this benchmark is **target selection** (optimizes
elevation, orthogonal to the benchmark's sealability criterion); its strength is
**candidate ranking** (its per-object scored candidates fill the top-k well →
higher AP_top50). Possible follow-up ablation: a "seal-aware top-1" that commits
to Seg2Grasp's own highest-scored candidate scene-wide instead of the
most-elevated target, to isolate planner quality from target-selection policy.

---

## 2026-09-27 — FULL benchmark results (test_seen, realsense, all 30 scenes × 256 frames)

Ran the full official `suctionnetAPI` eval over 7,680 frames/method. Numbers
independently re-aggregated from the saved raw `res` arrays — match the run.

| Method | AP_top1 | AP_top50 | AP_top1 per-thr [0.2/0.4/0.6/0.8] |
|---|---|---|---|
| SuctionNet normal_std | 0.1540 | 0.1010 | 0.315 / 0.188 / 0.095 / 0.018 |
| **Seg2Grasp (committed)** | **0.2055** | **0.1204** | 0.342 / 0.269 / 0.169 / 0.042 |
| Seg2Grasp (bestscore) | 0.1503 | 0.1188 | 0.251 / 0.203 / 0.119 / 0.029 |

**Findings (supersede the pilot):**
- **Seg2Grasp (committed) beats normal_std on BOTH metrics** — AP_top1 +33%
  (0.206 vs 0.154), AP_top50 +19% (0.120 vs 0.101). The gap widens at stricter
  seal thresholds (0.4/0.6), i.e. its committed grasp yields higher-quality
  seals, not just barely-valid ones.
- **committed > bestscore on AP_top1** (0.206 vs 0.150): Seg2Grasp's
  most-elevated-object target selection is a genuine asset — it produces a better
  single grasp than blindly taking the planner's globally highest-scored
  candidate. The pilot's opposite impression (target selection looked like a
  liability) was an artifact of just 2 scenes; over all 30 it reverses. `bestscore`
  (raw planner quality, no target policy) lands ~level with normal_std's top-1.
- AP_top50 ~identical across the two Seg2Grasp variants (0.120 vs 0.119) — they
  share the candidate pool, differing only at rank-1, as designed.

**Caveats:** (1) `suction_nms` compiled op is genuinely unavailable, so the eval
used a pure-numpy 2cm translation-only NMS — exact for the API's `nms(0.02,181°)`
call (181°>180° disables the rotation gate), but note it if publishing. (2) Only
the **realsense** camera and **test_seen** split were run; kinect and
test_similar/test_novel (the latter is Seg2Grasp's claimed generalization
strength) are not yet measured. (3) Classification (Qwen) is not part of this —
grasping only. Raw per-scene results saved at
`~/Desktop/Codes/s2g_suctionnet_bench/preds_full/*_res.npy` (shape 30×256×50×4)
for re-aggregation without re-evaluating.

---

## 2026-09-28 — Added SuctionNet LEARNED model (DeepLabV3+ RGB-D); full 4-way table

Ran SuctionNet's supervised DeepLabV3+ RGB-D model (pretrained realsense weights,
Google-Drive id `18TbctdhpNXEKLYDWFzI9cT1Wnhe-tn9h`, `--model deeplabv3plus_resnet101`
with a 4-channel RGBD backbone; state_dict confirms conv1 = 64×4×7×7, epoch 71)
through `neural_network/inference.py` (compat fix: `torchvision.models.utils` →
`torch.hub` for `load_state_dict_from_url` across 6 backbone files; new
`suctionnet_nn_infer.py` adds `--scenes/--frames`). Same suctionnetAPI eval.
Numbers independently re-aggregated from the raw res arrays.

| Method | AP_top1 | AP_top50 | top1 per-thr [0.2/0.4/0.6/0.8] |
|---|---|---|---|
| SuctionNet normal_std (analytic) | 0.1540 | 0.1010 | 0.314/0.188/0.095/0.018 |
| **SuctionNet DeepLabV3+ (learned RGB-D)** | **0.4625** | **0.2826** | 0.707/0.629/0.422/0.092 |
| Seg2Grasp (committed) | 0.2055 | 0.1204 | 0.342/0.269/0.169/0.042 |
| Seg2Grasp (bestscore) | 0.1503 | 0.1188 | 0.251/0.203/0.119/0.029 |

**Verdict on the seen split:** ranking is **learned SuctionNet ≫ Seg2Grasp
(committed) > normal_std ≈ Seg2Grasp (bestscore)**. The learned model roughly
DOUBLES Seg2Grasp on both AP_top1 (2.2×) and AP_top50 (2.3×). This is expected and
in-distribution: it is a supervised model evaluated on its own dataset's *seen*
objects — the setting most favorable to it (AP_top50 0.283 matches the paper's
~0.28 for RGB-D/seen, confirming the pipeline). Seg2Grasp still clearly beats the
analytic SuctionNet baseline.

**The discriminating test is `test_novel`** (Seg2Grasp's claimed strength: analytic
grasping generalizes to unseen objects while a supervised model may drop). That
split's data is NOT downloaded yet — the seen result alone does not settle the
overall "which is better" question; it only shows the learned model dominates
in-distribution.

---

## 2026-09-29 — Consolidated the learned SuctionNet model INTO the repo

**Context:** the 2026-09-28 learned-model run above happened entirely in the
external scratch repo (`~/Desktop/Codes/suctionnet-baseline/neural_network/`),
unlike the training-free pair which was already consolidated into
`benchmarks/suctionnet_comparison/` (2026-09-28, commit `2034263`). User asked to
finish the same consolidation for this arm before committing.

**Decision:** vendored `DeepLabV3Plus/` + `ConvNet/` (the model code, 236 KB, 20
files) under `benchmarks/suctionnet_comparison/third_party/suctionnet_neural_network/`,
carrying over the existing local patch (6 backbone files:
`torchvision.models.utils.load_state_dict_from_url` → `torch.hub.load_state_dict_from_url`,
required because the former was removed from modern torchvision). Added a
portable adapter `suctionnet_nn_infer.py` at the benchmark root — a rewrite of
`neural_network/inference.py` using `config.py` for paths/checkpoint (new
`SUCTIONNET_NN_CHECKPOINT` env var, default `<bench>/checkpoints/realsense-deeplabplus-RGBD`),
same dump contract as `normal_std_infer.py`/`s2g_infer.py` so it drops into the
existing `full_eval.py` unchanged. Checkpoint itself (~700 MB, Google-Drive id
`18TbctdhpNXEKLYDWFzI9cT1Wnhe-tn9h`) is gitignored, NOT committed — same pattern
as the GraspNet dataset and prediction dumps. Smoke-tested: vendored
`DeepLabV3Plus.network`/`ConvNet` import and build models cleanly in the
`suctionnet_eval` conda env, and the new script's CLI runs correctly up to (and
fails only at) the missing-checkpoint step.

**Why:** keeps the whole 4-arm comparison (normal_std, Seg2Grasp×2, learned
DeepLabV3+) in one portable, path-independent place per the user's original goal
for this benchmark directory — otherwise the learned-model numbers in
`DECISIONS.md`/`README.md` would reference code that only exists on this machine.

**Not done:** no prediction/eval re-run through the new in-repo script (the
already-recorded 2026-09-28 numbers came from the external-repo run and are
reused as-is); `run_all_evals.sh` was left as its pre-existing, non-portable,
training-free-pair-only convenience script rather than extended, to avoid scope
creep — extend it if the 3-way + learned loop becomes routine.
