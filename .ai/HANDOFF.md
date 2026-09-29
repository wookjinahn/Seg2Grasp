# Handoff

Living state for agent-to-agent handoff on this repo. Update this file at every
work boundary (before ending a session, before a risky change, or when you pause
mid-task) so the next agent can resume without re-deriving context. Keep entries
terse; put rationale in `DECISIONS.md` instead of repeating it here.

- **Last agent:** claude (Sonnet 5)
- **Timestamp:** 2026-09-29T01:00Z
- **Branch:** main  ·  **HEAD:** `2034263` "add SuctionNet comparison benchmark
  (portable, in-repo)". Seg2Grasp source tree still unmodified vs the `7c09574`
  release. Uncommitted at end of session: `.ai/DECISIONS.md`/`.ai/HANDOFF.md`
  updates + the new learned-model consolidation under
  `benchmarks/suctionnet_comparison/` (see checkpoint log below) — about to be
  committed per user instruction ("학습 모델 벤치마크도 리포에 통합 후 커밋").

## Current state

Seg2Grasp (IROS 2024) public code release: a modular suction bin-picking
pipeline (segmentation → grasping → classification).

**Work done across sessions:** built a local segmentation+grasping env, verified
the bundled demo, then ran a full quantitative + qualitative comparison of
Seg2Grasp's analytic suction planner vs SuctionNet's `normal_std` baseline (both
training-free) on the GraspNet/SuctionNet-1Billion dataset, scored by the
official `suctionnetAPI`: **Seg2Grasp wins on both AP_top1 and AP_top50** (table
in `DECISIONS.md`). Then added SuctionNet's **learned** DeepLabV3+ RGB-D model
(supervised, pretrained realsense weights) as a 4th arm on the same eval: it
roughly **doubles** Seg2Grasp on test_seen (expected — in-distribution supervised
model). Full 4-way table + verdict in `DECISIONS.md` (2026-09-28 entries). Side-
by-side and GT-scored visualizations produced for the training-free pair.

**Repo state / what to commit:** the Seg2Grasp source tree was NOT modified at
any point (still matches the `7c09574` release). Commit `0f300cc` already added
the `.ai/` handoff files and the user's `AGENTS.md`/`CLAUDE.md`/`.gitignore`
scaffolding. Added this session but not yet committed:
- `environment.yml` (repo root) — conda spec of the `s2g_seg` env actually used
  (conda/sm_89; alternative to the uv/Blackwell `scripts/setup_seg_env.sh`).
- This HANDOFF edit.
All benchmark code + data live OUTSIDE the repo in
`~/Desktop/Codes/s2g_suctionnet_bench/` and are not part of this repo's commit.

**Local machine state (this workstation only, not portable/committed):**
GPU is an RTX 4060 Ti (16GB, Ada Lovelace / sm_89) — NOT the Blackwell (sm_120)
the repo's `scripts/setup_seg_env.sh` targets by default. Two conda envs exist
locally: `s2g_seg` (segmentation+grasping, detectron2 built from source) and
`suctionnet_eval` (SuctionNet normal_std + official suctionnetAPI eval). The Qwen
classification env/model has **not** been set up (35B default doesn't fit 16GB
VRAM; FP8 alt flagged broken/slow on non-Blackwell — classification is out of
scope on this machine). `libero` env untouched.

## Next steps

The training-free comparison (Seg2Grasp vs normal_std, test_seen, realsense) and
its visualizations are **complete**, and the learned-model 4-way comparison
(test_seen, realsense) is also **complete** — see checkpoint log + `DECISIONS.md`
(2026-09-27 "FULL benchmark results", 2026-09-28 "Added SuctionNet LEARNED
model"). Nothing is in-progress or blocked; the only loose end is that the
DECISIONS.md entry for the learned-model run was uncommitted at the start of
this session (now reconciled with this HANDOFF, still uncommitted — ask user).

Possible follow-ups the user may want (none started):
1. **test_similar / test_novel** splits — Seg2Grasp's *claimed* generalization
   strength (paper's Hard column), and now also **the discriminating test**
   against the learned model (does its edge survive on unseen objects, or is it
   an in-distribution artifact?). Data for these splits is NOT downloaded yet
   (only `test_seen.zip` was provided). This is the most interesting next step.
2. **kinect** camera (only realsense measured so far).
3. ~~Learned SuctionNet baseline~~ — **done** (2026-09-28, DeepLabV3+ RGB-D,
   test_seen only; see DECISIONS.md). Its code is now also consolidated in-repo
   (2026-09-29, see below) — the numbers themselves came from the external run
   and were not re-generated through the in-repo script.
4. Resolve `suction_nms` properly (currently a faithful numpy fallback) before
   publishing any numbers.
5. Aggregate visualizations (per-scene top-1 success-rate bar chart) if a summary
   figure is wanted; also none yet for the learned-model arm.

### Benchmark code — NOW IN-REPO at `benchmarks/suctionnet_comparison/`
Consolidated into the repo (2026-09-28) so it moves between PCs with the repo.
Portable via `config.py` (paths from `GRASPNET_ROOT` env var etc.; no hardcoded
`~/Desktop/...`). Vendored + numpy-patched `suctionnetAPI` under its `third_party/`.
Both conda env specs in `envs/`. Representative galleries in `results/`. Large
data/predictions are gitignored (regenerated per machine). See its `README.md`
for the new-PC setup + run commands. On THIS machine, set
`GRASPNET_ROOT=~/Desktop/Codes/graspnet_dataset` and
`S2G_BENCH_PREDS=~/Desktop/Codes/s2g_suctionnet_bench/preds_full` to reuse the
already-generated data. Also now includes the learned SuctionNet model
(DeepLabV3+/ConvNet, `third_party/suctionnet_neural_network/` +
`suctionnet_nn_infer.py`, consolidated 2026-09-29) — set
`SUCTIONNET_NN_CHECKPOINT` to reuse the checkpoint at
`~/Desktop/Codes/s2g_suctionnet_bench/checkpoints/realsense-deeplabplus-RGBD`.

### Original scratch working dir (this machine only, NOT committed)
- Bench working dir: `~/Desktop/Codes/s2g_suctionnet_bench/`
  - `graspnet_loader.py` — GraspNet depth → organized mm cloud + intrinsics.
  - `s2g_infer.py` — Seg2Grasp → suctionnetAPI dump adapter; one GPU pass writes
    both `<save-dir>_committed/` (real policy, rank-1 = elevated-target grasp) and
    `<save-dir>_bestscore/` (rank-1 = top planner candidate). Runs in `s2g_seg`.
  - `normal_std_infer.py` — SuctionNet baseline port (np.bool fix, pilot-param).
  - `full_eval.py`, `run_all_evals.sh` — official eval driver (applies the numpy
    NMS monkeypatch, calls `SuctionNetEval.eval_seen`). Runs in `suctionnet_eval`.
  - `diag_scene100.py` — per-frame GT sub-score diagnostic.
  - `viz_compare.py` (score-colored), `viz_gt.py` (GT-quality colored, 2-panel),
    `viz_gt3.py` (3-panel: normal_std | committed | bestscore).
  - `preds_full/` — prediction dumps for all 3 sets + raw `*_res.npy`
    (shape 30×256×50×4, re-aggregate without re-evaluating). ~566 MB.
  - `viz/`, `viz_gt/`, `viz_gt3/` — rendered comparison galleries.
- GraspNet data: `~/Desktop/Codes/graspnet_dataset/` (scene_0100..0129, models,
  dense_point_clouds, seal/collision/wrench labels; a `scenes/` symlink dir was
  created for the API. Depth png = mm, factor_depth 1000).
- SuctionNet baseline repo: `~/Desktop/Codes/suctionnet-baseline/` (`normal_std/`).
- Envs: `s2g_seg` (Seg2Grasp/detectron2, py3.10, torch 2.5.1+cu121, detectron2
  0.6), `suctionnet_eval` (normal_std + API, py3.9, torch 2.8 CPU, transforms3d
  0.3.1). Reproducible specs exported to `s2g_suctionnet_bench/envs/`:
  `{s2g_seg,suctionnet_eval}.yml` (full conda export) + `*.pip.txt` (pip freeze).
- suctionnetAPI clone (read-only ref): in this session's scratchpad dir.

## Notes for future agents

- The repo's own setup scripts (`scripts/setup_seg_env.sh`, `setup_qwen_env.sh`)
  use `uv` and target Blackwell (sm_120). On this machine we used **conda**
  instead (user preference) and retargeted `TORCH_CUDA_ARCH_LIST=8.9` for the
  actual GPU — see `DECISIONS.md` for the exact substitution and the 3
  toolchain issues that had to be fixed (mkl version, cuda-nvcc version drift,
  gcc/nvcc incompatibility). None of it required touching repo code.
- Two separate envs are required for the full pipeline — segmentation
  (detectron2) and classification (transformers 5) have conflicting deps. See
  `README.md` → Installation. Locally, `s2g_seg` (segmentation/grasping) and
  `suctionnet_eval` (benchmark) exist; the Qwen classification env does not.
- `third_party/mask2former/` is vendored upstream code — avoid modifying it
  unless the task specifically requires patching the vendored copy; prefer
  wrapping/extending from `seg2grasp/segmentation/` instead. In practice, on
  this machine's detectron2 version, no patch was needed at all.
- `seg2grasp/robot/` is an optional, hardware-specific stub (UR5e + Robotiq
  AirPick) — not exercised by the demo or CI-equivalent paths.
- Segmentation checkpoint (~2.5 GB) is not in the repo; it's pulled from
  Hugging Face via `scripts/download_weights.sh` — already downloaded locally
  to `data/checkpoints/segmentation/` (gitignored).

## Checkpoint log

Append one entry per work boundary, most recent last.

- **2026-09-26 — Claude (Sonnet 5):** Created `.ai/HANDOFF.md` and
  `.ai/DECISIONS.md` to establish the handoff convention. No code changes.
- **2026-09-26 — Claude (Sonnet 5):** Built a local conda env (`s2g_seg`) for
  the segmentation+grasping half of the pipeline (adapted from the repo's
  uv/Blackwell-targeted script to conda/sm_89 — see `DECISIONS.md`), downloaded
  the segmentation checkpoint, and ran `demo/run.py` (+ `--grasp-steps`) over
  all 24 bundled sample frames — 24/24 succeeded, no repo code changes needed.
  Verified `demo/outputs/gallery.png` visually. Classification (Qwen) env not
  set up — out of scope for this pass and likely infeasible on this GPU at the
  README's default model size.
- **2026-09-27 — claude (Opus 4.8):** Set up the SuctionNet comparison. User
  brought GraspNet/SuctionNet-1Billion data (`~/Desktop/Codes/graspnet_dataset/`)
  and the SuctionNet baseline repo (`~/Desktop/Codes/suctionnet-baseline/`).
  Extracted all zips, verified GraspNet frame format (depth png = mm,
  factor_depth 1000), reverse-engineered the full `suctionnetAPI` eval contract
  (see DECISIONS.md), wrote the Seg2Grasp→suctionnetAPI adapter + GraspNet loader
  in `~/Desktop/Codes/s2g_suctionnet_bench/`. Built the `suctionnet_eval` conda
  env (`suction_nms` compiled op unavailable → faithful numpy 2cm-translation NMS
  fallback). Ran a 2-scene/16-frame pilot end-to-end.
- **2026-09-27 — claude (Opus 4.8):** Diagnosed the pilot's Seg2Grasp AP_top1=0
  on scene 100 (per-frame GT sub-scores via `diag_scene100.py`): NOT a bug — the
  committed grasp lands on-surface with good wrench but the elevated object isn't
  sealable by a 1 cm cup, so seal=0. Added a "seal-aware top-1" (`bestscore`)
  variant to the adapter to separate target-selection policy from planner quality.
- **2026-09-27 — claude (Opus 4.8):** Ran the FULL benchmark — test_seen (scenes
  100–129, realsense, all 256 frames = 7,680 frames/method), 3 prediction sets
  (normal_std, seg2grasp_committed, seg2grasp_bestscore), official suctionnetAPI.
  0 errors, 0 empty frames. Independently re-verified AP from saved raw `res`
  arrays. **Seg2Grasp (committed) wins both metrics** (AP_top1 0.206 vs 0.154,
  AP_top50 0.120 vs 0.101); committed > bestscore on top-1 → target selection is
  an asset. Full table + interpretation in DECISIONS.md.
- **2026-09-28 — claude (Opus 4.8):** Built qualitative visualizations overlaying
  both methods' suctions on GraspNet RGB (`viz_compare.py` score-colored;
  `viz_gt.py`/`viz_gt3.py` GT-quality colored, per-point pass/fail from the
  official scorer). Confirmed the visuals match the quantitative story. Committed
  `environment.yml` (s2g_seg conda spec) + handoff updates as `6c92b59`, pushed.
  Noted `AGENTS.md`/`CLAUDE.md`/`.gitignore` are the user's own scaffolding
  (committed by the user as `0f300cc`); Seg2Grasp source tree still unmodified.
- **2026-09-28 — claude (Opus 4.8):** Consolidated the whole benchmark INTO the
  repo at `benchmarks/suctionnet_comparison/` (user goal: one repo, portable to
  another PC). Made all scripts path-portable via `config.py` (env-var dataset
  root, standard GraspNet `scenes/` layout, repo-relative Seg2Grasp import),
  removed the ephemeral scratchpad path, vendored the numpy-patched suctionnetAPI
  under `third_party/`, added both env specs, a README with new-PC setup, a
  `.gitignore` for large data, and 3 downscaled result galleries. Smoke-tested
  all imports + a live 1-frame render through the portable config. 35 files,
  ~2.9 MB. Committed + pushed as `2034263`.
- **2026-09-28 — claude (Opus 4.8):** Ran SuctionNet's learned DeepLabV3+ RGB-D
  model (external `suctionnet-baseline/neural_network/`, new
  `suctionnet_nn_infer.py`; compat fix `torchvision.models.utils`→`torch.hub`
  across 6 backbone files) through the same official suctionnetAPI eval,
  test_seen/realsense. Result: learned model roughly doubles Seg2Grasp on both
  AP_top1/AP_top50 (expected, in-distribution supervised advantage; AP_top50
  matches the SuctionNet paper's own reported number, confirming the pipeline).
  Recorded as a 4-way table in `DECISIONS.md`. **Not consolidated into the
  in-repo `benchmarks/suctionnet_comparison/`** (still lives only in the
  external scratch repo) and the DECISIONS.md entry was left uncommitted.
- **2026-09-29 — claude (Sonnet 5):** Startup handoff read. Found the
  2026-09-28 learned-model DECISIONS.md entry uncommitted and this HANDOFF not
  yet reflecting it (Next steps still listed the learned baseline as
  not-started). Reconciled HANDOFF (current state, Next steps, this log) with
  DECISIONS.md; no code or benchmark changes yet. Asked the user how to
  proceed; they chose "consolidate the learned-model benchmark into the repo,
  then commit."
- **2026-09-29 — claude (Sonnet 5):** Consolidated the learned SuctionNet model
  (DeepLabV3+/ConvNet RGB-D) into `benchmarks/suctionnet_comparison/`, matching
  the pattern already used for normal_std/suctionnetAPI: vendored
  `DeepLabV3Plus/` + `ConvNet/` (236 KB, carrying the existing 6-file
  `torch.hub` compat patch) under `third_party/suctionnet_neural_network/`;
  added a portable `suctionnet_nn_infer.py` (uses `config.py`, new
  `SUCTIONNET_NN_CHECKPOINT` env var, same dump contract as the other two
  infer scripts so `full_eval.py` needs no changes); gitignored `checkpoints/`
  (weights not committed, ~700 MB); updated `README.md` (4-way results table,
  layout, setup step 6, run step 2b, attribution) and added a DECISIONS.md
  entry. Smoke-tested: vendored `DeepLabV3Plus.network`/`ConvNet` import and
  build in `suctionnet_eval` conda env; new script's CLI runs correctly up to
  the (expected) missing-checkpoint error — no live GPU inference/eval re-run
  (the 2026-09-28 numbers stand as-is). Ready to commit.
