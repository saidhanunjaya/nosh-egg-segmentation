# Egg Segmentation and Tracking

CV intern task: segmenting eggs in photos with a pretrained zero-shot pipeline and a classical (non-deep-learning) pipeline, evaluating the classical pipeline against the pretrained one, and tracking eggs in video.

- **Author:** Sai Dhananjaya
- **Full report:** see `report.pdf` in this repo, or view it on [Google Drive](https://drive.google.com/file/d/1w9duSYIKSJmNavPIwfnuVqRcjzlIbJuB/view?usp=sharing), for complete methodology, experiment logs, and figures.
- **Task 4 output video:** [Google Drive link](https://drive.google.com/file/d/15EBvtN9CcjhM4N869Stfo_eskTJgfvig/view?usp=sharing)

## Contents

- [Overview](#overview)
- [Task 1: Pretrained segmentation (OWL-ViT + SAM)](#task-1-pretrained-segmentation-owl-vit--sam)
- [Task 2: Classical segmentation](#task-2-classical-segmentation)
- [Task 3: Evaluation](#task-3-evaluation)
- [Task 4: Video tracking](#task-4-video-tracking)
- [Setup](#setup)
- [Reproducing results](#reproducing-results)
- [Known limitations](#known-limitations)
- [AI tools disclosure](#ai-tools-disclosure)

## Overview

Four tasks, built in sequence:

1. **Pretrained segmentation** — OWL-ViT (detection) + SAM (box-prompted segmentation), zero-shot, no training.
2. **Classical segmentation** — a from-scratch GrabCut-based pipeline using only traditional CV techniques, tuned through 9 logged trials.
3. **Evaluation** — Task 2's masks scored against Task 1's masks as the reference, with a failure-mode taxonomy.
4. **Video tracking** — a custom detector + SORT tracker + CSRT short-term tracker, keeping a stable identity per egg across a video clip.

## Task 1: Pretrained segmentation (OWL-ViT + SAM)

A two-stage, zero-shot detect-then-segment pipeline:

1. **OWL-ViT** detects egg regions from the text prompt `"egg"`, producing bounding boxes with confidence scores.
2. **SAM**, prompted with each box, produces a pixel-level instance mask.
   A fallback box-proposal step (`source: fallback-1`) supplies a substitute box when the primary detector's output is missing or low-confidence; any image using the fallback, or manually corrected, is automatically flagged in `review_list.txt` for manual review.

**Dataset:** 110 images (Roboflow export, de-duplicated), grouped by original source photo and split 4:1 to prevent leakage:

| Set  | Images |
| ---- | ------ |
| Dev  | 88     |
| Eval | 22     |

The split (`task1_outputs_final/split.csv`) is reused unchanged in Task 2.

**Outputs per image:** `masks/`, `labels/`, `instances/`, `overlays/`, `diagnostics/` + `review_list.txt`, `results.csv`.

**Known limitation:** the review flagging is not a complete safety net — at least one confidently-wrong mask passed through unflagged (see report Fig. 4). Task 1 outputs are therefore an _operational reference_, not verified ground truth.

## Task 2: Classical segmentation

A non-deep-learning pipeline built through 9 logged trials (full log in the report, §2.2), moving from global Otsu thresholding, through Canny/ellipse-fit contour scoring, to the final GrabCut-based approach.

**Final algorithm** (`task2_segment_eggs.py`):

1. Soft-trimap-seeded GrabCut (hard background border only, no hard foreground core).
2. One-time adaptive retry if foreground covers >55% of the frame.
3. Saturation-gated YCrCb skin removal (protects pale egg highlights).
4. Morphological opening (~2% of image diagonal) to sever thin background-bleed "necks."
5. Centroid-based connected-component selection (closest to image center, not largest).
6. Hole-filling for a clean final mask.
   **Full-dataset run (110 images):**

- 110/110 processed, no crashes
- 39/110 (35.5%) flagged for review
- 2/110 complete failures (area = 0)
  **Known limitations:**
- No native multi-instance handling — exactly one connected component per image.
- Global per-image color model can bleed into similarly-colored background.
- No semantic concept of "egg"; can confidently segment a non-egg object.
- Parameters tuned by visual inspection on the dev set, not formal grid search / cross-validation.

## Task 3: Evaluation

Task 2 masks were compared pixel-by-pixel against Task 1 masks (treated as reference) using IoU, Dice, precision, and recall, across all 110 images.

| Metric    | Mean   | Median |
| --------- | ------ | ------ |
| IoU       | 0.3436 | 0.2685 |
| Dice      | 0.4422 | 0.4234 |
| Precision | 0.4312 | 0.2873 |
| Recall    | 0.7217 | 0.9003 |

| IoU range            | Images | %     |
| -------------------- | ------ | ----- |
| ≥ 0.75 (good)        | 19/110 | 17.3% |
| 0.50–0.75 (moderate) | 7/110  | 6.4%  |
| < 0.50 (poor/failed) | 84/110 | 76.4% |

Recall >> precision indicates the classical pipeline tends to capture the egg but over-segments into background (consistent with the bleed behavior observed in Task 2).

**Failure-mode taxonomy** (7 mutually exclusive categories, applied in priority order):

| Category              | Images | %     | Mean IoU |
| --------------------- | ------ | ----- | -------- |
| pass                  | 26     | 23.6% | 0.833    |
| total_miss            | 2      | 1.8%  | 0.000    |
| disjoint_wrong_region | 13     | 11.8% | 0.000    |
| mislocalized          | 5      | 4.5%  | 0.119    |
| bleed_overseg         | 56     | 50.9% | 0.228    |
| partial_underseg      | 7      | 6.4%  | 0.339    |
| mixed_moderate        | 1      | 0.9%  | 0.415    |

Two main contributing factors were separated (report §3.3.1):

- **Factor A — single-object color-model failure:** `bleed_overseg` + `disjoint_wrong_region` (62.7% of images) have below-baseline multi-instance rates, pointing to GrabCut's color model spilling into background rather than a multi-egg issue.
- **Factor B — no multi-instance handling:** all 7 `partial_underseg` cases, plus several `mislocalized`/`total_miss` cases, are multi-instance scenes that Task 2's single-component design cannot represent.
  **Suggested improvements** (not implemented, out of trial budget): an edge-aware/boundary-constrained refinement step to reduce bleed, and retaining multiple candidate components to support multi-egg scenes.

## Task 4: Video tracking

Applied to a 31 s clip (0:09–0:40, 1280×720, 24 fps, 744 frames) containing two eggs — one on the table, one held in a hand.

The Task 2 classical pipeline was **not** reused for video (it over-segments 50.9% of the time). Instead, a simpler box-only detector was built and tuned for this clip.

**Detector** (`task4_step2_detector.py`): grayscale threshold (150) below the table edge, morphological open/close, contour filters on area/shape/saturation. Reliable on the table egg; rarely finds the held egg (finger occlusion, motion blur, bright background).

**Tracker** (`sort_tracker.py`): custom SORT — Kalman filter per track, Hungarian/IoU matching. Modified to zero velocity on missed updates, suppress ghost boxes, and revive dead tracks under the same ID (`lost_buffer_age=120`).

**Short-term tracking** (`task4_step4_csrt.py`, needs `opencv-contrib-python`): CSRT fills in when the detector loses an egg with an existing ID, drawn as a thin `"(est.)"` box, guarded against drift (max coast, area-ratio bounds, overlap check, frame-boundary check).

**Results** (no ground-truth annotations; box coverage + visual check over 1488 max egg-frames):

| Quantity                      | Egg-frames | Share |
| ----------------------------- | ---------- | ----- |
| Boxes from detector + tracker | 1231       | 82.7% |
| Boxes filled by CSRT          | 223        | 15.0% |
| Total with a box              | 1454       | 97.7% |
| No box                        | 34         | 2.3%  |

No ID swaps observed between the two eggs across the full video. CSRT boxes can be offset during holding (see report Fig. 4.2).

## Setup

Tested on Windows with Python and a virtual environment.

```bash
git clone https://github.com/saidhanunjaya/nosh-egg-segmentation.git
cd nosh-egg-segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Notes:

- Task 4 needs `opencv-contrib-python` for CSRT.
- Videos (`*.mp4`), dataset folders, and model weights (`*.pth`, `*.pt`) are **not** stored in this repo — download the dataset and the SAM checkpoint separately.
- The Task 4 clip is the 0:09–0:40 segment, saved as `task4_trimmed.mp4`.

## Reproducing results

| Task                                | Command                                              | Output                                 |
| ----------------------------------- | ---------------------------------------------------- | -------------------------------------- |
| Task 1 segmentation (OWL-ViT + SAM) | `python task1_segment_eggs.py`                       | `task1_outputs_final/`                 |
| Task 1 report figures               | `python task1_make_figures.py`                       | `task1_figures/`                       |
| Task 2 dev/eval split               | `python task2_step0_split.py`                        | split of 110 images (dev 88 / eval 22) |
| Task 2 trials 1–9                   | `task2_step1_otsu.py` ... `task2_step9_openbreak.py` | dev-set trial outputs                  |
| Task 2 final algorithm              | `python task2_segment_eggs.py`                       | `task2_outputs_final/`                 |
| Task 3 metrics                      | `python task3_evaluate.py`                           | `task3_outputs/`                       |
| Task 3 failure categories           | `python task3_failure_modes.py`                      | category table                         |
| Task 3 multi-instance check         | `python task3_multiinstance_check.py`                | §3.3.1 table                           |
| Task 3 plots                        | `python task3_plots.py`                              | `task3_outputs/plots/`                 |
| Task 3 qualitative examples         | `python task3_qualitative.py`                        | `task3_outputs/qualitative/`           |
| Task 4 detection + tracking         | `python task4_step3_track.py`                        | `task4_output.mp4`                     |
| Task 4 with CSRT                    | `python task4_step4_csrt.py`                         | `task4_output_csrt.mp4`                |
| Task 4 comparison figures           | `python task4_make_figures.py`                       | `task4_figures/`                       |
| Build the PDF report                | `python build_pdf.py`                                | `report.pdf`                           |

## Known limitations

- Task 1's review flagging does not catch all failures (silent failure case documented).
- Task 2 has no multi-instance handling and is prone to global-color-model bleed.
- Task 3's metrics measure agreement with Task 1's masks, not verified human-annotated ground truth.
- Task 4's detector/tracker thresholds are tuned to one clip and may not transfer; no quantitative tracking metric (e.g. MOTA) was computed, as there are no tracking annotations.

## AI tools disclosure

Claude (Anthropic) was used as an assistant for writing and debugging code, planning experiments, and drafting/formatting the report. All code was run and all outputs/videos checked by the author, who made the final decisions on methods.
