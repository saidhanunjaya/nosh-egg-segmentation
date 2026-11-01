"""
build_pdf.py
Generates report.pdf (Sections 1-3) directly from project outputs.
Run: python build_pdf.py
Requires: pip install xhtml2pdf
"""

import base64
import sys
from pathlib import Path

try:
    from xhtml2pdf import pisa
except ImportError:
    print("Missing dependency. Run: pip install xhtml2pdf")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PDF = PROJECT_ROOT / "report.pdf"

MISSING_IMAGES = []
FOUND_IMAGES = []


def _try_glob(rel_pattern: str):
    """Return first match for a glob pattern relative to project root, else None."""
    matches = sorted((PROJECT_ROOT).glob(rel_pattern))
    return matches[0] if matches else None


def img_tag(rel_path_or_pattern, alt="", width="90%"):
    """
    Embed an image as base64. Tries exact path first, then falls back
    to treating the string as a glob pattern (handles unknown hash suffixes).
    """
    full_path = PROJECT_ROOT / rel_path_or_pattern
    if not full_path.exists():
        full_path = _try_glob(rel_path_or_pattern)
        if full_path is None:
            MISSING_IMAGES.append(str(rel_path_or_pattern))
            return f'<p style="color:red; border:1px solid red; padding:6px;">[MISSING IMAGE: {rel_path_or_pattern}]</p>'

    FOUND_IMAGES.append(str(full_path.relative_to(PROJECT_ROOT)))
    suffix = full_path.suffix.lower()
    mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    data = base64.b64encode(full_path.read_bytes()).decode("ascii")
    return f'<img src="data:{mime};base64,{data}" style="width:{width}; display:block; margin:8px auto;" alt="{alt}"/>'


def figure(rel_path_or_pattern, caption, width="70%"):
    return f"""
    <div style="margin:14px 0 22px 0; text-align:center;">
        {img_tag(rel_path_or_pattern, alt=caption, width=width)}
        <p style="font-size:9.5pt; font-style:italic; color:#333; margin-top:4px;">{caption}</p>
    </div>
    """


CSS = """
<style>
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; color: #111; border-bottom: 2px solid #444; padding-bottom: 6px; margin-top: 0; page-break-before: always; }
h1.no-break { page-break-before: avoid; }
h2 { font-size: 15pt; color: #1a1a1a; margin-top: 22px; }
h3 { font-size: 12.5pt; color: #1a1a1a; margin-top: 16px; }
p { margin: 8px 0; text-align: justify; }
code { background: #f1f1f1; padding: 1px 4px; border-radius: 3px; font-family: Courier, monospace; font-size: 9.5pt; }
pre { background: #f1f1f1; padding: 8px; font-family: Courier, monospace; font-size: 9pt; white-space: pre-wrap; border: 1px solid #ddd; }
table { width: 100%; margin: 12px 0; font-size: 8.5pt; }
th, td { border: 1px solid #999; padding: 5px 6px; text-align: left; vertical-align: top; }
th { background: #e8e8e8; font-weight: bold; }
ul { margin: 6px 0; padding-left: 20px; }
li { margin: 4px 0; }
.subtitle { font-size: 13pt; color: #444; margin-bottom: 4px; }
.coverline { text-align:center; margin-top: 40px; }
hr { border: none; border-top: 1px solid #ccc; margin: 24px 0; }
</style>
"""

# -------------------- COVER --------------------
cover_html = """
<div class="coverline">
<h1 class="no-break" style="border:none; font-size:26pt;">CV Intern Task — Report</h1>
<p class="subtitle">Egg Segmentation and Tracking</p>
<p style="color:#666;">Tasks 1-4: Pretrained Segmentation, Classical Segmentation, Evaluation, and Video Tracking</p>
</div>
"""

# -------------------- SECTION 1 --------------------
section1_html = f"""
<h1>Section 1: Task 1 - Pretrained-Model Egg Segmentation</h1>

<h2>1.1 Objective</h2>
<p>The goal of Task 1 was to segment individual eggs in a dataset of real-world photos using pretrained, off-the-shelf models — no custom training or fine-tuning. The output of this task (instance masks) serves two purposes downstream: it is treated as <b>ground truth</b> for evaluating the classical algorithm built in Task 2 (see Task 3), and it establishes the dev/eval split reused throughout the project.</p>

<h2>1.2 Pipeline</h2>
<p>A two-stage, zero-shot detect-then-segment pipeline was used:</p>
<ol>
<li><b>Detection — OWL-ViT (open-vocabulary object detector).</b> Each image is passed through OWL-ViT with the text prompt <code>"egg"</code>. OWL-ViT is a text-prompted detector, able to localize objects matching a natural-language description without being trained specifically on eggs. This produces bounding boxes around candidate egg regions, with confidence scores.</li>
<li><b>Segmentation — SAM (Segment Anything Model), box-prompted.</b> Each OWL-ViT bounding box is passed as a prompt into SAM, which produces a precise pixel-level instance mask for the object inside that box.</li>
</ol>
<p>This combination allows fully automatic, label-free instance segmentation: OWL-ViT answers "where are the eggs," and SAM answers "what exactly are their pixel boundaries."</p>
<p>A lightweight <b>fallback mechanism</b> supplements primary OWL-ViT detections: when the primary detector's output is missing or low-confidence for an image, a fallback box-proposal step (<code>source: fallback-1</code> in <code>review_list.txt</code>) supplies a substitute box, and the image is automatically flagged for manual review so this substitution is never silent.</p>

<h2>1.3 Dataset and Train/Eval Split</h2>
<p>The dataset (Roboflow export, egg images across varied backgrounds — hands, trays, marble, cloth, keyboards, leaves, etc.) contains 110 total images after de-duplication of Roboflow's augmented copies.</p>
<p>To prevent data leakage (Roboflow's augmentation pipeline can produce multiple near-duplicate copies of the same original photograph), images were grouped by <b>original source photo</b> before splitting, and a 4:1 grouped split was applied:</p>
<ul>
<li>Total images: <b>110</b></li>
<li>Dev set: <b>88</b> images</li>
<li>Eval set: <b>22</b> images</li>
</ul>
<p>This exact split (recorded in <code>task1_outputs_final/split.csv</code>) is reused unchanged in Task 2, so the classical algorithm is developed and evaluated on the same images as the pretrained pipeline, making the Task 3 comparison fair.</p>

<h2>1.4 Outputs</h2>
<p>For each image, the pipeline produces:</p>
<ul>
<li><code>masks/</code> — binary instance mask(s), one per detected egg</li>
<li><code>labels/</code> — class/instance label files</li>
<li><code>instances/</code> — per-instance crops/metadata</li>
<li><code>overlays/</code> — the original image with mask(s) drawn on top, for visual QA</li>
<li><code>diagnostics/</code> + <code>review_list.txt</code> — images flagged for manual review (fallback detections or manually-corrected boxes, <code>source: fallback-1</code> / <code>source: manual</code>)</li>
<li><code>results.csv</code> — per-image summary</li>
</ul>

<h2>1.5 Example Results</h2>
<p><b>Figure 1 (clean result):</b></p>
{figure("task1_outputs_final/overlays/not_damaged_2_jpg.rf.*_overlay.png",
        "Figure 1: A clean, tight segmentation - OWL-ViT correctly localizes the egg held in a hand, and SAM produces an accurate mask over it.")}

<p><b>Figure 2 (occlusion artifact):</b></p>
{figure("task1_outputs_final/overlays/not_damaged_11_jpg.rf.*_overlay.png",
        "Figure 2: The egg is partially occluded by a crossing plant stem. SAM's mask boundary is disrupted at the occlusion point, illustrating a minor limitation under occlusion.")}

<p><b>Figure 3 (fallback detection, flagged):</b></p>
{figure("task1_outputs_final/overlays/not_damaged_18_jpg.rf.*_overlay.png",
        "Figure 3: Despite a visually clean result, this image's box came from the fallback detector rather than primary OWL-ViT output, and was automatically flagged in review_list.txt.")}

<p><b>Figure 4 (unflagged failure):</b></p>
{figure("task1_outputs_final/overlays/not_damaged_125_jpg.rf.*_overlay.png",
        "Figure 4: A genuine failure case - the segmented region sits on a fold of tissue paper, not an egg. Unlike Figure 3, this image was NOT flagged for review, a silent failure.")}

<h2>1.6 Notes and Limitations</h2>
<ul>
<li>As a zero-shot pipeline, quality depends on OWL-ViT's detection recall/precision for the word "egg." A fallback mechanism catches some low-confidence cases and flags them for review (Figure 3), but is not a complete safety net: Figure 4 shows a confidently-wrong mask that passed through unflagged, meaning the review process does not catch 100% of real failures. Any downstream use of Task 1 outputs as "ground truth" (as in Task 3) should account for this.</li>
<li>Multi-instance images (multiple eggs in frame) are handled naturally since OWL-ViT can return multiple boxes per image, each independently segmented by SAM — this differs materially from Task 2's classical pipeline, which has no native multi-instance handling (discussed in Task 3).</li>
<li>No console logs/timing/threshold values were retained from the original run beyond the split statistics above; this section is written from the saved outputs (split.csv, review_list.txt, overlays) rather than run logs.</li>
</ul>
"""

# -------------------- SECTION 2 --------------------
section2_html = """
<h1>Section 2: Task 2 - Classical Image-Processing Egg Segmentation</h1>

<h2>2.1 Objective</h2>
<p>The goal of Task 2 was to build a <b>classical (non-deep-learning) image-processing algorithm</b> that segments eggs as closely as possible to the quality of Task 1's pretrained OWL-ViT + SAM pipeline, using only traditional computer vision techniques (thresholding, edge detection, color-space operations, morphological operations, etc.) — no neural networks, no learned weights.</p>
<p>To keep the comparison in Task 3 fair, Task 2 reuses <b>the exact same dev/eval split</b> established in Task 1 (<code>task1_outputs_final/split.csv</code>: 110 total images, dev=88, eval=22). All development and trial-and-error below was performed on the <b>dev set only</b>; the eval set was not used to tune any parameter, and the final locked-in algorithm was run once over the full 110-image dataset for reporting.</p>
<p>It was expected going in — and is confirmed by Task 3's quantitative analysis — that a classical pipeline would <b>not</b> match SAM's pixel-level accuracy. The goal was to get as close as a classical approach reasonably can, build genuine intuition about why it falls short, and document that gap rigorously. That gap is the subject of Task 3.</p>

<h2>2.2 Experiment Log</h2>
<p>Nine trials were run in sequence on the dev set, each building on diagnosis of the previous trial's failures. The full log — including abandoned approaches — is reproduced below.</p>

<table>
<tr><th>Trial</th><th>Change</th><th>Why Tried</th><th>Dev-Set Result</th><th>Verdict</th></tr>
<tr><td>1</td><td>Global Otsu thresholding on grayscale image</td><td>Simplest possible baseline — test if a single global threshold separates egg from background</td><td>Worked only on a handful of high-contrast, plain-background images; failed when background brightness overlapped egg brightness</td><td>Discarded — too simplistic, no spatial reasoning</td></tr>
<tr><td>2</td><td>Otsu + morphological clean-up + "pick most egg-shaped blob" scorer</td><td>Add shape-based selection to reject non-egg-shaped blobs from noisy segmentation</td><td>Marginal improvement; threshold itself remained the bottleneck</td><td>Discarded — downstream filtering can't fix an unreliable upstream threshold</td></tr>
<tr><td>3</td><td>Canny edge detection to contour extraction to ellipse-fit scoring (RETR_EXTERNAL)</td><td>Eggs are elliptical; edge+ellipse fitting should be more lighting-robust than intensity thresholding</td><td>Found edges often, but shadows/texture created spurious contours; false "egg-shaped" picks on clutter</td><td>Discarded as-is, iterated further (3b-3e)</td></tr>
<tr><td>3b</td><td>Switched to RETR_CCOMP to also search holes/internal contours</td><td>Hypothesis: true egg boundary sometimes appears as an internal contour</td><td>Recovered a few missed cases, introduced new false positives from internal texture (cloth folds, marble veining)</td><td>Discarded — net neutral, instability increased</td></tr>
<tr><td>3c</td><td>Added relative-size filter on contours</td><td>Prune obviously wrong-scale false candidates from 3b</td><td>Removed some false positives, but filtered out legitimate eggs at unusual scales</td><td>Discarded — couldn't separate true-but-odd-scale from false-but-plausible-scale</td></tr>
<tr><td>3d</td><td>Added absolute HSV brightness filter</td><td>Exploit that eggs are often lighter than many backgrounds</td><td>Worked on tuning images only; broke on darker eggs / bright backgrounds (skin, pale tile, keyboard highlights)</td><td>Discarded — absolute thresholds don't generalize across lighting diversity</td></tr>
<tr><td>3e</td><td>Switched to relative brightness/uniformity filter</td><td>Make the brightness heuristic lighting-invariant</td><td>Better than 3d but still failed on ash/rock textures, hand creases, keyboard gaps, patterned cloth</td><td>Discarded — confirmed local shape+color+brightness heuristics cannot generalize across this dataset</td></tr>
<tr><td>4</td><td>Combined: accept if EITHER Otsu blob OR Canny/ellipse contour scored well</td><td>Hypothesis: the two methods fail on different images; combining covers more cases</td><td>Some images improved, but combining two brittle methods mostly combined their failure modes (more false positives)</td><td>Discarded</td></tr>
<tr><td>5</td><td>Lighter variant of Trial 4: tuned closing + combined scoring</td><td>Last attempt before abandoning the family — rule out that a lighter touch fixes brittleness</td><td>Continued failing on same categories as 3e; each fix for one image broke a previously-working one</td><td><b>Entire contour/shape-scoring family abandoned.</b> Root cause: local heuristics are fundamentally brittle and cannot generalize across this dataset's background diversity</td></tr>
<tr><td>6</td><td>GrabCut seeded with plain centered rectangle + YCrCb skin-removal post-step</td><td>GrabCut models global fg/bg color distributions — fundamentally different from local shape heuristics</td><td>1/8 clean, 1/8 near-success; 2/8 partial failures from skin-removal over-firing on pale egg highlights; 3-4/8 severe background bleed; 1/8 total fg/bg inversion on patterned cloth</td><td>Kept concept; diagnosed two bugs: (A) skin mask too broad, catches pale egg highlights; (B) plain rectangle seed gives no guaranteed-fg anchor, allows inversion</td></tr>
<tr><td>7</td><td>Trimap seeding with hard GC_FGD center core + saturation-gated skin removal</td><td>Directly address both Trial 6 bugs</td><td>Net regression — 0/8 improved, several much worse; one hand image produced a totally unrelated diagonal wedge mask</td><td>Discarded — hard FGD core poisons the color model when background pixels fall inside the box (nearly always); largest-component pick can select an unrelated object if skin removal fragments the true foreground</td></tr>
<tr><td>8</td><td>Soft trimap (hard GC_BGD border only, no hard foreground) + one-time retry if fg &gt;55% of frame + saturation-gated skin removal + pick component closest to image center (not largest)</td><td>Fix Trial 7's two bugs directly</td><td>Substantial improvement — most images now produce a plausible single egg-centered region; bleed reduced but not eliminated; residual thin "neck" connections remained</td><td><b>Kept</b> — best result so far, neck-shaped bleed artifacts remained</td></tr>
<tr><td>9</td><td>Added morphological opening (kernel approx 2% of image diagonal) to sever thin necks, then re-apply centroid-pick and hole-filling</td><td>Opening removes thin connecting structures while preserving the wider main blob</td><td>Fixed several neck-connected cases cleanly; confirmed opening did NOT sever wide/stubby necks on marble, ash/rock, steel-pan backgrounds; widening kernel further risked eroding smaller eggs</td><td><b>Locked in as final.</b> Accepted as a real technique ceiling: opening helps thin-neck bleed but not wide-neck bleed</td></tr>
</table>

<p><b>Trial budget note:</b> after Trial 7's regression, a hard budget of 2-3 further trials was agreed before locking in whatever variant performed best. Trial 9 was the final trial under that budget.</p>

<h2>2.3 Final Algorithm (Locked In)</h2>
<p>The final pipeline, implemented in <code>task2_segment_eggs.py</code>, is:</p>
<ol>
<li><b>Soft-trimap-seeded GrabCut</b> — only the border region outside a detection box is marked hard background (GC_BGD); everything inside is left as "probable" fg/bg for GrabCut to resolve iteratively. No hard foreground core is used.</li>
<li><b>One-time adaptive retry</b> — if the resulting foreground covers more than 55% of the frame, the box is tightened and GrabCut is re-run once.</li>
<li><b>Saturation-gated YCrCb skin removal</b> — skin-colored pixels are removed from the foreground only above a saturation threshold, protecting pale egg highlights.</li>
<li><b>Morphological opening</b> (kernel approx 2% of image diagonal) to break thin background-bleed "necks."</li>
<li><b>Centroid-based component selection</b> — the connected component whose centroid is closest to the image center is kept, not the largest by area.</li>
<li><b>Hole-filling</b> to produce a clean, solid final mask.</li>
</ol>
<p>Output structure mirrors Task 1 (<code>task2_outputs_final/masks/</code>, <code>labels/</code>, <code>instances/</code>, <code>overlays/</code>, <code>results.csv</code>, <code>review_list.txt</code>), with <code>results.csv</code> additionally logging <code>area</code>, <code>frame_frac</code>, <code>retried</code>, <code>flagged</code>, and <code>flag_reason</code> per image for transparency.</p>

<h2>2.4 Full-Dataset Results</h2>
<p>The final script was run once over the complete 110-image dataset (both dev and eval), with no crashes:</p>
<ul>
<li><b>110/110 images processed successfully</b></li>
<li><b>39/110 (35.5%) flagged for review</b>, predominantly due to suspected background bleed</li>
<li><b>2/110 complete failures</b> (area = 0): a three-eggs-on-marble scene, and a hand+egg scene on low-contrast tile floor</li>
<li>An additional confirmed qualitative failure: a foil snack packet fused with the egg through a thick color-similarity "neck," surviving the morphological opening step due to its width</li>
</ul>

<h2>2.5 Known Limitations</h2>
<ul>
<li><b>No native multi-instance handling.</b> The pipeline selects exactly one connected component per image by design. Scenes with multiple eggs will have at most one correctly captured — a structural limitation, and the dominant root cause behind several of Task 3's failure categories (see Section 3.3.1).</li>
<li><b>Global color-model bleed.</b> GrabCut's foreground/background color model is global per image; when the egg's color is statistically similar to a large part of the background, the foreground region can bleed into or fuse with it. The morphological-opening fix only resolves thin-neck cases; wide necks are a confirmed, accepted failure mode.</li>
<li><b>No semantic understanding of "egg."</b> Unlike Task 1's OWL-ViT detector, the classical pipeline has no concept of what an egg looks like — it can confidently segment a non-egg object if that object's color statistics happen to separate well from its local surroundings.</li>
<li>Parameter choices were tuned on the dev set by visual inspection, not formal grid search or cross-validation — consistent with the trial-and-error approach documented above.</li>
</ul>
"""

# -------------------- SECTION 3 --------------------
section3_html = f"""
<h1>Section 3: Task 3 - Quantitative and Qualitative Evaluation</h1>

<h2>3.1 Methodology</h2>
<p>Task 2's classical segmentation outputs were evaluated against Task 1's SAM-based outputs, treated as ground truth. For each of the 110 images, the corresponding Task 1 mask and Task 2 mask were paired by filename stem and compared pixel-by-pixel using standard segmentation metrics: Intersection-over-Union (IoU), Dice coefficient, precision, and recall.</p>
<p><b>Mask pairing bug, found and fixed.</b> The original evaluation script's image-lookup function assumed a flat folder structure, but the dataset's actual Roboflow "Folder" export nests images one level deeper. This caused the visualization step's image lookup to silently fail for every image, producing an empty worst_cases folder while the script still printed a success message. Critically, this bug was isolated to the visualization lookup only — the core metrics computation loaded masks directly by filename via dictionary lookups, reported zero missing files throughout, and was unaffected. The bug was fixed by switching to a recursive file search, and the resulting worst-case comparison images were manually inspected to confirm all reported failures were genuine.</p>
<p><b>3.1.1 Verifying failures were real, not a scripting artifact.</b> Before accepting the quantitative results, a specific concern was investigated: several of the worst-scoring images showed precision = 0 and recall = 0 simultaneously, which could indicate either (a) a genuine wrong-region segmentation, or (b) a mask-pairing/indexing bug. After fixing the visualization bug and inspecting the actual image pairs, this was conclusively resolved as <b>(a): real failures.</b> In every zero-IoU case inspected, the Task 1 ground-truth mask was a clean, correctly-placed mask on the actual egg, while the Task 2 prediction was a confidently-segmented but completely unrelated region (a printed packaging graphic, a tray rail, a fabric seam, a floor shadow, a cable, a printed label). This confirmed genuine wrong-object capture, not a script bug, and directly informed the disjoint_wrong_region failure category defined below.</p>

<h2>3.2 Quantitative Results</h2>
<p>Full-dataset results (110 images, 0 missing/unmatched):</p>
<pre>IoU       mean=0.3436  median=0.2685  min=0.0000  max=0.9943
Dice      mean=0.4422  median=0.4234
Precision mean=0.4312  median=0.2873
Recall    mean=0.7217  median=0.9003

IoU &gt;= 0.75 (good):      19 / 110  (17.3%)
IoU 0.5-0.75 (ok):        7 / 110  ( 6.4%)
IoU &lt; 0.5 (poor/failed): 84 / 110  (76.4%)</pre>

<p><b>Figure 5 — IoU distribution across the dataset:</b></p>
{figure("task3_outputs/plots/iou_histogram.png", "Figure 5: IoU distribution - heavily left-skewed, most images score well below 0.5 IoU with a smaller cluster above 0.75.")}

<p>The distribution is heavily left-skewed: the majority of images score well below 0.5 IoU, with a smaller cluster of genuinely strong results above 0.75. There is no meaningful middle ground — the algorithm tends to either work well or fail badly.</p>

<p><b>Figure 6 — Precision vs. recall per image:</b></p>
{figure("task3_outputs/plots/precision_recall_scatter.png", "Figure 6: Precision vs. recall scatter - recall is systematically higher than precision, indicating over-segmentation as the dominant error mode.")}

<p>The mean recall (0.72) is substantially higher than mean precision (0.43), and the median recall (0.90) is far higher than median precision (0.29). This asymmetry indicates the classical pipeline's dominant error mode is <b>over-segmentation</b> rather than under-segmentation, matching the GrabCut background-bleed behavior documented in Task 2.</p>

<h2>3.3 Failure-Mode Taxonomy</h2>
<p>Each image was classified into exactly one of seven categories using a fixed rule set applied in priority order (pass, total_miss, disjoint_wrong_region, mislocalized, bleed_overseg, partial_underseg, mixed_moderate):</p>
<pre>pass                    26 ( 23.6%)  mean_iou=0.833
total_miss               2 (  1.8%)  mean_iou=0.000
disjoint_wrong_region   13 ( 11.8%)  mean_iou=0.000
mislocalized             5 (  4.5%)  mean_iou=0.119
bleed_overseg           56 ( 50.9%)  mean_iou=0.228
partial_underseg         7 (  6.4%)  mean_iou=0.339
mixed_moderate           1 (  0.9%)  mean_iou=0.415</pre>

<p><b>Figure 7 — Category counts:</b></p>
{figure("task3_outputs/plots/category_counts.png", "Figure 7: Count of images in each failure-mode category. bleed_overseg dominates at 50.9% of the dataset.")}

<p><b>Figure 8 — Predicted/GT area ratio vs. IoU:</b></p>
{figure("task3_outputs/plots/area_ratio_vs_iou.png", "Figure 8: Predicted/ground-truth area ratio vs. IoU. bleed_overseg cases cluster at high area ratios, confirming over-segmentation.")}

<p><code>bleed_overseg</code> is the single dominant failure mode at 50.9% of the dataset, consistent with the precision/recall asymmetry in Section 3.2. Its median predicted-area/ground-truth-area ratio of 4.21 confirms these predictions are, on average, over four times larger than the true egg region — direct numerical evidence of background bleed.</p>

<h3>3.3.1 Root-cause separation via multi-instance analysis</h3>
<p>The seven raw categories above describe symptoms, not root causes. Each image's Task 1 ground-truth mask was checked for multi-instance content (2+ connected components, each 3%+ of mask area), since Task 2's single-component-picking design is known to be structurally blind to multi-egg scenes.</p>
<table>
<tr><th>Category</th><th>N</th><th>Multi-instance</th><th>Multi-instance rate</th></tr>
<tr><td>pass</td><td>26</td><td>3</td><td>11.5%</td></tr>
<tr><td>total_miss</td><td>2</td><td>1</td><td>50.0%</td></tr>
<tr><td>disjoint_wrong_region</td><td>13</td><td>1</td><td>7.7%</td></tr>
<tr><td>mislocalized</td><td>5</td><td>3</td><td>60.0%</td></tr>
<tr><td>bleed_overseg</td><td>56</td><td>3</td><td>5.4%</td></tr>
<tr><td>partial_underseg</td><td>7</td><td>7</td><td>100.0%</td></tr>
<tr><td>mixed_moderate</td><td>1</td><td>0</td><td>0.0%</td></tr>
</table>
<p>Dataset baseline multi-instance rate: 16.4% (18/110)</p>

<p>This resolves the seven categories into <b>two independent root causes</b>:</p>
<ul>
<li><b>Root Cause A — single-object color-model failure</b> (bleed_overseg + disjoint_wrong_region = 69 images, 62.7% of the dataset): GrabCut's global per-image color model either over-grows the foreground onto similarly-colored background (bleed) or latches onto an entirely different salient region (wrong-region capture). Both categories occur at or below the dataset's baseline multi-instance rate (5.4% and 7.7% vs. 16.4% baseline), confirming they are <b>not</b> explained by multi-egg scenes.</li>
<li><b>Root Cause B — no multi-instance handling</b> (partial_underseg fully, plus the majority of mislocalized/total_miss, approx 14 images, 12.7% of the dataset): partial_underseg is <b>100%</b> explained by multi-instance scenes — every image in this category contains multiple eggs, of which the algorithm can only capture one. mislocalized (60%) and total_miss (50%) are majority multi-instance-driven as well, with a minority of genuine single-egg exceptions.</li>
</ul>

<h2>3.4 Qualitative Analysis</h2>
<p>Nine representative examples were hand-selected across categories for close visual inspection, in addition to the 10 zero-IoU worst-case images already reviewed in Section 3.1.1.</p>

<h3>Root Cause A examples (bleed / wrong-region)</h3>
{figure("task3_outputs/qualitative/bleed_overseg__not_damaged_43*.png",
        "Figure 9: not_damaged_43 (IoU=0.269, P=0.291, R=0.782) - the egg is captured cleanly (yellow overlap), but a large red false-positive region extends across the laptop body via a shared dark-color neck, illustrating the wide-neck bleed ceiling from Trial 9.")}

{figure("task3_outputs/qualitative/bleed_overseg__not_damaged_135*.png",
        "Figure 10: not_damaged_135 (IoU=0.057, P=0.057, R=1.000) - recall of 1.0 confirms the true egg region is fully contained within the prediction, but precision of 0.057 shows it is swamped by an enormous false-positive region (a bead-filled bucket).")}

{figure("task3_outputs/worst_cases/not_damaged_123*compare*.png",
        "Figure 11: not_damaged_123 (IoU=0.000) - the prediction (red) sits entirely on an egg tray's plastic rail, while the true egg (green) is untouched - a complete wrong-object capture.")}

<h3>Root Cause B examples (multi-instance)</h3>
{figure("task3_outputs/qualitative/mislocalized__not_damaged_114*.png",
        "Figure 12: not_damaged_114 (IoU=0.170, P=0.228, R=0.400) - a multi-egg marble scene. One egg is partially captured (yellow), while the remaining eggs are entirely untouched (pure green).")}

{figure("task3_outputs/qualitative/partial_underseg__not_damaged_133*.png",
        "Figure 13: not_damaged_133 (IoU=0.322, P=0.995, R=0.322) - precision near-1.0 shows the captured region is a clean, accurate segmentation of the one egg it selected; low recall is entirely due to other eggs being missed.")}

<h3>Passing-case contrast examples</h3>
{figure("task3_outputs/qualitative/pass__not_damaged_34*.png",
        "Figure 14: not_damaged_34 (IoU=0.994) - near-pixel-perfect agreement with Task 1's SAM mask, confirming the classical pipeline can produce excellent results under favorable conditions.")}

{figure("task3_outputs/qualitative/pass__not_damaged_161*.png",
        "Figure 15: not_damaged_161 (IoU=0.515) - a boundary case where the overall region is correct, but the bottom rim of the egg is slightly missed, likely due to a shadow being excluded by GrabCut's color model.")}

<h2>3.5 Discussion and Suggested Improvements</h2>
<p>The quantitative and qualitative analysis converges on two independent, well-evidenced root causes behind Task 2's gap relative to Task 1:</p>
<p><b>For Root Cause A (single-object color-model failure, 62.7% of the dataset):</b> GrabCut's global color-distribution model has no notion of object boundaries beyond color/contrast statistics. A practical improvement would be an <b>edge-aware refinement pass</b> (using detected object boundaries or a learned saliency/objectness prior to constrain the foreground region, rather than relying purely on global color separation). This would target both bleed and wrong-region capture, without requiring full deep-learning detection.</p>
<p><b>For Root Cause B (no multi-instance handling, 12.7% of the dataset):</b> The current design deliberately picks exactly one connected component per image. A straightforward fix — <b>retaining all sufficiently large connected components instead of a single centroid-pick</b> — would allow the algorithm to output multiple egg instances per image where present, directly eliminating the partial_underseg category and likely improving a majority of mislocalized and total_miss cases as well.</p>
<p>Both improvements are classical (non-deep-learning) in nature and were considered out of scope for the trial budget agreed during Task 2's development, but are flagged here as the clear next steps if further iteration were pursued.</p>
"""

FULL_HTML = f"""
<html>
<head>
<meta charset="utf-8"/>
{CSS}
</head>
<body>
{cover_html}
{section1_html}
{section2_html}
{section3_html}
</body>
</html>
"""


def main():
    print("Building PDF...")
    with open(OUTPUT_PDF, "wb") as f:
        result = pisa.CreatePDF(src=FULL_HTML, dest=f)
    if result.err:
        print(f"Finished with {result.err} error(s). Check output above.")
    else:
        print(f"Success. Wrote: {OUTPUT_PDF}")

    if MISSING_IMAGES:
        print(f"\nWARNING: {len(MISSING_IMAGES)} image(s) not found and were skipped:")
        for m in MISSING_IMAGES:
            print(f"  - {m}")
    else:
        print(f"\nAll {len(FOUND_IMAGES)} images embedded successfully:")
        for f_ in FOUND_IMAGES:
            print(f"  - {f_}")


if __name__ == "__main__":
    main()