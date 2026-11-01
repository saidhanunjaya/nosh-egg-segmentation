"""Builds report.pdf (cover + Sections 1-6) from inline HTML.
Run:  python build_pdf.py
"""
import base64
from pathlib import Path

from xhtml2pdf import pisa

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PDF = PROJECT_ROOT / "report.pdf"
MISSING_IMAGES = []
FOUND_IMAGES = []

# Task 1 figures (relative to the project folder)
FIG1_PATH = "task1_figures/not_damaged_2_original_vs_sam.png"
FIG2_PATH = "task1_figures/not_damaged_11_original_vs_sam.png"
FIG3_PATH = "task1_figures/not_damaged_18_original_vs_sam.png"
FIG4_PATH = "task1_figures/not_damaged_125_original_vs_sam.png"

GITHUB_URL = "https://github.com/saidhanunjaya/nosh-egg-segmentation"
DRIVE_URL = "https://drive.google.com/file/d/15EBvtN9CcjhM4N869Stfo_eskTJgfvig/view?usp=sharing"

NAVY = "#1f3a5f"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _try_glob(rel_pattern):
    matches = sorted(PROJECT_ROOT.glob(rel_pattern))
    return matches[0] if matches else None


def img_tag(rel_path_or_pattern, alt="", width="90%"):
    p = PROJECT_ROOT / rel_path_or_pattern
    if not p.is_file():
        p = _try_glob(rel_path_or_pattern)
    if p is None:
        MISSING_IMAGES.append(rel_path_or_pattern)
        return ('<div style="border:2px solid red; color:red; padding:6px;">'
                '[MISSING IMAGE: %s]</div>' % rel_path_or_pattern)
    FOUND_IMAGES.append(str(p.relative_to(PROJECT_ROOT)))
    ext = p.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return '<img src="data:image/%s;base64,%s" width="%s" alt="%s" />' % (mime, b64, width, alt)


def figure(rel_path_or_pattern, caption, width="70%", title=None):
    """Title + image + caption, kept together on one page."""
    h = ""
    if title:
        h += '<p class="figtitle" style="-pdf-keep-with-next: true;"><b>%s</b></p>' % title
    h += '<p style="text-align:center; -pdf-keep-with-next: true;">%s</p>' % img_tag(rel_path_or_pattern, width=width)
    h += '<p class="cap">%s</p>' % caption
    return h


def figure_grid(items, img_width="225"):
    """2-column grid of (path, caption) pairs."""
    h = '<table width="100%" cellpadding="3" cellspacing="0" class="grid">'
    for i in range(0, len(items), 2):
        h += "<tr>"
        for path, caption in items[i:i + 2]:
            h += ('<td valign="top" align="center" class="gridcell">%s<p class="gcap">%s</p></td>'
                  % (img_tag(path, width=img_width), caption))
        if len(items[i:i + 2]) == 1:
            h += '<td class="gridcell">&nbsp;</td>'
        h += "</tr>"
    return h + "</table>"


def table(headers, rows, width="100%", col_widths=None):
    """All tables are full width so left/right edges line up with the text."""
    h = '<table width="100%" cellpadding="4" cellspacing="0" class="data" repeat="1">'
    h += "<tr>"
    for i, c in enumerate(headers):
        w = ' width="%s"' % col_widths[i] if col_widths else ""
        h += "<th%s>%s</th>" % (w, c)
    h += "</tr>"
    for n, r in enumerate(rows):
        bg = ' style="background-color:#f3f5f9;"' if n % 2 == 1 else ""
        h += "<tr>" + "".join('<td valign="top"%s>%s</td>' % (bg, c) for c in r) + "</tr>"
    return h + "</table>"


CSS = """
@page {
  size: a4;
  margin: 2cm 2cm 2.3cm 2cm;
  @frame footer_frame {
    -pdf-frame-content: footer_content;
    left: 2cm; width: 17cm; top: 27.6cm; height: 1cm;
  }
}
body { font-family: Helvetica; font-size: 10pt; line-height: 1.4; color: #222222; }
p { margin-top: 3px; margin-bottom: 5px; text-align: justify; }
li { margin-bottom: 2px; text-align: justify; }

h1 { font-size: 18pt; color: #1f3a5f; margin-top: 20px; margin-bottom: 8px;
     border-bottom: 2px solid #1f3a5f; padding-bottom: 4px; -pdf-keep-with-next: true; }
h1.newpage { page-break-before: always; margin-top: 0px; }
h1.no-break { page-break-before: avoid; }
h2 { font-size: 12.5pt; color: #1f3a5f; margin-top: 12px; margin-bottom: 4px; -pdf-keep-with-next: true; }
h2.pb { page-break-before: always; margin-top: 0px; }
h3 { font-size: 11pt; color: #333333; margin-top: 8px; margin-bottom: 3px; -pdf-keep-with-next: true; }
h3.pb { page-break-before: always; margin-top: 0px; }

code { font-family: Courier; font-size: 8.5pt; background-color: #eef0f4; }
pre { font-family: Courier; font-size: 8.5pt; background-color: #eef0f4; padding: 6px; text-align: left; }

table.data { border: 0.6pt solid #8a94a6; margin-top: 4px; margin-bottom: 6px; }
table.data th { background-color: #1f3a5f; color: #ffffff; text-align: left; font-size: 8.5pt;
                border: 0.5pt solid #8a94a6; }
table.data td { font-size: 8.5pt; text-align: left; border: 0.5pt solid #b7bdc9; }
table.data code { font-size: 7.5pt; }

p.figtitle { font-size: 9.5pt; color: #1f3a5f; margin-top: 8px; margin-bottom: 2px; text-align: left; }
p.cap { font-size: 8.5pt; font-style: italic; text-align: center; color: #444444; margin-top: 2px; margin-bottom: 8px; }
table.grid { margin-top: 4px; }
td.gridcell { border: 0.5pt solid #d0d4dc; }
p.gcap { font-size: 7.5pt; font-style: italic; text-align: center; color: #444444; }
p.footer { font-size: 8pt; color: #666666; text-align: center; border-top: 0.5pt solid #b7bdc9; padding-top: 3px; }
"""


# ----------------------------------------------------------------------------
# cover
# ----------------------------------------------------------------------------
def cover():
    h = """
<div id="footer_content"><p class="footer">CV Intern Task - Egg Segmentation and Tracking &nbsp;|&nbsp; Sai Dhananjaya &nbsp;|&nbsp; Page <pdf:pagenumber></p></div>
<h1 class="no-break" style="text-align:center; margin-top:110px; font-size:26pt;">CV Intern Task - Report</h1>
<h2 style="text-align:center; font-size:15pt;">Egg Segmentation and Tracking</h2>
<p style="text-align:center; color:#555555;">Tasks 1-4: Pretrained Segmentation, Classical Segmentation, Evaluation, and Video Tracking</p>
<p>&nbsp;</p>
"""
    h += table(["Item", "Details"], [
        ["<b>Author</b>", "Sai Dhananjaya"],
        ["<b>Code repository</b>", '<a href="%s">%s</a>' % (GITHUB_URL, GITHUB_URL)],
        ["<b>Task 4 output video</b>", '<a href="%s">%s</a>' % (DRIVE_URL, DRIVE_URL)],
    ], col_widths=["24%", "76%"])
    h += """
<h2 style="margin-top:36px;">Contents</h2>
<ul>
<li><b>Section 1</b> - Task 1: Pretrained-model egg segmentation (OWL-ViT + SAM)</li>
<li><b>Section 2</b> - Task 2: Classical image-processing segmentation and experiment log</li>
<li><b>Section 3</b> - Task 3: Quantitative and qualitative evaluation</li>
<li><b>Section 4</b> - Task 4: Egg tracking in video</li>
<li><b>Section 5</b> - Setup and reproduction</li>
<li><b>Section 6</b> - AI tools disclosure</li>
</ul>
"""
    return h


# ----------------------------------------------------------------------------
# Section 1 - Task 1
# ----------------------------------------------------------------------------
def section1():
    s = """
<h1 class="newpage">Section 1: Task 1 - Pretrained-Model Egg Segmentation</h1>

<h2>1.1 Objective</h2>
<p>The goal of Task 1 was to segment individual eggs in a dataset of real-world photos using pretrained,
off-the-shelf models &mdash; no custom training or fine-tuning. The output of this task (instance masks)
serves two purposes downstream: it is treated as <b>ground truth</b> for evaluating the classical algorithm
built in Task 2 (see Task 3), and it establishes the dev/eval split reused throughout the project.</p>

<h2>1.2 Pipeline</h2>
<p>A two-stage, zero-shot detect-then-segment pipeline was used:</p>
<ol>
<li><b>Detection &mdash; OWL-ViT (open-vocabulary object detector).</b> Each image is passed through OWL-ViT
with the text prompt <code>"egg"</code>. OWL-ViT is a text-prompted detector, able to localize objects matching
a natural-language description without being trained specifically on eggs. This produces bounding boxes around
candidate egg regions, with confidence scores.</li>
<li><b>Segmentation &mdash; SAM (Segment Anything Model), box-prompted.</b> Each OWL-ViT bounding box is passed
as a prompt into SAM, which produces a precise pixel-level instance mask for the object inside that box.</li>
</ol>
<p>This combination allows fully automatic, label-free instance segmentation: OWL-ViT answers "where are the
eggs," and SAM answers "what exactly are their pixel boundaries."</p>
<p>A lightweight <b>fallback mechanism</b> supplements primary OWL-ViT detections: when the primary detector's
output is missing or low-confidence for an image, a fallback box-proposal step
(<code>source: fallback-1</code> in <code>review_list.txt</code>) supplies a substitute box, and the image is
automatically flagged for manual review so this substitution is never silent.</p>

<h2>1.3 Dataset and Train/Eval Split</h2>
<p>The dataset (Roboflow export, egg images across varied backgrounds &mdash; hands, trays, marble, cloth,
keyboards, leaves, etc.) contains 110 total images after de-duplication of Roboflow's augmented copies.</p>
<p>To prevent data leakage (Roboflow's augmentation pipeline can produce multiple near-duplicate copies of the
same original photograph), images were grouped by <b>original source photo</b> before splitting, and a 4:1
grouped split was applied:</p>
<ul>
<li>Total images: <b>110</b></li>
<li>Dev set: <b>88</b> images</li>
<li>Eval set: <b>22</b> images</li>
</ul>
<p>This exact split (recorded in <code>task1_outputs_final/split.csv</code>) is reused unchanged in Task 2, so
the classical algorithm is developed and evaluated on the same images as the pretrained pipeline, making the
Task 3 comparison fair.</p>

<h2>1.4 Outputs</h2>
<p>For each image, the pipeline produces:</p>
<ul>
<li><code>masks/</code> &mdash; binary instance mask(s), one per detected egg</li>
<li><code>labels/</code> &mdash; class/instance label files</li>
<li><code>instances/</code> &mdash; per-instance crops/metadata</li>
<li><code>overlays/</code> &mdash; the original image with mask(s) drawn on top, for visual QA</li>
<li><code>diagnostics/</code> + <code>review_list.txt</code> &mdash; images flagged for manual review (fallback
detections or manually-corrected boxes, <code>source: fallback-1</code> / <code>source: manual</code>)</li>
<li><code>results.csv</code> &mdash; per-image summary</li>
</ul>

<h2>1.5 Example Results</h2>
"""
    s += figure(FIG1_PATH, "Figure 1: A clean, tight segmentation - OWL-ViT correctly localizes the egg held in a hand, and SAM produces an accurate mask over it.",
                width="72%", title="Figure 1 (clean result)")
    s += figure(FIG2_PATH, "Figure 2: The egg is partially occluded by a crossing plant stem. SAM's mask boundary is disrupted at the occlusion point, illustrating a minor limitation under occlusion.",
                width="72%", title="Figure 2 (occlusion artifact)")
    s += figure(FIG3_PATH, "Figure 3: Despite a visually clean result, this image's box came from the fallback detector rather than primary OWL-ViT output, and was automatically flagged in review_list.txt.",
                width="72%", title="Figure 3 (fallback detection, flagged)")
    s += figure(FIG4_PATH, "Figure 4: A genuine failure case - the segmented region sits on a fold of tissue paper, not an egg. Unlike Figure 3, this image was NOT flagged for review, a silent failure.",
                width="72%", title="Figure 4 (unflagged failure)")
    s += """
<h2>1.6 Notes and Limitations</h2>
<ul>
<li>As a zero-shot pipeline, quality depends on OWL-ViT's detection recall/precision for the word "egg." A
fallback mechanism catches some low-confidence cases and flags them for review (Figure 3), but is not a
complete safety net: Figure 4 shows a confidently-wrong mask that passed through unflagged, meaning the review
process does not catch 100% of real failures. Any downstream use of Task 1 outputs as "ground truth" (as in
Task 3) should account for this.</li>
<li>Multi-instance images (multiple eggs in frame) are handled naturally since OWL-ViT can return multiple
boxes per image, each independently segmented by SAM &mdash; this differs materially from Task 2's classical
pipeline, which has no native multi-instance handling (discussed in Task 3).</li>
<li>No console logs/timing/threshold values were retained from the original run beyond the split statistics
above; this section is written from the saved outputs (split.csv, review_list.txt, overlays) rather than run
logs.</li>
</ul>
"""
    return s


# ----------------------------------------------------------------------------
# Section 2 - Task 2
# ----------------------------------------------------------------------------
def section2():
    s = """
<h1>Section 2: Task 2 - Classical Image-Processing Egg Segmentation</h1>

<h2>2.1 Objective</h2>
<p>The goal of Task 2 was to build a <b>classical (non-deep-learning) image-processing algorithm</b> that
segments eggs as closely as possible to the quality of Task 1's pretrained OWL-ViT + SAM pipeline, using only
traditional computer vision techniques (thresholding, edge detection, color-space operations, morphological
operations, etc.) &mdash; no neural networks, no learned weights.</p>
<p>To keep the comparison in Task 3 fair, Task 2 reuses <b>the exact same dev/eval split</b> established in
Task 1 (<code>task1_outputs_final/split.csv</code>: 110 total images, dev=88, eval=22). All development and
trial-and-error below was performed on the <b>dev set only</b>; the eval set was not used to tune any
parameter, and the final locked-in algorithm was run once over the full 110-image dataset for reporting.</p>
<p>It was expected going in &mdash; and is confirmed by Task 3's quantitative analysis &mdash; that a classical
pipeline would <b>not</b> match SAM's pixel-level accuracy. The goal was to get as close as a classical
approach reasonably can, build genuine intuition about why it falls short, and document that gap rigorously.
That gap is the subject of Task 3.</p>

<h2>2.2 Experiment Log</h2>
<p>Nine trials were run in sequence on the dev set, each building on diagnosis of the previous trial's
failures. The full log &mdash; including abandoned approaches &mdash; is reproduced below.</p>
"""
    rows = [
        ["1", "Global Otsu thresholding on grayscale image",
         "Simplest possible baseline &mdash; test if a single global threshold separates egg from background",
         "Worked only on a handful of high-contrast, plain-background images; failed when background brightness overlapped egg brightness",
         "Discarded &mdash; too simplistic, no spatial reasoning"],
        ["2", "Otsu + morphological clean-up + &quot;pick most egg-shaped blob&quot; scorer",
         "Add shape-based selection to reject non-egg-shaped blobs from noisy segmentation",
         "Marginal improvement; threshold itself remained the bottleneck",
         "Discarded &mdash; downstream filtering can't fix an unreliable upstream threshold"],
        ["3", "Canny edge detection to contour extraction to ellipse-fit scoring (RETR_EXTERNAL)",
         "Eggs are elliptical; edge+ellipse fitting should be more lighting-robust than intensity thresholding",
         "Found edges often, but shadows/texture created spurious contours; false &quot;egg-shaped&quot; picks on clutter",
         "Discarded as-is, iterated further (3b-3e)"],
        ["3b", "Switched to RETR_CCOMP to also search holes/internal contours",
         "Hypothesis: true egg boundary sometimes appears as an internal contour",
         "Recovered a few missed cases, introduced new false positives from internal texture (cloth folds, marble veining)",
         "Discarded &mdash; net neutral, instability increased"],
        ["3c", "Added relative-size filter on contours",
         "Prune obviously wrong-scale false candidates from 3b",
         "Removed some false positives, but filtered out legitimate eggs at unusual scales",
         "Discarded &mdash; couldn't separate true-but-odd-scale from false-but-plausible-scale"],
        ["3d", "Added absolute HSV brightness filter",
         "Exploit that eggs are often lighter than many backgrounds",
         "Worked on tuning images only; broke on darker eggs / bright backgrounds (skin, pale tile, keyboard highlights)",
         "Discarded &mdash; absolute thresholds don't generalize across lighting diversity"],
        ["3e", "Switched to relative brightness/uniformity filter",
         "Make the brightness heuristic lighting-invariant",
         "Better than 3d but still failed on ash/rock textures, hand creases, keyboard gaps, patterned cloth",
         "Discarded &mdash; confirmed local shape+color+brightness heuristics cannot generalize across this dataset"],
        ["4", "Combined: accept if EITHER Otsu blob OR Canny/ellipse contour scored well",
         "Hypothesis: the two methods fail on different images; combining covers more cases",
         "Some images improved, but combining two brittle methods mostly combined their failure modes (more false positives)",
         "Discarded"],
        ["5", "Lighter variant of Trial 4: tuned closing + combined scoring",
         "Last attempt before abandoning the family &mdash; rule out that a lighter touch fixes brittleness",
         "Continued failing on same categories as 3e; each fix for one image broke a previously-working one",
         "<b>Entire contour/shape-scoring family abandoned.</b> Root cause: local heuristics are fundamentally brittle and cannot generalize across this dataset's background diversity"],
        ["6", "GrabCut seeded with plain centered rectangle + YCrCb skin-removal post-step",
         "GrabCut models global fg/bg color distributions &mdash; fundamentally different from local shape heuristics",
         "1/8 clean, 1/8 near-success; 2/8 partial failures from skin-removal over-firing on pale egg highlights; 3-4/8 severe background bleed; 1/8 total fg/bg inversion on patterned cloth",
         "Kept concept; diagnosed two bugs: (A) skin mask too broad, catches pale egg highlights; (B) plain rectangle seed gives no guaranteed-fg anchor, allows inversion"],
        ["7", "Trimap seeding with hard GC_FGD center core + saturation-gated skin removal",
         "Directly address both Trial 6 bugs",
         "Net regression &mdash; 0/8 improved, several much worse; one hand image produced a totally unrelated diagonal wedge mask",
         "Discarded &mdash; hard FGD core poisons the color model when background pixels fall inside the box (nearly always); largest-component pick can select an unrelated object if skin removal fragments the true foreground"],
        ["8", "Soft trimap (hard GC_BGD border only, no hard foreground) + one-time retry if fg &gt;55% of frame + saturation-gated skin removal + pick component closest to image center (not largest)",
         "Fix Trial 7's two bugs directly",
         "Substantial improvement &mdash; most images now produce a plausible single egg-centered region; bleed reduced but not eliminated; residual thin &quot;neck&quot; connections remained",
         "<b>Kept</b> &mdash; best result so far, neck-shaped bleed artifacts remained"],
        ["9", "Added morphological opening (kernel approx 2% of image diagonal) to sever thin necks, then re-apply centroid-pick and hole-filling",
         "Opening removes thin connecting structures while preserving the wider main blob",
         "Fixed several neck-connected cases cleanly; confirmed opening did NOT sever wide/stubby necks on marble, ash/rock, steel-pan backgrounds; widening kernel further risked eroding smaller eggs",
         "<b>Locked in as final.</b> Accepted as a real technique ceiling: opening helps thin-neck bleed but not wide-neck bleed"],
    ]
    s += table(["Trial", "Change", "Why Tried", "Dev-Set Result", "Verdict"], rows,
               col_widths=["6%", "22%", "22%", "26%", "24%"])
    s += """
<p><b>Trial budget note:</b> after Trial 7's regression, a hard budget of 2-3 further trials was agreed before
locking in whatever variant performed best. Trial 9 was the final trial under that budget.</p>

<h2>2.3 Final Algorithm (Locked In)</h2>
<p>The final pipeline, implemented in <code>task2_segment_eggs.py</code>, is:</p>
<ol>
<li><b>Soft-trimap-seeded GrabCut</b> &mdash; only the border region outside a detection box is marked hard
background (GC_BGD); everything inside is left as "probable" fg/bg for GrabCut to resolve iteratively. No hard
foreground core is used.</li>
<li><b>One-time adaptive retry</b> &mdash; if the resulting foreground covers more than 55% of the frame, the
box is tightened and GrabCut is re-run once.</li>
<li><b>Saturation-gated YCrCb skin removal</b> &mdash; skin-colored pixels are removed from the foreground only
above a saturation threshold, protecting pale egg highlights.</li>
<li><b>Morphological opening</b> (kernel approx 2% of image diagonal) to break thin background-bleed
"necks."</li>
<li><b>Centroid-based component selection</b> &mdash; the connected component whose centroid is closest to the
image center is kept, not the largest by area.</li>
<li><b>Hole-filling</b> to produce a clean, solid final mask.</li>
</ol>
<p>Output structure mirrors Task 1 (<code>task2_outputs_final/masks/</code>, <code>labels/</code>,
<code>instances/</code>, <code>overlays/</code>, <code>results.csv</code>, <code>review_list.txt</code>), with
<code>results.csv</code> additionally logging <code>area</code>, <code>frame_frac</code>, <code>retried</code>,
<code>flagged</code>, and <code>flag_reason</code> per image for transparency.</p>

<h2>2.4 Full-Dataset Results</h2>
<p>The final script was run once over the complete 110-image dataset (both dev and eval), with no crashes:</p>
<ul>
<li><b>110/110 images processed successfully</b></li>
<li><b>39/110 (35.5%) flagged for review</b>, predominantly due to suspected background bleed</li>
<li><b>2/110 complete failures</b> (area = 0): a three-eggs-on-marble scene, and a hand+egg scene on
low-contrast tile floor</li>
<li>An additional confirmed qualitative failure: a foil snack packet fused with the egg through a thick
color-similarity "neck," surviving the morphological opening step due to its width</li>
</ul>

<h2>2.5 Known Limitations</h2>
<ul>
<li><b>No native multi-instance handling.</b> The pipeline selects exactly one connected component per image by
design. Scenes with multiple eggs will have at most one correctly captured &mdash; a structural limitation, and
a major root cause behind several of Task 3's failure categories (see Section 3.3.1).</li>
<li><b>Global color-model bleed.</b> GrabCut's foreground/background color model is global per image; when the
egg's color is statistically similar to a large part of the background, the foreground region can bleed into or
fuse with it. The morphological-opening fix only resolves thin-neck cases; wide necks are a confirmed, accepted
failure mode.</li>
<li><b>No semantic understanding of "egg."</b> Unlike Task 1's OWL-ViT detector, the classical pipeline has no
concept of what an egg looks like &mdash; it can confidently segment a non-egg object if that object's color
statistics happen to separate well from its local surroundings.</li>
<li>Parameter choices were tuned on the dev set by visual inspection, not formal grid search or cross-validation
&mdash; consistent with the trial-and-error approach documented above.</li>
</ul>
"""
    return s


# ----------------------------------------------------------------------------
# Section 3 - Task 3
# ----------------------------------------------------------------------------
def section3():
    QUAL = "task3_outputs/qualitative/"
    s = """
<h1>Section 3: Task 3 - Quantitative and Qualitative Evaluation</h1>

<h2>3.1 Methodology</h2>
<p>Task 2's classical segmentation masks were evaluated against Task 1's SAM-based masks, which were treated as
the reference masks. The two masks for each image were paired by filename stem and compared pixel by pixel
using Intersection-over-Union (IoU), Dice coefficient, precision, and recall.</p>
<p>For predicted mask P and reference mask G:</p>
<ul>
<li><b>IoU</b> = |P intersection G| / |P union G|</li>
<li><b>Dice</b> = 2 |P intersection G| / (|P| + |G|)</li>
<li><b>Precision</b> = TP / (TP + FP)</li>
<li><b>Recall</b> = TP / (TP + FN)</li>
</ul>
<p>The reported summary statistics are calculated across the 110 per-image results. Task 1 masks are treated
as operational reference masks, not manually verified ground truth. As noted in Section 1, Task 1 contains at
least one unflagged failure; therefore, these metrics measure agreement with Task 1 rather than absolute
accuracy against human-annotated masks.</p>
<p><b>Visualization lookup bug, found and fixed.</b> The original evaluation script's image-lookup function
assumed that source images were stored in a flat folder. The Roboflow export instead nests images one level
deeper. This caused the visualization step to fail to locate source images, leaving the
<code>worst_cases</code> folder empty even though the script reported success. The bug affected visualization
only: the metrics were computed from mask files paired by filename, and there were zero missing or unmatched
masks. The image lookup was fixed by using a recursive search, after which the generated worst-case comparisons
were manually inspected.</p>
<p><b>Verifying zero-IoU failures.</b> Some images had both precision and recall equal to zero. This can occur
when the prediction is a non-empty mask that does not overlap the reference mask at all. Manual inspection of
the zero-IoU examples confirmed that these were genuine wrong-region predictions, not a pairing error: the
Task 1 mask was on the egg, while the Task 2 prediction was on unrelated regions such as a printed graphic, tray
rail, fabric seam, floor shadow, cable, or label.</p>
<p>The results below summarize all 110 images. Task 2's parameters were tuned on the 88-image dev set, with the
22-image eval set held out during development. Because the table reports the pooled full-dataset results rather
than a separate eval-only score, these figures should be read as a full-dataset comparison, not as an
independent holdout estimate.</p>

<h2 class="pb">3.2 Quantitative Results</h2>
"""
    s += table(["Metric", "Mean", "Median"], [
        ["IoU", "0.3436", "0.2685"],
        ["Dice", "0.4422", "0.4234"],
        ["Precision", "0.4312", "0.2873"],
        ["Recall", "0.7217", "0.9003"],
    ], col_widths=["40%", "30%", "30%"])
    s += "<p>Per-image IoU ranged from 0.0000 (minimum) to 0.9943 (maximum).</p>"
    s += table(["IoU range", "Images", "Percentage"], [
        ["IoU &gt;= 0.75 - good", "19 / 110", "17.3%"],
        ["0.50 &lt;= IoU &lt; 0.75 - moderate", "7 / 110", "6.4%"],
        ["IoU &lt; 0.50 - poor or failed", "84 / 110", "76.4%"],
    ], col_widths=["50%", "25%", "25%"])
    s += figure_grid([
        ("task3_outputs/plots/iou_histogram.png",
         "<b>Figure 5 - IoU distribution.</b> A large concentration of low-IoU results and a smaller group of strong results: performance varies substantially between images."),
        ("task3_outputs/plots/precision_recall_scatter.png",
         "<b>Figure 6 - Precision and recall per image.</b> Recall is generally higher than precision, so over-segmentation is a more common error than missing egg pixels."),
    ], img_width="230")
    s += """
<p>Mean recall (0.7217) is substantially higher than mean precision (0.4312); the median values show the same
pattern. This suggests that the classical pipeline often captures some or most of the reference egg region, but
includes too much background in its prediction. This is consistent with the background-bleed behaviour observed
during Task 2 development.</p>

<h2 class="pb">3.3 Failure-Mode Taxonomy</h2>
<p>Each image was assigned to one of seven mutually exclusive categories using a fixed set of rules applied in
priority order.</p>
"""
    s += table(["Category", "Description", "Images", "Percentage", "Mean IoU"], [
        ["<code>pass</code>", "Strong agreement with the reference mask", "26", "23.6%", "0.833"],
        ["<code>total_miss</code>", "The egg region was effectively missed", "2", "1.8%", "0.000"],
        ["<code>disjoint_wrong_region</code>", "The prediction landed on a separate, unrelated region", "13", "11.8%", "0.000"],
        ["<code>mislocalized</code>", "The predicted region was misplaced, with limited overlap", "5", "4.5%", "0.119"],
        ["<code>bleed_overseg</code>", "The egg was partly captured, but the mask spilled into background", "56", "50.9%", "0.228"],
        ["<code>partial_underseg</code>", "Only part of the reference region or one of several eggs was captured", "7", "6.4%", "0.339"],
        ["<code>mixed_moderate</code>", "A combination of moderate segmentation errors", "1", "0.9%", "0.415"],
    ], col_widths=["27%", "38%", "10%", "14%", "11%"])
    s += figure_grid([
        ("task3_outputs/plots/category_counts.png",
         "<b>Figure 7 - Images per category.</b> <code>bleed_overseg</code> is the largest category, accounting for 56 of 110 images."),
        ("task3_outputs/plots/area_ratio_vs_iou.png",
         "<b>Figure 8 - Area ratio versus IoU.</b> Many low-IoU <code>bleed_overseg</code> predictions have a much larger area than the reference mask."),
    ], img_width="230")
    s += """
<p><code>bleed_overseg</code> is the dominant failure mode, affecting 50.9% of the images. Its median
predicted-area-to-reference-area ratio is 4.21, meaning that the predicted masks in this category are typically
several times larger than the reference egg region. This provides quantitative evidence of the background bleed
observed in Task 2.</p>

<h3 class="pb">3.3.1 Separating the Main Contributing Factors</h3>
<p>To examine the effect of Task 2's single-component design, each Task 1 reference mask was checked for
multiple instances. An image was counted as multi-instance when the reference mask contained at least two
connected components, each occupying at least 3% of the mask area.</p>
"""
    s += table(["Category", "Images in category", "Multi-instance images", "Multi-instance rate"], [
        ["<code>pass</code>", "26", "3", "11.5%"],
        ["<code>total_miss</code>", "2", "1", "50.0%"],
        ["<code>disjoint_wrong_region</code>", "13", "1", "7.7%"],
        ["<code>mislocalized</code>", "5", "3", "60.0%"],
        ["<code>bleed_overseg</code>", "56", "3", "5.4%"],
        ["<code>partial_underseg</code>", "7", "7", "100.0%"],
        ["<code>mixed_moderate</code>", "1", "0", "0.0%"],
    ], col_widths=["32%", "22%", "24%", "22%"])
    s += """
<p>The dataset's overall multi-instance rate is <b>16.4% (18/110 images)</b>. The results point to two main
contributing factors. These factors can co-occur in an image; the table above assigns each image to only one
failure category.</p>
<ul>
<li><b>Factor A - single-object color-model failure.</b> The <code>bleed_overseg</code> and
<code>disjoint_wrong_region</code> categories contain 69 images (62.7% of the dataset). Their multi-instance
rates (5.4% and 7.7% respectively) are below the dataset baseline. This suggests that the dominant bleed and
wrong-region failures are generally not explained by multiple eggs; they are more consistent with GrabCut's
color model spilling into similar-looking background or selecting an unrelated region.</li>
<li><b>Factor B - no multi-instance handling.</b> All 7 <code>partial_underseg</code> cases are multi-instance
images. Multi-instance scenes also occur in 3 of the 5 <code>mislocalized</code> cases and 1 of the 2
<code>total_miss</code> cases. That is 11 images (10.0% of the dataset) in these failure categories where a
multi-instance scene coincides with a failure associated with missing or mislocalizing eggs. Task 2's design
selects one connected component per image, so it cannot reliably represent multiple separate eggs.</li>
</ul>

<h2 class="pb">3.4 Qualitative Analysis</h2>
<p>The following examples illustrate the main failure categories and contrast them with stronger results.
Figures 9-11 show background bleed and wrong-region predictions (Factor A), Figures 12-13 show multi-instance
failures (Factor B), and Figures 14-15 show stronger results. In each image, yellow marks agreement between the
prediction and the reference, green marks reference pixels that were missed, and red marks false-positive
prediction pixels.</p>
"""
    s += figure_grid([
        (QUAL + "bleed_overseg__not_damaged_43_jpg.rf.66c31342150e751c30214786e72c8683.png",
         "Figure 9. <code>not_damaged_43</code> (IoU = 0.269, precision = 0.291, recall = 0.782). The egg is captured, but the predicted mask also spreads across the laptop body through a dark, connected region."),
        (QUAL + "bleed_overseg__not_damaged_135_jpg.rf.a6d81ffb82470e031fb3ab469c8f0db2.png",
         "Figure 10. <code>not_damaged_135</code> (IoU = 0.057, precision = 0.057, recall = 1.000). The reference egg is contained in the prediction, but a large false-positive region around the bead-filled bucket overwhelms the mask."),
        ("task3_outputs/worst_cases/not_damaged_123_jpg.rf.11d40a86277d3ab3fce1c5c6cbf14db0_compare.png",
         "Figure 11. <code>not_damaged_123</code> (IoU = 0.000). The prediction falls on a plastic rail of the egg tray, while the reference egg region is untouched - a complete wrong-region capture."),
        (QUAL + "mislocalized__not_damaged_114_jpg.rf.7ccf87a5610bafd963468463b2e2907c.png",
         "Figure 12. <code>not_damaged_114</code> (IoU = 0.170, precision = 0.228, recall = 0.400). In this multi-egg marble scene, one egg is partly captured while the other eggs are missed."),
    ])
    s += "<pdf:nextpage />"
    s += figure_grid([
        (QUAL + "partial_underseg__not_damaged_133_jpg.rf.a386f448c321cfb45a902bab446a0536.png",
         "Figure 13. <code>not_damaged_133</code> (IoU = 0.322, precision = 0.995, recall = 0.322). The selected egg is segmented cleanly, but the remaining eggs are missed, resulting in low recall."),
        (QUAL + "pass__not_damaged_34_jpg.rf.4be2a53f051e4372100c90f2d36ea129.png",
         "Figure 14. <code>not_damaged_34</code> (IoU = 0.994). This is a near-pixel-perfect match with the Task 1 reference mask."),
        (QUAL + "pass__not_damaged_161_jpg.rf.fd216aa5e11e147fc1527478cceae94e.png",
         "Figure 15. <code>not_damaged_161</code> (IoU = 0.515). The main egg region is located, but part of the lower boundary is missed, likely because the shadowed rim is difficult to distinguish using the colour model."),
    ])
    s += """
<h2>3.5 Discussion and Suggested Improvements</h2>
<p>The quantitative and qualitative results identify two main improvement areas:</p>
<ul>
<li><b>Reduce bleed and wrong-region captures.</b> GrabCut relies on foreground and background colour
statistics and does not understand object boundaries or the concept of an egg. An edge-aware refinement step,
boundary constraint, or saliency/objectness prior could help prevent the mask from spreading into similarly
coloured backgrounds or settling on an unrelated region.</li>
<li><b>Support multiple eggs.</b> The current algorithm retains one connected component per image. Retaining
multiple sufficiently large candidate components, followed by appropriate filtering, could allow the algorithm
to represent several eggs in the same frame and address the <code>partial_underseg</code> cases.</li>
</ul>
<p>These improvements were not implemented within the Task 2 trial budget. The results show that the final
classical pipeline can produce strong masks under favourable conditions, but its global colour model and
single-component selection impose clear limits on varied backgrounds and multi-egg scenes.</p>
"""
    return s


# ----------------------------------------------------------------------------
# Section 4 - Task 4
# ----------------------------------------------------------------------------
def section4():
    s = """
<h1>Section 4: Task 4 - Egg Tracking in Video</h1>

<h2>4.1 Objective</h2>
<p>Run an egg segmentation and tracking pipeline on the provided YouTube video and keep a separate identity for
each egg. I used the clip from 0:09 to 0:40 (31 s, 1280x720, 24 fps, 744 frames). The clip contains two eggs:
one on the table and one held in a hand.</p>

<h2>4.2 Why the Task 2 algorithm was not reused</h2>
<p>Section 3 shows that the Task 2 pipeline over-segments often (mean IoU 0.3436 against the Task 1 masks,
bleed/over-segmentation in 50.9% of images). For the video I wrote a simpler detector that outputs boxes only,
since tracking needs object positions rather than exact masks. Its thresholds were tuned on this one clip.</p>

<h2>4.3 Detector</h2>
<p>File: <code>task4_step2_detector.py</code>. Steps per frame:</p>
<ul>
<li>Grayscale, then a fixed threshold of 150.</li>
<li>Only the region below the table edge (<code>TABLE_TOP_Y = 255</code>) is searched.</li>
<li>Morphological open then close, 5x5 elliptical kernel, 2 iterations each.</li>
<li>Contour filters: area 3000-50000 px, minimum side 60 px, circularity &gt;= 0.60, aspect ratio &lt;= 1.8,
mean HSV saturation &lt;= 55 (eggs are white and low in saturation).</li>
</ul>
<p>The detector finds the egg on the table reliably. It rarely finds the held egg. Lowering the threshold to
100 merges the egg with the hand, wall and shirt. A threshold of 150 fragments it, and a saturation mask
(S &lt; 12 and V &gt; 110) isolates only the table egg. The causes are finger occlusion, motion blur and a bright
background. This is the main reason a short-term tracker was added (Section 4.5).</p>

<h2>4.4 Tracker</h2>
<p>File: <code>sort_tracker.py</code>, a custom SORT implementation. Each track has a Kalman filter (predict,
then update). Detections are matched to predicted boxes by IoU using the Hungarian algorithm. Parameters:
<code>max_age=15, min_hits=2, iou_threshold=0.25, lost_buffer_age=120</code>.</p>
<p>I made four changes to the base tracker:</p>
<ul>
<li>Keep a <code>last_bbox</code> for each track.</li>
<li>Set velocity to zero when a track was not updated, so predictions do not drift.</li>
<li>Store dead tracks with their <code>last_bbox</code> instead of a drifted prediction.</li>
<li>Output only tracks updated in the current frame (<code>time_since_update == 0</code>) with enough hits, so
coasting tracks never produce ghost boxes.</li>
</ul>
<p>Dead tracks are kept for <code>lost_buffer_age</code> frames and are revived when a new detection appears near
the last known position, reusing the old ID. So the tracker reports 7 tracks born, but only 2 distinct IDs are
ever drawn.</p>

<h2>4.5 Short-term tracking with CSRT</h2>
<p>File: <code>task4_step4_csrt.py</code> (requires <code>opencv-contrib-python</code>). When the detector loses
an egg that already has an ID, a CSRT tracker started from its last detected box follows it and draws a thin box
labelled "(est.)". Guards that stop CSRT from drawing wrong boxes:</p>
<ul>
<li>Maximum coasting of 120 frames (5 s).</li>
<li>The box area must stay within 0.4 to 2.5 times the last detected area.</li>
<li>The estimated box is dropped if it overlaps another egg's box with IoU &gt; 0.30.</li>
<li>The box centre must stay inside the frame.</li>
<li>After CSRT gives up, it only restarts on the next real detection.</li>
</ul>

<h2>4.6 Development log</h2>
"""
    s += table(["#", "Change or trial", "Observation", "Decision"], [
        ["1", "Threshold 150 plus shape and saturation filters", "Reliable on the table egg, misses the held egg", "Kept"],
        ["2", "Threshold lowered to 100 to catch the held egg", "Egg merges with hand, wall and shirt", "Discarded"],
        ["3", "Saturation mask (S &lt; 12, V &gt; 110)", "Isolates only the table egg", "Discarded"],
        ["4", "Plain SORT output", "Ghost boxes: coasting tracks were drawn", "Fixed: output only tracks updated this frame"],
        ["5", "Zero velocity when not updated, store <code>last_bbox</code>", "Predicted boxes no longer drift away", "Kept"],
        ["6", "<code>lost_buffer_age = 120</code> with revival", "Same ID returns when the egg reappears (7 born, 2 IDs)", "Kept"],
        ["7", "First CSRT script", "Bug: drawn boxes leaked into the tracker input frame", "Fixed with <code>clean = frame.copy()</code>"],
        ["8", "CSRT guards (area ratio, overlap, centre, max coast)", "No stray &quot;(est.)&quot; boxes in the full video", "Kept"],
    ], col_widths=["5%", "35%", "38%", "22%"])
    s += """
<h2>4.7 Results</h2>
<p>There are no ground-truth annotations for the video, so I report box coverage and visual checks. The maximum
possible is 744 frames x 2 eggs = 1488 egg-frames.</p>
"""
    s += table(["Quantity", "Egg-frames", "Share of 1488"], [
        ["Boxes from detector + tracker", "1231", "82.7%"],
        ["Boxes filled by CSRT", "223", "15.0%"],
        ["Total with a box", "1454", "97.7%"],
        ["No box", "34", "2.3%"],
    ], col_widths=["50%", "25%", "25%"])
    s += """
<p>I watched the full output video. I saw no label swaps between the two eggs, and no "(est.)" box remained on
the hand, shirt or background after an egg left, although during holding the estimated box can be offset from
the egg (Figure 4.2). The roles of Egg 0 and Egg 1 (table or held) swap over time because the eggs are exchanged
in the video. This is not a label swap: the IDs follow the earlier ID segments.</p>
"""
    s += figure("task4_figures/compare_11.5s.png",
                "Figure 4.1 - t = 11.5 s: held egg, detector and tracker only (left) and final output with CSRT (right). The estimated box covers the egg but is slightly shifted to its right.",
                width="95%")
    s += figure("task4_figures/compare_12.5s.png",
                "Figure 4.2 - t = 12.5 s: the CSRT box is offset toward the fingers and covers only part of the egg.",
                width="95%")
    s += figure("task4_figures/compare_21.5s.png",
                "Figure 4.3 - t = 21.5 s: the other egg (Egg 0) is held and the CSRT box fits it well. Egg 1 is now the one on the table.",
                width="95%")
    s += """
<h2>4.8 Output video</h2>
<p>The final video (detector + tracker + CSRT) is available at:
<a href="%s">%s</a></p>
<p>Solid boxes come from the detector. Thin boxes labelled "(est.)" come from CSRT, meaning the egg was not
detected in that frame and the position is an estimate.</p>

<h2>4.9 Limitations</h2>
<ul>
<li>CSRT boxes are approximate. They keep the right ID on the held egg, but they are not tight and can be offset
toward the fingers (visible at 12.5 s in Figure 4.2). I did not measure box accuracy against annotations.</li>
<li>The detector almost never finds the held egg, so the held egg is mostly tracked by CSRT.</li>
<li>Thresholds and filters were tuned on this one clip and may not transfer to other videos.</li>
<li>The setup assumes at most two eggs. The detector never produced three or more in this clip.</li>
<li>There is no quantitative tracking metric (such as MOTA), because I have no tracking annotations.</li>
</ul>

<h2>4.10 Possible improvements</h2>
<ul>
<li>Use the Task 1 model (OWL-ViT + SAM) as the detector on video frames, so the held egg is detected directly.</li>
<li>Annotate a few frames by hand and measure the IoU of the CSRT boxes.</li>
<li>Add appearance features to the association step for crowded scenes.</li>
</ul>
""" % (DRIVE_URL, DRIVE_URL)
    return s


# ----------------------------------------------------------------------------
# Sections 5 and 6
# ----------------------------------------------------------------------------
def section5():
    s = """
<h1>Section 5: Setup and Reproduction</h1>
<p>Tested on Windows with Python and a virtual environment.</p>
<pre>
git clone %s.git
cd nosh-egg-segmentation
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
</pre>
<p>Notes: Task 4 needs <code>opencv-contrib-python</code> for CSRT. Videos (<code>*.mp4</code>), the dataset
folders and model weights (<code>*.pth</code>, <code>*.pt</code>) are not stored in the repository. Download the
dataset and the SAM checkpoint separately. The Task 4 clip is the 0:09-0:40 part of the video, saved as
<code>task4_trimmed.mp4</code>.</p>
""" % GITHUB_URL
    s += table(["Task", "Command", "Output"], [
        ["Task 1 segmentation (OWL-ViT + SAM)", "<code>python task1_segment_eggs.py</code>", "<code>task1_outputs_final/</code>"],
        ["Task 1 report figures (1-4)", "<code>python task1_make_figures.py</code>", "<code>task1_figures/</code>"],
        ["Task 2 dev/eval split", "<code>python task2_step0_split.py</code>", "split of the 110 images (dev 88 / eval 22)"],
        ["Task 2 trials 1-9 (experiment log)", "<code>task2_step1_otsu.py</code> ... <code>task2_step9_openbreak.py</code>", "dev-set trial outputs"],
        ["Task 2 final algorithm", "<code>python task2_segment_eggs.py</code>", "<code>task2_outputs_final/</code>"],
        ["Task 3 metrics", "<code>python task3_evaluate.py</code>", "<code>task3_outputs/</code>"],
        ["Task 3 failure categories", "<code>python task3_failure_modes.py</code>", "category table"],
        ["Task 3 multi-instance check", "<code>python task3_multiinstance_check.py</code>", "Section 3.3.1 table"],
        ["Task 3 plots", "<code>python task3_plots.py</code>", "<code>task3_outputs/plots/</code>"],
        ["Task 3 qualitative examples", "<code>python task3_qualitative.py</code>", "<code>task3_outputs/qualitative/</code>"],
        ["Task 4 detection + tracking", "<code>python task4_step3_track.py</code>", "<code>task4_output.mp4</code>"],
        ["Task 4 with CSRT", "<code>python task4_step4_csrt.py</code>", "<code>task4_output_csrt.mp4</code>"],
        ["Task 4 comparison figures", "<code>python task4_make_figures.py</code>", "<code>task4_figures/</code>"],
        ["Build this report", "<code>python build_pdf.py</code>", "<code>report.pdf</code>"],
    ], col_widths=["28%", "42%", "30%"])
    return s


def section6():
    return """
<h1>Section 6: AI Tools Disclosure</h1>
<p>I used Claude (by Anthropic) as an assistant during this project. It helped me with writing and debugging
code, planning the experiments, and drafting and formatting the report. I ran all code myself, checked the
outputs and videos, and made the final decisions about which methods to keep. I can explain every method in this
report.</p>
"""


# ----------------------------------------------------------------------------
def main():
    html = ("<html><head><meta charset=\"utf-8\"><style>" + CSS + "</style></head><body>" +
            cover() + section1() + section2() + section3() + section4() + section5() + section6() +
            "</body></html>")
    with open(OUTPUT_PDF, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)
    if result.err:
        print("PDF generation reported errors:", result.err)
    print("Wrote", OUTPUT_PDF)
    print("Images embedded:", len(FOUND_IMAGES))
    if MISSING_IMAGES:
        print("MISSING IMAGES (%d):" % len(MISSING_IMAGES))
        for m in MISSING_IMAGES:
            print("  [MISSING IMAGE]", m)
    else:
        print("No missing images.")


if __name__ == "__main__":
    main()