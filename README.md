# HVAC Duct Annotator — Hybrid CV + VLM Pipeline

## Overview

This tool automatically analyzes HVAC floor plan PDFs and produces a fully annotated output image overlaid with:

- **Duct centerlines** with unique IDs and real-world lengths (in feet)
- **Duct dimension labels** (e.g., `12x10`, `18" ⌀`) classified by type (Supply / Return / Exhaust)

The core design philosophy is a **hybrid pipeline**: computationally cheap geometric tasks are handled by classical Computer Vision (OpenCV), while semantically complex tasks — reading dimension annotations, understanding symbols, and identifying duct types — are delegated to a Vision-Language Model (Google Gemini).

---

## Architecture: Hybrid Workflow

```
PDF Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Classical CV — Duct Centerline Detection       │
│  (OpenCV · HoughLinesP · LUT · Morphology · fitLine)    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: VLM — Semantic Label Extraction                │
│  (Google Gemini · JSON-structured prompt · SHA-256 Cache)│
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Double Annotation & Rendering                  │
│  (PIL text rendering · OpenCV line drawing)              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
final_assignment_output.png  +  duct_lines.json
```

---

## Stage 1 — Classical CV: Duct Centerline Detection

Ducts in HVAC drawings appear as **parallel line pairs** representing the two walls of a duct. The goal is to detect these pairs and reduce them to a single centerline.

### Steps:

#### 1.1 Background Cleaning & Contrast Enhancement

A custom **Look-Up Table (LUT)** is applied to the grayscale image to crush the light gray background to white and snap dark lines to pure black. This isolates structural duct lines from noise, grid marks, and faint annotations.

```python
xp, fp = [0, 45, 65, 255], [0, 0, 255, 255]
lut = np.interp(np.arange(256), xp, fp).astype('uint8')
```

![Background Removal Result](intermediate_background_removal.png)

#### 1.2 Region of Interest Masking

A rectangular mask crops the image to the plan area only (roughly 84% width × 65% height), discarding title blocks, legends, and borders that would otherwise generate false positives.

#### 1.3 Probabilistic Hough Transform

`cv2.HoughLinesP` detects all candidate line segments on the cleaned binary image. Parameters are tuned to favor longer, solid lines (minLineLength=130) while tolerating small gaps within a segment (maxLineGap=15).

#### 1.4 Parallel Pair Detection

Detected line segments are compared pairwise. Two segments are considered a **duct wall pair** if:

- Their angles differ by less than **1.5°** (nearly parallel)
- Their midpoints are between **25 px and 210 px** apart (plausible duct width at 300 DPI)

The midpoints of each qualifying pair are averaged to produce an initial centerline candidate.

#### 1.5 Centerline Merging (`clean_and_group_lines`)

Multiple Hough detections often fragment a single long duct into several short overlapping segments. This function:

1. Sorts candidates by length (longest-first, greedy merge)
2. Groups segments that share a similar angle (within π/45 rad) **and** are collinear (cross-distance < 8 px)
3. Fits a single `cv2.fitLine` (L2) through the entire group's points
4. Projects all points onto the fitted axis to find the true start and end, yielding one clean, continuous centerline per duct

---

## Stage 2 — VLM: Semantic Label Extraction (Google Gemini)

Classical CV cannot reliably read mixed text, special symbols (⌀, φ), or distinguish Supply vs. Return vs. Exhaust ducts from visual context alone. For this, the raw page image is sent to **Google Gemini** (`gemini-3-flash-preview`) via a structured prompt.

### Prompt Design

The prompt instructs Gemini to:

1. Extract the **drawing scale** (e.g., `1/4" = 1'-0"`)
2. Identify all **duct dimension annotations** with their pixel position (normalized to a 1000×1000 grid) and classify each as Supply / Return / Exhaust

The response is requested as **structured JSON** using `response_mime_type="application/json"`, eliminating fragile regex parsing of free-form text:

```json
{
  "scale": "1/4\"=1'-0\"",
  "labels": [
    { "text": "12x10", "type": "Supply", "pos": [320, 450] },
    { "text": "18\" phi", "type": "Exhaust", "pos": [510, 280] }
  ]
}
```

### Scale-Aware Length Calculation

The extracted scale string is parsed with a regex to compute **pixels-per-foot (ppf)**:

```python
ppf = DPI * (numerator / denominator)  # e.g., 300 * 0.25 = 75 px/ft
```

Each centerline's pixel length is then divided by `ppf` to produce a real-world length in feet.

---

## Caching — Avoiding Redundant API Calls

Every call to the Gemini API is **cached on disk** in `vlm_cache.json` using the **SHA-256 hash of the image bytes** as the key.

```python
img_hash = hashlib.sha256(image_bytes).hexdigest()
if img_hash in self.cache:
    return self.cache[img_hash]   # instant local lookup
```

**Why this matters:**

- Re-running the tool on the same PDF (e.g., during debugging or parameter tuning) incurs **zero API cost and zero latency** for the VLM stage.
- The cache is human-readable JSON, making it easy to inspect or invalidate specific entries.
- Handles the common engineering scenario of iterating on the CV pipeline while the semantic extraction result stays constant.

---

## Stage 3 — Double Annotation & Rendering

The final image is produced by compositing two separate rendering passes:

### 3.1 PIL Pass — Duct Dimension Labels

`Pillow (PIL)` is used — not OpenCV — for rendering the VLM-extracted labels. This is a deliberate choice because OpenCV's built-in font does **not support Unicode**, making it unable to render the diameter symbol (⌀/φ) or other special characters common in HVAC drawings.

A cross-platform font resolution chain ensures the correct font is loaded on Windows, macOS, or Linux:

```
Arial (Windows) → Arial (macOS) → DejaVu Sans (Linux) → default fallback
```

Each label is drawn at its Gemini-reported position (coordinates de-normalized from the 1000×1000 grid back to actual pixel space), with a `(S/R/E)` type suffix appended.

### 3.2 OpenCV Pass — Centerline Drawing

After the PIL pass, the image is converted back to an OpenCV array. Each final centerline is drawn with:

- A **color-coded line** (cycling through 5 colors for visual separation)
- A **numeric ID** rendered at the midpoint

Short segments (< 110 px) are filtered out as noise before drawing.

---

## Outputs

| File                          | Description                                                |
| ----------------------------- | ---------------------------------------------------------- |
| `final_assignment_output.png` | Annotated HVAC plan with centerlines, IDs, and duct labels |
| `duct_lines.json`             | Structured data for each detected duct centerline          |
| `vlm_cache.json`              | On-disk cache of Gemini API responses keyed by image hash  |

### Why a Separate JSON for Duct Lines?

Embedding computed lengths directly into the image annotation would make the data inaccessible for downstream use (e.g., bill-of-materials generation, CAD export, length scheduling). `duct_lines.json` separates the **data layer from the visualization layer**, keeping the pipeline composable:

```json
[
  {
    "id": 1,
    "length": "12.4'",
    "coords": [
      [120, 340],
      [1050, 340]
    ]
  },
  {
    "id": 2,
    "length": "8.7'",
    "coords": [
      [430, 200],
      [430, 855]
    ]
  }
]
```

Example: See the label for 22x14 big duct, the label for this is 22x14(S) means this is a supply
duct with dimension 22x14, you will see 27 is written near the duct centerline, which is the line id.
If you open the duct_line.json, the length of this is 5.8'.

---

## Design Rationale Summary

| Task                                    | Approach                       | Why                                                       |
| --------------------------------------- | ------------------------------ | --------------------------------------------------------- |
| Line/geometry detection                 | OpenCV (HoughLinesP + fitLine) | Deterministic, fast, no API cost                          |
| Text reading & duct type classification | Google Gemini VLM              | Cannot be solved reliably by classical CV                 |
| Re-run cost reduction                   | SHA-256 image hash cache       | Avoids redundant API calls during iteration               |
| Unicode label rendering                 | PIL (Pillow)                   | OpenCV cannot render ⌀ / φ symbols                        |
| Structured data export                  | `duct_lines.json`              | Separates data from visualization; enables downstream use |

---

## Requirements

```bash
pip install -r requirements.txt
# requirements.txt should include: google-generativeai, pdf2image, Pillow
```

## Usage

```bash
python duct_annotator.py
```

Place `testset2.pdf` in the same directory. Outputs will be written to `final_assignment_output.png`, `duct_lines.json`, and `vlm_cache.json`.

## Future Work and Potential Enhancements

While the current Gemini-based solution is highly effective, the field of VLMs is rapidly evolving. Future iterations could explore:

- **Open-Source VLMs**: As open-source models mature, it would be beneficial to evaluate self-hosted alternatives to reduce dependency on proprietary APIs and potentially lower operational costs. Promising models include:
  - **LLaVA (Large Language and Vision Assistant)**: A leading open-source VLM that combines Vicuna (a language model) with a vision encoder.
  - **BakLLaVA**: A variant of LLaVA using the Mistral 7B model, known for its strong performance.
  - These could be deployed on-premise or in a private cloud for enhanced data privacy and control.

- **Alternative Cloud APIs**: For production environments where a managed service is preferred, other powerful VLMs could be benchmarked for performance and cost-effectiveness:
  - **Anthropic's Claude 3 (Opus/Sonnet)**: The Claude 3 model family has demonstrated state-of-the-art vision capabilities that are highly competitive with Gemini, making it an excellent candidate for evaluation.

- **Enhanced Error Correction**: Implement a validation layer where results from the VLM are cross-referenced with a more lightweight model or a simplified heuristic to flag potential inconsistencies for human review, creating a human-in-the-loop system for mission-critical applications.
