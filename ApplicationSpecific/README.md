# CHIRP Dataset: Application-Specific Benchmark

This benchmark evaluates how well a combined computer vision pipeline recovers biologically meaningful measurements from feeder videos. It goes beyond task-specific metrics (accuracy, F1, etc.) to measure the downstream impact on actual biological quantities.

## Quick Start

```bash
python ApplicationSpecific/ComputeMetrics.py \
    --dataset_path /path/to/CHIRP_Dataset \
    --inference_dir /path/to/InferenceData \
    --output_dir ./results
```

Published inference results for all models and trackers in the paper are in `ApplicationSpecific/InferenceData/`. To reproduce all paper results, point `--inference_dir` at that directory.

---

## Requirements

```bash
pip install pandas numpy scikit-learn
```
---

## How the Benchmarking Works

The benchmark evaluates a **modular 3-stage pipeline** that processes raw video into individual-level behavioural measurements:

```
Video → [1. Detection + Tracking] → [2. Behaviour Classification] → [3. Re-Identification] → Biological Metrics
```

**Stage 1 — Detection & Tracking**: Detect birds in each frame and link detections across frames into consistent anonymous tracks (e.g. `track_1`, `track_2`).

**Stage 2 — Behaviour Classification**: For each track, classify short overlapping clip windows as `Peck` or `Other`.

**Stage 3 — Re-Identification**: Map each anonymous track ID to a known individual bird ID (e.g. `bird_A`) using colour ring appearance.

The three stages produce three inference files per video. The `ComputeMetrics.py` script loads all three, assembles the full pipeline, and computes biological metrics against ground truth.

### Modularity

The pipeline is modular, so you dont need to recompute everything everytime, except for detection and tracking. So if you want to benchmark re-id or action recognition, you can take the existing detection + tracking results and run your algorithm over it. Make sure you save the inference results in the correct format for the script to read it (see [Formatting Requirements](#formatting-requirements)).

If you would like to update the object detection + tracking results, then you will have to rerun re-id and behaviour inference (see[Rerun Corvid and Action Recognition](#rerun-corvid-and-action-recognition))

The script then automatically discovers all valid algorithm combinations from the filenames present in the inference directory (see [Algorithm Discovery](#algorithm-discovery)).

---

## Biological Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Feed Rate** | Pecks per minute per individual | Mean/Median absolute error + Pearson correlation |
| **PropTime** | Proportion of frames where a bird pair co-occurs | Absolute error + Pearson correlation |
| **Behaviour Classification** | Peck detection accuracy | Precision / Recall / F1 in 1-second (25-frame) windows |
| **Frame Accuracy** | Correct bird ID + bbox IoU ≥ 0.5 | % of frames |

---

## Formatting Requirements

### Detection + Tracking 

Provide one file per validation video named:
```
{video}_TrackedBBox_{tracker}.csv
```

CSV format — columns: `Frame, ID, Xmin, Ymin, Xmax, Ymax`

```
Frame,ID,Xmin,Ymin,Xmax,Ymax
0,track-1,498,360,929,589
0,track-2,102,215,430,480
1,track-1,501,362,933,592
...
```

- `ID` is an arbitrary but **consistent** track identifier across frames (e.g. `track-1`)
- Use the published `BehavDict` and `IDMatch` files from `InferenceData/` for stages 2 and 3

### Behaviour Classification 

Provide one pickle file per validation video named:
```
{video}_BehavDict_{model}_{tracker}.p
```

Python pickle containing a nested dict of track → time-window → behaviour label:

```python
{
    "track_1": {
        (0, 25):   "Other",
        (25, 50):  "Peck",
        (50, 75):  "Other",
        ...
    },
    "track_2": {
        (0, 25):   "Other",
        ...
    },
}
```

- Keys are `(start_frame, end_frame)` tuples for each clip window
- Values are `"Peck"` or `"Other"`

### Re-Identification 

Provide one file per validation video named:
```
{video}_{id_algo}_IDMatch_{tracker}.p
```

Python pickle mapping track IDs to individual bird IDs:

```python
{
    "track_1": "bird_A",
    "track_2": "bird_B",
    "track_3": "unringed",  # use "unringed" for tracks that could not be identified
}
```

- Bird IDs should match the individual IDs in `GroundTruth/`
- Use `"unringed"` for any track that cannot be matched to a known individual
- Use the published `TrackedBBox` and `BehavDict` files from `InferenceData/` for stages 1 and 2

---

## Validation Videos

The script evaluates on these 12 validation videos:

```
20210909_OiFHBq_1    20210909_CFfo1Z      20210914_h0h9CW      20210922_GWDswk
20211018_WS3GWi      20220816_RB15Aq      20220829_BrmjDE      20220901_WS3GWi
20220922_ebqkf4      20221004_ukRCut      20221009_WS3GWi      20221015_csG6Ba_2
```

---

## Required Directory Structure

### Dataset Directory
```
CHIRP_Dataset/
└── ApplicationSpecific/
    ├── MetaData.csv                        # Video metadata (required)
    ├── GroundTruth/                        # Ground truth annotations (required)
    │   ├── {video}_BBox.csv
    │   ├── {video}_PeckEvents.csv
    │   └── ...
    ├── InferenceData/                      # Published inference results
    │   ├── {video}_TrackedBBox_{tracker}.csv
    │   ├── {video}_BehavDict_{model}_{tracker}.p
    │   ├── {video}_{id_algo}_IDMatch_{tracker}.p
    │   └── ...
    └── HumanBenchmarkBORIS/                # Human benchmark (optional)
        ├── {video}_{bird_id}.tsv
        └── ...
```

### Inference Directory Structure
Your inference directory should follow this naming convention exactly:
```
InferenceData/
├── {video}_TrackedBBox_{tracker}.csv           # Stage 1: tracking output
├── {video}_BehavDict_{model}_{tracker}.p       # Stage 2: behaviour output
├── {video}_{id_algo}_IDMatch_{tracker}.p       # Stage 3: re-ID output
└── ...
```

All 12 validation videos must have a corresponding file for every stage you are submitting.

---

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset_path` | Yes | Path to CHIRP dataset root directory |
| `--inference_dir` | Yes | Directory containing inference result files |
| `--output_dir` | No | Output directory (default: `./results`) |
| `--force` | No | Force recomputation of existing results |

---

## Output Files

For each algorithm combination discovered, the script writes:

| File | Description |
|------|-------------|
| `{id_algo}_{tracker}_{model}_metrics.pkl` | Complete raw metrics |
| `{id_algo}_{tracker}_{model}_peck_rates.csv` | Per-bird feed rate predictions vs. ground truth |
| `{id_algo}_{tracker}_{model}_proptime.csv` | Per-pair co-occurrence predictions vs. ground truth |
| `{id_algo}_{tracker}_{model}_precision_recall.csv` | Peck classification metrics |
| `{id_algo}_{tracker}_{model}_summary.pkl` | Frame accuracy summary |

Two aggregate summary files are also written:
- **`comprehensive_summary.csv`**: All metrics for all combinations, including human and random baselines
- **`ranking_summary.csv`**: Top performers ranked by feed rate accuracy and PropTime accuracy

---

## Metrics Detail

### Feed Rate Accuracy
- Counts pecks per bird per video, normalised by video duration (pecks/minute)
- Reports mean, median, and std of absolute error across birds and videos
- Reports Pearson correlation between predicted and ground truth feed rates

### Social Interaction Timing (PropTime)
- For each pair of birds, counts frames where both are simultaneously present in the frame
- Divided by total video frames to get a proportion
- Reports absolute error and correlation between predicted and ground truth

### Behaviour Classification
- Divides each video into 1-second (25-frame) windows
- A window is labelled `Peck` if any peck event falls within it (ground truth) or if any track's BehavDict entry is `Peck` (predicted)
- Reports precision, recall, and F1

### Frame Accuracy
- For each frame, checks if the correct bird ID is assigned to a detection with IoU ≥ 0.5 against the ground truth bounding box
- Reports the percentage of frames where this holds across all birds and videos

---

## Algorithm Discovery

`ComputeMetrics.py` automatically discovers all algorithm combinations by scanning filenames in the inference directory:

- **Trackers** — extracted from `_TrackedBBox_{tracker}.csv` filenames
- **Models** — extracted from `_BehavDict_{model}_{tracker}.p` filenames
- **ID algorithms** — extracted from `_{id_algo}_IDMatch_{tracker}.p` filenames

All valid three-way combinations are processed. Results for existing combinations are skipped unless `--force` is passed.

---



## Rerun CORVID and Action Recognition

If you update the detection + tracking results (Stage 1), you need to rerun Stage 2 (behaviour) and Stage 3 (re-ID) against the new TrackedBBox files. Two scripts are provided for this:

### Re-Identification: `RunCORVID.py`

Runs ring segmentation (Mask2Former) followed by CORVID colour-ring matching. Automatically discovers all trackers from the files present in `--bbox_dir`.

To run this script, make sure you have [MMDetection](https://mmdetection.readthedocs.io/en/latest/) installed, for the ring segmentation.

```bash
python ApplicationSpecific/RunCORVID.py \
    --video_dir  /path/to/videos \
    --bbox_dir   /path/to/InferenceData \
    --output_dir /path/to/InferenceData \
    --weights_dir /path/to/CHIRP_Dataset/Weights \
    --metadata   /path/to/CHIRP_Dataset/ApplicationSpecific/MetaData.csv
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--video_dir` | Yes | Directory containing `{video}.mp4` files |
| `--bbox_dir` | Yes | Directory containing `{video}_TrackedBBox_{tracker}.csv` files |
| `--output_dir` | Yes | Where to save `CORVID_IDMatch` and intermediate `RingSeg` pickles |
| `--weights_dir` | Yes | Root weights directory (auto-discovers segmentation + RF weights) |
| `--metadata` | Yes | Path to `MetaData.csv` (provides valid bird IDs per video) |

Outputs per video/tracker:
- `{video}_RingSeg_{tracker}.p` — intermediate ring segmentation (cached, reused on re-runs)
- `{video}_CORVID_IDMatch_{tracker}.p` — final ID matching result

### Behaviour Classification: `RunBehav.py`

Runs C3D action recognition on tracked bird crops. Automatically discovers all trackers from `--bbox_dir`.

To run this script, make sure you have [MMAction2](https://mmaction2.readthedocs.io/en/latest/) installed, for the ring segmentation.

```bash
python ApplicationSpecific/RunBehav.py \
    --video_dir  /path/to/videos \
    --bbox_dir   /path/to/InferenceData \
    --output_dir /path/to/InferenceData \
    --weights_dir /path/to/CHIRP_Dataset/Weights
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--video_dir` | Yes | Directory containing `{video}.mp4` files |
| `--bbox_dir` | Yes | Directory containing `{video}_TrackedBBox_{tracker}.csv` files |
| `--output_dir` | Yes | Where to save `BehavDict` pickles |
| `--weights_dir` | Yes | Root weights directory (auto-discovers C3D weights) |
| `--device` | No | CUDA device string (default: `cuda:0`) |

Outputs per video/tracker:
- `{video}_BehavDict_C3D_{tracker}.p`

### Notes

- Both scripts skip files that already exist — safe to re-run after partial completion.
- `--bbox_dir` and `--output_dir` can point to the same directory (e.g. `InferenceData/`).
- Weights are loaded from subdirectories of `--weights_dir`:
  - `Segmentation/RingSegMask2Former.{py,pth}` — ring segmentation model
  - `CORVID/RandomForestModel.p` + `TrainImagesFeatures.p` — colour classifier
  - `ActionRecognition/c3d_CHIRP-rgb.py` + `c3d_weights.pth` — behaviour model

---
