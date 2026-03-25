# 2D Keypoint Estimation

## Description

Single-frame 2D keypoint estimation dataset with 13 anatomical landmarks annotated per bird instance, covering 1,178 instances across 846 images. Annotations are provided in both COCO JSON and CSV formats to support a wide range of frameworks.

**Keypoints (13 total):**
head bill, left eye, right eye, left shoulder, right shoulder, left tail, right tail, left ankle, right ankle, left foot, right foot, left wing tip, right wing tip

**Visibility states:** `2` = visible, `0` = not visible (occluded or out of frame)

## File Structure

```
Keypoint2D/
├── images/                     # 846 .jpg images
│   └── {MMDDYYYY}_Cam{id}_{code}_F{frame}.jpg
└── annotation/
    ├── keypoint_classes.json   # Keypoint index-to-name mapping
    ├── train.json              # COCO format — train split
    ├── test.json               # COCO format — test split
    ├── train.csv               # CSV format — train split
    └── test.csv                # CSV format — test split
```

### Annotation Formats

**COCO JSON** (`train.json` / `test.json`): Standard COCO keypoint format. Each annotation includes `keypoints` as a flat list of `[x, y, v, x, y, v, ...]` triplets for all 13 keypoints.

**CSV** (`train.csv` / `test.csv`): Each row corresponds to one instance. Keypoint columns are ordered by the index defined in `keypoint_classes.json`, with `[x, y, visibility]` values per keypoint.

## Visualization

Use `tools/VisualizeImages.py` to inspect keypoint annotations on images.

```bash
# COCO JSON format
python tools/VisualizeImages.py --annot Keypoint2D/annotation/train.json --image Keypoint2D/images/

# CSV format
python tools/VisualizeImages.py --annot Keypoint2D/annotation/train.csv --image Keypoint2D/images/
```

Keypoints are drawn as colour-coded circles (one colour per landmark). Only visible keypoints (`v > 0`) are shown.

**Controls:** `n` = next image, `q` = quit
