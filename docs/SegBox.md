# Bird Segmentation, Bounding Box & Colour Ring Segmentation

## Description

Two complementary segmentation datasets sharing a common image source:

**BirdBoxMask** — Object detection and instance segmentation for 3 classes (Bird, Feeder, Food) across 1,762 images (1,669 annotated instances). Includes both manually annotated masks and SAM2-generated masks (`Mask_SAM`).

**RingMask** — Fine-grained segmentation of colour rings on bird legs, covering 12 ring colour classes across 944 images. Supports both multi-class and single-class (ring vs. background) evaluation.

Both datasets provide annotations in COCO JSON and CSV formats.

## File Structure

```
BirdBoxMask/
├── Images/                         # 1,762 .jpg images
│   └── {MMDDYYYY}_Cam{id}_{code}_F{frame}.jpg
└── Annotations/
    ├── train.json                  # COCO bbox + segmentation — train split
    ├── test.json                   # COCO bbox + segmentation — test split
    ├── train_masks.json            # COCO segmentation masks only — train split
    ├── test_masks.json             # COCO segmentation masks only — test split
    ├── train.csv                   # CSV format — train split
    └── test.csv                    # CSV format — test split

RingMask/
├── Images/                         # 944 .jpg images
│   └── {uuid}-{YYYYMMDD}_{location}_F{frame}_{species}_{N}.jpg
└── Annotations/
    ├── train.json                  # COCO multi-class ring segmentation — train split
    ├── test.json                   # COCO multi-class ring segmentation — test split
    ├── train_singleclass.json      # COCO single-class (ring/background) — train split
    ├── test_singleclass.json       # COCO single-class (ring/background) — test split
    ├── train.csv                   # CSV format — train split
    └── test.csv                    # CSV format — test split
```

### BirdBoxMask Classes

| Class | Description |
|-------|-------------|
| Bird | Full bird |
| Feeder | Feeding stick |
| Food | Food items visible at the feeder |

**Annotation types:** Bounding box, instance segmentation mask (manual), `Mask_SAM` (SegmentAnything2 auto-generated masks)

### RingMask Classes

12 colour ring codes corresponding to ring colours used to identify individual birds (e.g. combinations of red, blue, green, yellow, white, black rings). The single-class variants treat all rings as one foreground class, useful for ring detection before colour classification.


## Visualization

Use `tools/VisualizeImages.py` to inspect bounding boxes, segmentation masks, and ring annotations.

```bash
# BirdBoxMask — COCO format (bboxes + masks)
python tools/VisualizeImages.py --annot BirdBoxMask/Annotations/train.json --image BirdBoxMask/Images/

# BirdBoxMask — CSV format
python tools/VisualizeImages.py --annot BirdBoxMask/Annotations/train.csv --image BirdBoxMask/Images/

# RingMask — COCO format
python tools/VisualizeImages.py --annot RingMask/Annotations/train.json --image RingMask/Images/

# RingMask — CSV format (ring polygons with colour overlay)
python tools/VisualizeImages.py --annot RingMask/Annotations/train.csv --image RingMask/Images/
```

Bounding boxes are drawn in green with class labels. Segmentation masks are drawn as cyan outlines. Ring annotations are shown as semi-transparent filled polygons.

**Controls:** `n` = next image, `q` = quit
