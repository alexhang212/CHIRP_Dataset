# Action Recognition

## Description

Clip-level action recognition dataset covering 3 bird behaviours across 1,387 labelled video clips. Each clip is a short `.mp4` of a single bird performing one behaviour at a feeder. Alongside the video clips, per-frame 13-keypoint annotations and segmentation masks are provided to support multi-modal approaches.

**Behaviour categories:**
- **Peck** — bird pecking at the food
- **Submissive** — stereotyped submissive behaviour (wing flapping and screeching)
- **Other** — all other behaviours (e.g. vigilance)

## File Structure

```
ActionRecognition/
├── BehaviourAnnotations.csv    # 1,388 rows of clip-level labels
│                               #   Columns: AnnotID, VideoPath, Behaviour, BBox,
│                               #            Corrupted, Split, Date
├── mmaction/
│   ├── train.txt               # mmaction2-compatible train split file
│   └── test.txt                # mmaction2-compatible test split file
├── Videos/                     # 1,574 .mp4 video clips
│   └── {YYYYMMDD}_{location}-{bird_id}_{Behavior}_{index}.mp4
├── Videos_Keypoints/           # 1,387 .csv files — per-frame keypoint annotations
│                               #   Columns: Frame, Cam, ID, Keypoint, x, y, Conf
│                               #   One row per keypoint per frame (long format)
└── Videos_Masks/               # 1,387 .csv files — per-frame segmentation masks
                                #   Columns: Frame, Cam, ID, Class, Segmentation
                                #   Segmentation: JSON nested list (birds → contours → flat x,y coords)
```

### Video Naming Convention

`{YYYYMMDD}_{location}-{bird_id}_{Behavior}_{index}.mp4`

- `YYYYMMDD` — recording date
- `location` — 6-character territory code (e.g. `ukZrTR`)
- `bird_id` — numeric bird identifier within the clip
- `Behavior` — one of `Peck`, `Submissive`, `Other`
- `index` — clip index

Example: `20200826_ukZrTR-0_Peck_119.mp4`

## Model annotated dataset
In order to facilitate other methods that might use multi-modal inputs (e.g. video + keypoints), we also provide **model-annotated dataset** by using the best performing models for 2D keypoints (vitpose-large) and instance segmentation (mask2former) to ALSO provide keypoints and mask annotation. We note **THESE ARE NOT HUMAN ANNOTATED DATA**


## Visualization

Use `tools/VisualizeAccRec.py` to play clips and optionally overlay per-frame keypoints and segmentation masks. The script reads directly from `BehaviourAnnotations.csv` and skips corrupted clips automatically.

```bash
# Play clips only
python tools/VisualizeAccRec.py --data /path/to/ActionRecognition/

# With keypoint and mask overlay
python tools/VisualizeAccRec.py --data /path/to/ActionRecognition/ --keypoints --masks
```

**Controls:** `n` = next clip, `space` = pause/resume, `q` = quit
