# Video Re-Identification

## Description

Video re-identification of 182 individual Siberian Jays (*Perisoreus infaustus*) across 16,110 video clips, organized as 25-frame tracklets at 480×480px resolution. The dataset supports three evaluation protocols of increasing difficulty: ClosedSet, DisjointedSet, and OpenSet.

Each tracklet captures a single bird across consecutive frames extracted from feeder monitoring videos. Birds are identified by unique colour ring combinations worn on their legs, providing ground truth identity labels.

## File Structure

```
ReID/
├── Annotation.csv                  # 402,750 rows of tracklet-level annotations
│                                   #   Columns: Video, Tracklet, id, Territory, Year,
│                                   #            img, UnqTrack, ClosedSetSplit,
│                                   #            DisjointedSetSplit, OpenSetSplit, Date
├── PossibleBirds_Territory.csv     # Which birds are possible IDs per territory
├── PossibleBirds_Neighbours.csv    # Neighbouring territory bird presence
└── data/
    └── {bird_id}/                  # One directory per individual (184 total)
        │                           #   e.g. agbo/, arbl/, argy/, arol/, ...
        └── {video}_{territory}/    # Clips per video-territory combination
            └── Tracklet-{NNN}/     # Zero-padded tracklet index
                └── {YYYYMMDD}_{location}_{bird_id}_{N}.jpg
                                    # 25 frames per tracklet, 480×480px
```

## Evaluation Splits

Splits are encoded as columns in `Annotation.csv`:

| Split | Column | Possible Values — Protocol |
|-------|--------|----------|
| **ClosedSet** | `ClosedSetSplit` | Train/Test — same individuals in both sets. Task is to match test samples to train IDs |
| **DisjointedSet** | `DisjointedSetSplit` | Train/Query/Gallery — disjoint individual identities across sets. Task is to match query samples to gallery IDs |
| **OpenSet** | `OpenSetSplit` | Unknown_Test/Known_Test/Known_Train — Some unseen individuals in test set, task is to first classify known or unknown, then assign sample to ID if known.|

## Model annotated dataset
In order to facilitate other methods tharequire different types of input (e.g. masks), we also provide **model-annotated dataset** by using the best performing models for 2D keypoints (vitpose-large) and instance segmentation (mask2former) to ALSO provide keypoints and mask annotation. We note **THESE ARE NOT HUMAN ANNOTATED DATA**

## Visualization

Use `tools/VisualizeReID.py` to browse tracklets as a horizontal strip of sampled frames.

```bash
# Browse tracklets with metadata overlay (territory, split)
python tools/VisualizeReID.py --data ReID/

# Add per-frame keypoints and masks
python tools/VisualizeReID.py --data ReID/ --keypoints --masks
```

Each tracklet is displayed as a strip of 8 evenly-sampled frames from the 25 available, with bird ID, video-territory, and tracklet name in the header.

**Controls:** `n` or `Enter` = next tracklet, `q` = quit
