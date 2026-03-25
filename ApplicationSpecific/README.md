# CHIRP Dataset: Computing Application Specific Metrics

This script computes all application-specific metrics for the CHIRP dataset. It automatically processes results, and generates detailed performance summaries including human benchmark comparisons.

## Quick Start

```bash
python ComputeMetrics_simplified.py \
    --dataset_path /path/to/CHIRP_Dataset \
    --inference_dir /path/to/your/inference/results \
    --output_dir ./results
```

## Required Directory Structure

Make sure you have downloaded the CHIRP dataset with this [link](https://keeper.mpdl.mpg.de/d/2b19ec1f87e44a41905b/)!

### Dataset Directory Structure
Your dataset path should contain:
```
CHIRP_Dataset/
├── ApplicationSpecific/
│   ├── MetaData.csv                    # Video metadata (required)
│   ├── GroundTruth/                    # Ground truth annotations (required)
│   │   ├── {video}_BBox.csv
│   │   ├── {video}_PeckEvents.csv
│   │   └── ...
│   └── HumanBenchmarkBORIS/            # Human benchmark data (optional)
│       ├── {video}_{bird_id}.tsv
│       └── ...
```

### Inference Results Directory Structure
Your inference directory should contain results following this naming convention:
```
inference_results/
├── {video}_BehavDict_{model}_{tracker}.p       # Behavior predictions
├── {video}_TrackedBBox_{tracker}.csv           # Tracking bounding boxes
├── {video}_{id_algo}_IDMatch_{tracker}.p       # ID matching results
└── ...
```

## Required Files
For all models and pipelines benchmarked in the publication, inference files are available under `CHIRP_Dataset/ApplicationSpecific/InferenceData`. All results can be reproduced by running the script with that set as the inference directory. If you wish to implement your own pipelines, you will have to follow all data conventions below:

### Inference Result Files
For each algorithm combination, you need three files per video:


#### Tracked Bounding Boxes (`{video}_TrackedBBox_{tracker}.csv`)
CSV file with tracking results:
- Columns: `Frame`, `ID`, `Xmin`, `Ymin`, `Xmax`, `Ymax`

#### Behavior Dictionary (`{video}_BehavDict_{model}_{tracker}.p`)
Python pickle file containing behavior predictions:
```python
{
    "track_1": {
        (start_frame, end_frame): "Peck",
        (start_frame2, end_frame2): "Other",
        ...
    },
    "track_2": {...},
    ...
}
```

#### ID Matching (`{video}_{id_algo}_IDMatch_{tracker}.p`)
Python pickle file mapping tracks to ground truth IDs:
```python
{
    "track_1": "bird_A",
    "track_2": "bird_B", 
    "track_3": "unringed",  # Use "unringed" for unmatched tracks
    ...
}
```

## Validation Videos
The script processes these specific validation videos:
- `20210909_fyra_1`
- `20221009_impossible` 
- `20210914_baggins`
- `20211018_impossile`
- `20210909_granmyran`
- `20210922_fello`
- `20220816_sodraguortes`
- `20220829_bengt`
- `20220901_impossible`
- `20221004_guorbatjakka`
- `20221015_djay_2`
- `20220922_mader`

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset_path` | ✅ | Path to CHIRP dataset root directory |
| `--inference_dir` | ✅ | Directory containing inference result files |
| `--output_dir` | ❌ | Output directory (default: `./results`) |
| `--force` | ❌ | Force recomputation of existing results |

## Example Usage

### Basic Usage
```bash
python ComputeMetrics_simplified.py \
    --dataset_path /data/CHIRP_Dataset \
    --inference_dir /results/my_inference \
    --output_dir ./my_results
```

### Force Recomputation
```bash
python ComputeMetrics_simplified.py \
    --dataset_path /data/CHIRP_Dataset \
    --inference_dir /results/my_inference \
    --output_dir ./my_results \
    --force
```

## Output Files

The script generates several output files for each algorithm combination:

### Individual Algorithm Results
- **`{id_algo}_{tracker}_{model}_metrics.pkl`**: Complete metrics data
- **`{id_algo}_{tracker}_{model}_peck_rates.csv`**: Feed rate analysis
- **`{id_algo}_{tracker}_{model}_proptime.csv`**: Social interaction timing
- **`{id_algo}_{tracker}_{model}_precision_recall.csv`**: Behavior classification metrics
- **`{id_algo}_{tracker}_{model}_summary.pkl`**: Frame accuracy summary

### Summary Files
- **`comprehensive_summary.csv`**: All metrics for all combinations (including human benchmark)
- **`ranking_summary.csv`**: Key metrics ranked by performance

## Metrics Computed

### 1. Feed Rate Accuracy
- **Mean/Median/Std Absolute Error**: Difference between predicted and ground truth peck rates
- **Correlation**: Pearson correlation between predicted and actual feed rates
- **Units**: Pecks per minute

### 2. Social Interaction Timing (PropTime)
- **Proportion Time Error**: Accuracy of co-occurrence timing predictions
- **Correlation**: How well the model captures social interaction patterns
- **Calculation**: Overlap frames / total video frames

### 3. Behavior Classification
- **Precision/Recall/F1**: Performance on peck behavior detection
- **Time Window**: 1-second windows (25 frames at 25 fps)

### 4. Frame Accuracy
- **Percentage Correct**: Frames with correct bird identification and location
- **BBox overlap Threshold**: 0.5 of ground truth bbox to be a match

## Performance Rankings

The script outputs two separate rankings:

### 🥇 Feed Rate Performers
Ranked by low absolute error + high correlation

### 🥈 PropTime Performers  
Ranked by low proportion time error + high correlation

### 📊 Human Benchmark
Shown separately for comparison (not included in algorithm rankings)

## Requirements

### Python Dependencies
```bash
pip install pandas numpy scikit-learn
```

### Data Requirements
- Minimum 4 validation videos with complete annotations
- At least one algorithm combination with all required files
- Consistent naming conventions across all files

## Troubleshooting

### Common Issues

**"No algorithm combinations found"**
- Check inference directory path
- Verify file naming conventions match exactly
- Ensure all three file types exist for each combination

**"Metadata file not found"**
- Verify `--dataset_path` points to dataset root
- Check that `MetaData.csv` exists in `ApplicationSpecific/` subdirectory

**"Ground truth directory not found"**
- Ensure `GroundTruth/` exists in `ApplicationSpecific/` subdirectory
- Check that ground truth files exist for validation videos

**"Warning: Could not compute human benchmark data"**
- Human benchmark is optional - script will continue without it
- Check if `HumanBenchmarkBORIS/` directory exists and contains `.tsv` files

### Performance Tips
- Use SSD storage for large datasets
- Ensure sufficient RAM (8GB+ recommended)
- The script uses caching - subsequent runs are faster

## Algorithm Discovery

The script automatically discovers algorithm combinations by scanning filenames:
- **Models**: Extracted from `BehavDict` files
- **ID Algorithms**: Extracted from `IDMatch` files  
- **Trackers**: Extracted from `TrackedBBox` files

All valid combinations of these three components are processed.

## Output Interpretation

### Summary CSV Columns
- **Combination**: Algorithm combination identifier
- **PerFrame_Percentage_Correct**: Overall tracking accuracy
- **Mean_FeedRate_AbsError**: Average error in feed rate prediction
- **FeedRate_Correlation**: Correlation with ground truth feed rates
- **Mean_PropTime_AbsError**: Average error in social interaction timing
- **PropTime_Correlation**: Correlation with ground truth interaction timing
- **Mean_Peck_Precision/Recall/F1**: Behavior classification performance

### Best Performance Indicators
- **Low Error**: Lower absolute error values are better
- **High Correlation**: Higher correlation values are better  
- **High Accuracy**: Higher percentage values are better

The human benchmark provides an upper bound for expected performance on this dataset.
