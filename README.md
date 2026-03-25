# CHIRP dataset: Combining beHaviour, Individual Re-identification and Postures

<!-- banner -->

**Note:** THIS REPOSITORY IS STILL UNDER MAINTENANCE, COMING SOON



Alex Hoi Hang Chan, Neha Singhal, Onur Kocahan, Andrea Meltzer, Saverio Lubrano, Miya Warrington, Michael Griesser*, Fumihiro Kano*, Hemal Naik*


## Dataset Download
Here is the download [link]() for the dataset, refer to readme in the dataset for dataset details

## Getting Started

### Application-specific metrics computation
For details of how to do the benchmarking procedure and data requirements, we refer to **[ApplicationSpecific/README.md](ApplicationSpecific/README.md)** 
- The script reads in all inference files from a directory, and computes metrics for all of them.


### Visualization Tools
The dataset also includes visualization tools to help analyze results:

#### VisualizeImages.py  
Interactive visualization tool for examining ground truth annotations:
- **COCO format support**: Visualize annotations from JSON files with bounding boxes, keypoints, and segmentation masks
- **CSV format support**: Display annotations from CSV files including bounding boxes, keypoints, and ring annotations  
- **Color-coded display**: Different annotation types shown with distinct colors and markers
- **Interactive navigation**: Step through images with keyboard controls to examine annotation quality

**Usage:**
```bash
python tools/VisualizeImages.py --annot [ANNOTATION_FILE] --image [IMAGE_DIR]
```
