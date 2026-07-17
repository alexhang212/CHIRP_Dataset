"""
CORVID Inference Pipeline
=========================
Runs the full colour ring identification pipeline on new video footage:
  1. YOLO bird detection  → bounding boxes per frame
  2. MMDetection ring segmentation → ring contours per bird crop
  3. Random Forest colour classification → colour probability vector per ring
  4. CORVID matching algorithm → ring-pair aggregation + individual ID assignment

Expected weights directory layout (matches the provided CHIRP pretrained
weights release, e.g. .../CHIRP_Dataset/Weights/):
    weights_dir/
    ├── YOLO/
    │   └── *.pt              (bird detector, any filename — auto-detected)
    ├── Segmentation/
    │   ├── *.py              (mmdet config, any filename — auto-detected)
    │   └── *.pth             (mmdet weights, any filename — auto-detected)
    └── CORVID/
        ├── RandomForestModel.p
        ├── TrainImagesFeatures.p
        └── Classes.p         (optional — falls back to the 12 CHIRP colour
                                 classes A,B,C,G,L,M,O,P,R,S,W,Y if absent)

Each of YOLO/ and Segmentation/ must contain exactly one file of the
relevant extension; this script does not require renaming files from
training output (e.g. Bird_YOLOv8.pt, RingSegMask2Former.pth/.py work as-is).

Outputs, per video:
    {output_dir}/{video}_BBoxes.csv           frame,track_id,x1,y1,x2,y2
        every tracked bird bounding box, from YOLO detection + IoU tracking
    {output_dir}/{video}_Segmentations.csv    frame,track_id,ring_id,contour
        every detected ring polygon per tracked bird box
    {output_dir}/{video}_CORVID_IDMatch.p     {track_id: bird_id_or_"unringed"}
        pickle, compatible with ComputeMetrics.py when using the CORVID algorithm
    {output_dir}/{video}_CORVID_IDMatch.csv   track_id,bird_id
        same mapping as the pickle above, as a plain CSV
    {output_dir}/{video}_BBoxes_WithID.csv    frame,track_id,x1,y1,x2,y2,bird_id
        every tracked bounding box joined with its final bird ID — the one
        file most users need for downstream analysis/visualisation

Usage:
    python 4_run_inference.py \
        --video_dir    /path/to/videos \
        --output_dir   /path/to/outputs \
        --weights_dir  /path/to/weights \
        --possible_ids /path/to/possible_birds.csv

    # possible_birds.csv format (per-video):
    #   Video,PossibleBirds
    #   20210909_OiFHBq_1,"['ABLM', 'RRYY', 'WGYB']"
    #   (if a video isn't found in the Video column, every bird ID listed
    #   anywhere in the file is used as the candidate set for it instead)
    #
    # OR a plain .txt file with one bird ID per line (applied to all videos):
    #   ABLM
    #   RRYY
    #   WGYB
"""

import argparse
import copy
import csv
import cv2
import glob
import itertools
import os
import pickle
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

DEFAULT_CHIRP_COLOURS = ['A', 'B', 'C', 'G', 'L', 'M', 'O', 'P', 'R', 'S', 'W', 'Y']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_one(directory, extension, label):
    """Find exactly one file with the given extension in `directory`."""
    matches = sorted(glob.glob(os.path.join(directory, f'*{extension}')))
    if not matches:
        print(f'ERROR: no {extension} file found for {label} in {directory}')
        sys.exit(1)
    if len(matches) > 1:
        print(f'ERROR: multiple {extension} files found for {label} in {directory}: {matches}. '
              f'Keep only one.')
        sys.exit(1)
    return matches[0]


def load_models(weights_dir):
    """Load YOLO, MMDet, and RF models from the expected weights directory layout."""
    from ultralytics import YOLO as UltralyticsYOLO
    from mmdet.apis import init_detector
    from mmengine.config import Config

    yolo_dir = os.path.join(weights_dir, 'YOLO')
    seg_dir  = os.path.join(weights_dir, 'Segmentation')
    corvid_dir = os.path.join(weights_dir, 'CORVID')

    yolo_path   = find_one(yolo_dir, '.pt', 'YOLO bird detector')
    seg_cfg     = find_one(seg_dir, '.py', 'mmdet config')
    seg_weights = find_one(seg_dir, '.pth', 'mmdet weights')

    rf_path       = os.path.join(corvid_dir, 'RandomForestModel.p')
    features_path = os.path.join(corvid_dir, 'TrainImagesFeatures.p')
    classes_path  = os.path.join(corvid_dir, 'Classes.p')

    for p in [rf_path, features_path]:
        if not os.path.exists(p):
            print(f'ERROR: missing file: {p}')
            sys.exit(1)

    yolo_model = UltralyticsYOLO(yolo_path)
    seg_model  = init_detector(Config.fromfile(seg_cfg), seg_weights, device='cuda:0')

    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    with open(features_path, 'rb') as f:
        train_features = pickle.load(f)

    if os.path.exists(classes_path):
        with open(classes_path, 'rb') as f:
            classes = pickle.load(f)
    else:
        classes = DEFAULT_CHIRP_COLOURS
        print(f'No Classes.p in {corvid_dir}, assuming the standard 12 CHIRP colour classes: {classes}')

    return yolo_model, seg_model, rf_model, train_features, classes


# ---------------------------------------------------------------------------
# Detection + tracking
# ---------------------------------------------------------------------------

def run_yolo_detection(frame, yolo_model, conf_thresh=0.5):
    """Run YOLO on a single frame. Returns {det_id: [x1,y1,x2,y2]}."""
    results = yolo_model(frame, conf=conf_thresh, verbose=False)[0]
    boxes = results.boxes
    detections = {}
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detections[i] = [float(x1), float(y1), float(x2), float(y2)]
    return detections


def _iou(b1, b2):
    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def iou_tracker(prev_tracks, detections, iou_thresh=0.4, next_id_ref=None):
    """
    Simple IoU-based tracker. Assigns stable track IDs across frames.

    Args:
        prev_tracks:  {track_id: [x1,y1,x2,y2]} from previous frame.
        detections:   {det_idx: [x1,y1,x2,y2]} from current frame.
        iou_thresh:   Minimum IoU to match a detection to a track.
        next_id_ref:  List [int] used as mutable counter for new track IDs.

    Returns:
        current_tracks: {track_id: [x1,y1,x2,y2]}
    """
    if next_id_ref is None:
        next_id_ref = [0]

    if not prev_tracks:
        new_tracks = {}
        for box in detections.values():
            new_tracks[next_id_ref[0]] = box
            next_id_ref[0] += 1
        return new_tracks

    matched = {}
    used_dets = set()
    for tid, tbox in prev_tracks.items():
        best_iou, best_det = 0.0, None
        for did, dbox in detections.items():
            if did in used_dets:
                continue
            iou = _iou(tbox, dbox)
            if iou > best_iou:
                best_iou, best_det = iou, did
        if best_iou >= iou_thresh and best_det is not None:
            matched[tid] = detections[best_det]
            used_dets.add(best_det)

    for did, dbox in detections.items():
        if did not in used_dets:
            matched[next_id_ref[0]] = dbox
            next_id_ref[0] += 1

    return matched


# ---------------------------------------------------------------------------
# Ring segmentation (adapted from RunCORVID.py)
# ---------------------------------------------------------------------------

def mmdet_mask_inference(frame, box, model, mask_thresh):
    """Run mmdet on a single bird crop. Returns {ring_label: contour_array}."""
    from mmdet.apis import inference_detector
    box = [max(0, v) for v in box]
    crop = frame[round(box[1]):round(box[3]), round(box[0]):round(box[2])]
    if crop.size == 0:
        return {}

    result = inference_detector(model, crop).to_dict()
    masks   = result['pred_instances']['masks'].cpu().numpy()
    confs   = result['pred_instances']['scores'].cpu().numpy()
    classes = result['pred_instances']['labels'].cpu().numpy()

    class_ids = list(model.cfg.metainfo['classes'])
    pred_masks   = [masks[i]   for i in range(len(confs)) if confs[i] > mask_thresh]
    pred_classes = [classes[i] for i in range(len(confs)) if confs[i] > mask_thresh]

    if not pred_masks:
        return {}

    out = {}
    class_counter = {cls: 0 for cls in set(class_ids)}
    for idx in range(len(pred_classes)):
        contours = cv2.findContours(
            pred_masks[idx].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            continue
        c = contours[0].reshape(-1, 2)
        c = c + np.array([box[0], box[1]])
        label = class_ids[pred_classes[idx]]
        key = f'{label}-{class_counter[label]}'
        out[key] = c
        class_counter[label] += 1

    return out


# ---------------------------------------------------------------------------
# RF colour classification (adapted from RunCORVID.py)
# ---------------------------------------------------------------------------

def run_rf(rings, rf_model, img, train_image_features):
    """
    Classify ring colours from segmentation contours.

    Returns:
        predict_dict: {ring_label: (num_classes,) probability array}
        image_dict:   {ring_label: 20x20 warped crop}
    """
    image_dict = {}
    for key, contour in rings.items():
        pts = contour.tolist()
        if not pts:
            continue
        pts = [[round(p[0]), round(p[1])] for p in pts]
        poly = np.array(pts)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        cropped = cv2.bitwise_and(img, img, mask=mask)

        x, y, w, h = cv2.boundingRect(poly)
        out_dim = (20, 20)
        src = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
        dst = np.array([[0, 0], [19, 0], [19, 19], [0, 19]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(cropped, M, out_dim)
        image_dict[key] = warped

    predict_dict = {}
    for key, crop in image_dict.items():
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [10], [0, 256]).flatten()
        s = cv2.calcHist([hsv], [1], None, [10], [0, 256]).flatten()
        v = cv2.calcHist([hsv], [2], None, [10], [0, 256]).flatten()
        feat = np.concatenate([h, s, v]).reshape(1, -1)

        combined = np.concatenate([copy.deepcopy(train_image_features), feat])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined)
        test_feat = scaled[len(train_image_features):]

        predict_dict[key] = rf_model.predict(test_feat)[0]

    return predict_dict, image_dict


# ---------------------------------------------------------------------------
# CORVID matching algorithm (adapted from RunCORVID.py)
# ---------------------------------------------------------------------------

def run_corvid(video_path, seg_dict, rf_model, possible_ids, train_image_features, classes):
    """
    Aggregate per-frame ring predictions into ring-pair matrices and assign
    bird identities via the CORVID scoring algorithm.

    Args:
        video_path:           Path to .mp4 video.
        seg_dict:             {frame: {cam: {track_id: {ring_label: contour}}}}
        rf_model:             Trained RF colour classifier.
        possible_ids:         List of 4-char bird ID strings (e.g. ['ABLM', 'RRYY']).
        train_image_features: (N_train, 30) raw feature array for StandardScaler re-fit.
        classes:              Ordered list of colour class names.

    Returns:
        {track_id: bird_id_or_"unringed"}
    """
    all_classes      = classes + ['U']
    class_pair_names = np.array([f'{a}{b}' for a, b in itertools.product(all_classes, repeat=2)])
    cam = 'cam'

    track_frames = {}
    for frame_idx in seg_dict:
        for cam_name in seg_dict[frame_idx]:
            for bird_id in seg_dict[frame_idx][cam_name]:
                track_frames.setdefault(bird_id, []).append(frame_idx)

    id_scores_dict = {}

    for track, frames in tqdm(track_frames.items(), desc='CORVID matching'):
        tracklet_cum = []

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

        for i in range(min(frames), max(frames) + 1):
            ret, img = cap.read()
            if not ret:
                break
            if i not in frames:
                continue
            if i not in seg_dict or cam not in seg_dict[i] or track not in seg_dict[i][cam]:
                continue

            rings = seg_dict[i][cam][track]
            colour_preds, _ = run_rf(rings, rf_model, img, train_image_features)

            if not rings:
                continue
            elif len(rings) == 1:
                pred_arr   = np.append(list(colour_preds.values())[0], 0)
                ring2_pred = np.zeros(len(all_classes))
                sum_mat    = np.add.outer(pred_arr, ring2_pred) / 2
                folded     = sum_mat + sum_mat.T - np.diag(sum_mat.diagonal())
                tracklet_cum.append(folded.flatten().tolist())
            else:
                mid_pts = {r: np.median(seg, axis=0) for r, seg in rings.items()}
                for r1, r2 in itertools.combinations(rings.keys(), 2):
                    dist = float(np.linalg.norm(
                        np.array(mid_pts[r1], float) - np.array(mid_pts[r2], float)))
                    if dist < 38:
                        arr1 = np.append(colour_preds[r1], 0)
                        arr2 = np.append(colour_preds[r2], 0)
                        sum_mat = np.add.outer(arr1, arr2) / 2
                        folded  = sum_mat + sum_mat.T - np.diag(sum_mat.diagonal())
                        tracklet_cum.append(folded.flatten().tolist())

        cap.release()

        if not tracklet_cum:
            id_scores_dict[track] = {}
            continue

        ring_pair_sum  = np.sum(np.array(tracklet_cum), axis=0) / len(frames)
        sorted_idx     = np.argsort(ring_pair_sum)[::-1]
        sorted_names   = class_pair_names[sorted_idx]
        sorted_scores  = ring_pair_sum[sorted_idx]
        name_list      = sorted_names.tolist()

        per_bird = {bird: [] for bird in possible_ids}
        for bird in possible_ids:
            if len(bird) != 4:
                continue
            r1 = bird[0:2].upper()
            r2 = bird[2:4].upper()
            if r1 in name_list:
                per_bird[bird].append(sorted_scores[name_list.index(r1)])
            if r2 in name_list:
                per_bird[bird].append(sorted_scores[name_list.index(r2)])

        id_scores_dict[track] = {b: float(np.mean(s)) for b, s in per_bird.items() if s}

    # Initial best-score assignment
    out = {}
    for track in id_scores_dict:
        if not id_scores_dict[track]:
            out[track] = 'unringed'
        else:
            out[track] = max(id_scores_dict[track], key=id_scores_dict[track].get)

    # Conflict resolution: co-visible tracks should have different IDs
    frame_overlap = {}
    for t1, f1 in track_frames.items():
        for t2, f2 in track_frames.items():
            if t1 != t2:
                frame_overlap[(t1, t2)] = len(set(f1) & set(f2))

    unique_pairs = list({tuple(sorted(p)) for p, v in frame_overlap.items() if v > 30})

    conflicts = {t: set() for t in out}
    for t1, t2 in unique_pairs:
        if t1 in conflicts and t2 in conflicts:
            conflicts[t1].add(t2)
            conflicts[t2].add(t1)

    track_scores = sorted(
        [(t, max(id_scores_dict[t].values())) for t in out if id_scores_dict.get(t)],
        key=lambda x: x[1], reverse=True
    )
    reassign = {t for t1, t2 in unique_pairs
                if t1 in out and t2 in out and out[t1] == out[t2]
                for t in (t1, t2)}

    for track, _ in track_scores:
        if track not in reassign:
            continue
        for bird_id, _ in sorted(id_scores_dict[track].items(), key=lambda x: x[1], reverse=True):
            if not any(out.get(ot) == bird_id for ot in conflicts[track]):
                out[track] = bird_id
                break

    return out


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------

def load_possible_ids(possible_ids_path, video_name):
    """Load possible bird IDs for a given video.
    Supports CSV (columns: Video, PossibleBirds) or plain .txt (one ID per line)."""
    if possible_ids_path.endswith('.txt'):
        with open(possible_ids_path) as f:
            return [line.strip() for line in f if line.strip()]

    df = pd.read_csv(possible_ids_path)
    if 'PossibleBirds' in df.columns:
        df['PossibleBirds'] = df['PossibleBirds'].apply(literal_eval)
        row = df[df['Video'] == video_name]
        if len(row):
            return row['PossibleBirds'].values[0]

        all_birds = sorted({bird for birds in df['PossibleBirds'] for bird in birds})
        print(f"WARNING: no possible-IDs row for video '{video_name}' in {possible_ids_path}; "
              f'falling back to all {len(all_birds)} bird IDs listed in the file')
        return all_birds

    # Flat single-column CSV (no Video/PossibleBirds columns): one bird ID per row, applied to every video.
    return df.iloc[:, 0].tolist()


def process_video(video_path, yolo_model, seg_model, rf_model, train_features, classes,
                  possible_ids, args):
    """Run full CORVID pipeline for a single video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    seg_dict = {}
    bbox_dict = {}
    prev_tracks = {}
    next_id = [0]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'  Processing {total} frames...')

    for frame_idx in tqdm(range(total), desc='Detection + segmentation'):
        ret, frame = cap.read()
        if not ret:
            break

        dets = run_yolo_detection(frame, yolo_model, args.yolo_thresh)
        tracks = iou_tracker(prev_tracks, dets, args.iou_thresh, next_id)
        prev_tracks = tracks

        if not tracks:
            continue

        bbox_dict[frame_idx] = tracks

        cam_seg = {}
        for tid, bbox in tracks.items():
            rings = mmdet_mask_inference(frame, bbox, seg_model, args.mask_thresh)
            if rings:
                cam_seg[str(tid)] = rings

        if cam_seg:
            seg_dict[frame_idx] = {'cam': cam_seg}

    cap.release()

    write_bbox_csv(bbox_dict, os.path.join(args.output_dir, f'{video_name}_BBoxes.csv'))
    write_segmentation_csv(seg_dict, os.path.join(args.output_dir, f'{video_name}_Segmentations.csv'))

    if not seg_dict:
        print('  No rings detected in video.')
        id_match = {}
    else:
        print(f'  Running CORVID matching...')
        id_match = run_corvid(video_path, seg_dict, rf_model, possible_ids, train_features, classes)

    write_bbox_with_id_csv(bbox_dict, id_match, os.path.join(args.output_dir, f'{video_name}_BBoxes_WithID.csv'))

    return id_match


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_id_match_csv(id_match, csv_path):
    """Write a {track_id: bird_id} mapping to a two-column CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'bird_id'])
        for track_id in sorted(id_match, key=int):
            writer.writerow([track_id, id_match[track_id]])


def write_bbox_csv(bbox_dict, csv_path):
    """Write every tracked bounding box to CSV: frame,track_id,x1,y1,x2,y2.

    Args:
        bbox_dict: {frame_idx: {track_id: [x1,y1,x2,y2]}}
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
        for frame_idx in sorted(bbox_dict):
            for track_id, box in sorted(bbox_dict[frame_idx].items()):
                writer.writerow([frame_idx, track_id, *box])


def write_segmentation_csv(seg_dict, csv_path):
    """Write every detected ring contour to CSV: frame,track_id,ring_id,contour.

    Args:
        seg_dict: {frame_idx: {'cam': {track_id: {ring_id: contour_array}}}}
        `contour` is written as a "[[x,y],[x,y],...]" string, readable back
        with ast.literal_eval.
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'track_id', 'ring_id', 'contour'])
        for frame_idx in sorted(seg_dict):
            cam_seg = seg_dict[frame_idx]['cam']
            for track_id, rings in sorted(cam_seg.items(), key=lambda kv: int(kv[0])):
                for ring_id, contour in rings.items():
                    writer.writerow([frame_idx, track_id, ring_id, contour.tolist()])


def write_bbox_with_id_csv(bbox_dict, id_match, csv_path):
    """Write every tracked bounding box joined with its final bird ID assignment.

    Args:
        bbox_dict: {frame_idx: {track_id: [x1,y1,x2,y2]}}
        id_match:  {track_id (str): bird_id_or_"unringed"}
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'bird_id'])
        for frame_idx in sorted(bbox_dict):
            for track_id, box in sorted(bbox_dict[frame_idx].items()):
                bird_id = id_match.get(str(track_id), 'unringed')
                writer.writerow([frame_idx, track_id, *box, bird_id])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading models...')
    yolo_model, seg_model, rf_model, train_features, classes = load_models(args.weights_dir)

    video_files = sorted([
        f for f in os.listdir(args.video_dir) if f.endswith('.mp4')
    ])
    if not video_files:
        print(f'No .mp4 files found in {args.video_dir}')
        sys.exit(1)

    print(f'Found {len(video_files)} video(s)')

    for vf in video_files:
        video_name = os.path.splitext(vf)[0]
        out_path   = os.path.join(args.output_dir, f'{video_name}_CORVID_IDMatch.p')
        csv_path   = os.path.join(args.output_dir, f'{video_name}_CORVID_IDMatch.csv')

        if os.path.exists(out_path):
            if not os.path.exists(csv_path):
                with open(out_path, 'rb') as f:
                    write_id_match_csv(pickle.load(f), csv_path)
                print(f'[SKIP] {video_name} already processed, wrote missing CSV: {csv_path}')
            else:
                print(f'[SKIP] already exists: {out_path}')
            continue

        possible_ids = load_possible_ids(args.possible_ids, video_name)
        print(f'\n=== {video_name} | {len(possible_ids)} possible IDs ===')

        id_match = process_video(
            os.path.join(args.video_dir, vf),
            yolo_model, seg_model, rf_model, train_features, classes,
            possible_ids, args
        )

        with open(out_path, 'wb') as f:
            pickle.dump(id_match, f)
        write_id_match_csv(id_match, csv_path)
        print(f'  Saved: {out_path}')
        print(f'  Saved: {csv_path}')

        summary = {k: v for k, v in id_match.items() if v != 'unringed'}
        print(f'  Assigned {len(summary)}/{len(id_match)} tracks to known IDs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CORVID colour ring identification pipeline')
    parser.add_argument('--video_dir',    required=True,
                        help='Directory containing .mp4 video files')
    parser.add_argument('--output_dir',   required=True,
                        help='Directory to save IDMatch pickle files')
    parser.add_argument('--weights_dir',  required=True,
                        help='Root weights directory (see expected layout in docstring)')
    parser.add_argument('--possible_ids', required=True,
                        help='CSV with Video,PossibleBirds columns OR .txt with one ID per line')
    parser.add_argument('--mask_thresh',  type=float, default=0.5,
                        help='Ring segmentation confidence threshold (default: 0.5)')
    parser.add_argument('--yolo_thresh',  type=float, default=0.5,
                        help='YOLO bird detection confidence threshold (default: 0.5)')
    parser.add_argument('--iou_thresh',   type=float, default=0.4,
                        help='IoU threshold for cross-frame tracking (default: 0.4)')
    args = parser.parse_args()
    main(args)
