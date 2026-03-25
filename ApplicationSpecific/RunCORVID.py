"""
CORVID: Colour Ring Video Identification
Run ring segmentation + colour matching to produce IDMatch pickle files
compatible with ComputeMetrics.py.

Output: {video}_CORVID_IDMatch_{tracker}.p  per video/tracker combination

Usage:
    python RunCORVID.py \
        --video_dir  /path/to/videos \
        --bbox_dir   /path/to/InferenceData \
        --output_dir /path/to/InferenceData \
        --weights_dir /media/alexchan/DataSSD/CHIRP_Dataset/Weights \
        --metadata   /path/to/CHIRP_Dataset/ApplicationSpecific/MetaData.csv
"""

import argparse
import copy
import cv2
import itertools
import numpy as np
import os
import pandas as pd
import pickle
import sys
from ast import literal_eval
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


ValidationVideos = [
    '20210909_OiFHBq_1', '20210909_CFfo1Z', '20210914_h0h9CW', '20210922_GWDswk',
    '20211018_WS3GWi', '20220816_RB15Aq', '20220829_BrmjDE', '20220901_WS3GWi',
    '20220922_ebqkf4', '20221004_ukRCut', '20221009_WS3GWi', '20221015_csG6Ba_2'
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

@contextmanager
def video_capture_manager(video_paths):
    caps = [cv2.VideoCapture(p) for p in video_paths]
    try:
        yield caps
    finally:
        for cap in caps:
            if cap is not None:
                cap.release()


def get_video_info(cap):
    return {
        'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps':    cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def set_video_position(caps, start_frame, frame_diffs):
    for i, cap in enumerate(caps):
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_diffs[i])


def convert_bbox_df(df):
    """Convert TrackedBBox DataFrame to nested dict {frame: {cam: {id: [xmin,ymin,xmax,ymax]}}}."""
    result = {}
    cam_col = 'Cam' if 'Cam' in df.columns else None
    for _, row in df.iterrows():
        frame = int(row['Frame'])
        cam   = str(row[cam_col]) if cam_col else 'cam'
        id_   = str(row['ID'])
        box   = [float(row['Xmin']), float(row['Ymin']),
                 float(row['Xmax']), float(row['Ymax'])]
        result.setdefault(frame, {}).setdefault(cam, {})[id_] = box
    return result


def discover_trackers(bbox_dir):
    """Scan bbox_dir for TrackedBBox CSVs and return the set of tracker names found."""
    trackers = set()
    for fname in os.listdir(bbox_dir):
        if '_TrackedBBox_' in fname and fname.endswith('.csv'):
            # {video}_TrackedBBox_{tracker}.csv
            tracker = fname.split('_TrackedBBox_')[1].replace('.csv', '')
            trackers.add(tracker)
    return sorted(trackers)


def get_euc_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


# ---------------------------------------------------------------------------
# Ring segmentation (Mask2Former via mmdet)
# ---------------------------------------------------------------------------

def mmdet_mask_inference(infer_frame, box, model, mask_thresh):
    """Run Mask2Former on a single bird crop. Returns {ring_label: contour_array}."""
    from mmdet.apis import inference_detector

    box = [0 if val < 0 else val for val in box]
    crop = infer_frame[round(box[1]):round(box[3]), round(box[0]):round(box[2])]

    result = inference_detector(model, crop).to_dict()
    masks   = result['pred_instances']['masks'].cpu().numpy()
    confs   = result['pred_instances']['scores'].cpu().numpy()
    classes = result['pred_instances']['labels'].cpu().numpy()

    class_ids = list(model.cfg.metainfo['classes'])

    pred_masks   = [masks[i]   for i in range(len(confs)) if confs[i] > mask_thresh]
    pred_classes = [classes[i] for i in range(len(confs)) if confs[i] > mask_thresh]

    if len(pred_masks) == 0:
        return {}

    contours = [
        cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for m in pred_masks
    ]
    contours = [c[0].reshape(-1, 2) for c in contours]
    contours = [c + np.array([box[0], box[1]]) for c in contours]  # back to image coords

    out = {}
    class_counter = {cls: 0 for cls in set(class_ids)}
    for x in range(len(pred_classes)):
        label = class_ids[pred_classes[x]]
        key   = f"{label}-{class_counter[label]}"
        out[key] = contours[x]
        class_counter[label] += 1

    return out


def run_mask_seg(video_path, weights, bbox_dict, mask_thresh=0.5):
    """
    Run ring segmentation on all frames where birds are tracked.

    Args:
        video_path:  Path to .mp4 video file.
        weights:     (config_path, weights_path) tuple.
        bbox_dict:   Nested dict {frame: {cam: {id: [xmin,ymin,xmax,ymax]}}}.
        mask_thresh: Confidence threshold for mask predictions.

    Returns:
        {frame: {cam: {id: {ring_label: contour_array}}}}
    """
    from mmdet.apis import init_detector
    from mmengine import Config

    cam_name = 'cam'

    with video_capture_manager([video_path]) as caps:
        video_info = get_video_info(caps[0])
        total_frames = video_info['frame_count']

    cfg   = Config.fromfile(weights[0])
    model = init_detector(cfg, weights[1], device='cuda:0')

    out_seg_dict = {}

    cap = cv2.VideoCapture(video_path)
    for i in tqdm(range(total_frames), desc='Running ring segmentation'):
        ret, frame = cap.read()
        if not ret:
            break

        if i not in bbox_dict or cam_name not in bbox_dict[i]:
            continue

        cam_point_dict = {}
        for bird_id, bird_bbox in bbox_dict[i][cam_name].items():
            bird_bbox = [0 if val < 0 else val for val in bird_bbox]
            seg = mmdet_mask_inference(frame, bird_bbox, model, mask_thresh)
            cam_point_dict[bird_id] = seg

        out_seg_dict[i] = {cam_name: cam_point_dict}

    cap.release()
    return out_seg_dict


# ---------------------------------------------------------------------------
# Random Forest colour classification
# ---------------------------------------------------------------------------

def run_rf(rings, rf_model, img, train_image_features):
    """
    Classify ring colours from segmentation masks.

    Args:
        rings:               {ring_label: contour_array}
        rf_model:            Trained sklearn RF model.
        img:                 Full BGR video frame.
        train_image_features: Training HSV features for StandardScaler normalisation.

    Returns:
        (predict_dict, image_dict)
        predict_dict: {ring_label: prediction}
        image_dict:   {ring_label: 20x20 warped crop}
    """
    image_out_dict = {}

    for key, ring_seg in rings.items():
        ring_seg_list = ring_seg.tolist()
        if len(ring_seg_list) == 0:
            continue

        ring_seg_points = [[round(ring_seg_list[i][0]), round(ring_seg_list[i][1])]
                           for i in range(len(ring_seg_list))]

        polygon_points = np.array(ring_seg_points)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)
        cropped = cv2.bitwise_and(img, img, mask=mask)

        x, y, w, h = cv2.boundingRect(polygon_points)
        x1, y1, x2, y2 = x, y, x + w, y + h
        out_dim = (20, 20)

        src_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        dst_points = np.array([[0, 0], [out_dim[0]-1, 0],
                               [out_dim[0]-1, out_dim[1]-1], [0, out_dim[1]-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(cropped, M, out_dim)
        image_out_dict[key] = warped

    ring_predict_dict = {}
    for key, ring_seg_img in image_out_dict.items():
        transformed = cv2.cvtColor(ring_seg_img, cv2.COLOR_BGR2HSV)
        r_histo = cv2.calcHist([transformed], [0], None, [10], [0, 256])
        g_histo = cv2.calcHist([transformed], [1], None, [10], [0, 256])
        b_histo = cv2.calcHist([transformed], [2], None, [10], [0, 256])

        feature_list = np.concatenate([r_histo.flatten(), g_histo.flatten(), b_histo.flatten()])

        dummy_train = copy.deepcopy(train_image_features)
        combined    = np.concatenate([dummy_train, feature_list.reshape(-1, 30)])

        scaler       = StandardScaler()
        scaled_array = scaler.fit_transform(combined)
        test_features = scaled_array[len(dummy_train):]

        pred_out = rf_model.predict(test_features)
        ring_predict_dict[key] = pred_out[0]

    return ring_predict_dict, image_out_dict


# ---------------------------------------------------------------------------
# CORVID: ring-pair matching and conflict resolution
# ---------------------------------------------------------------------------

def run_corvid(video_path, seg_dict, rf_model, possible_ids, train_image_features):
    """
    Match per-track ring colour predictions to individual bird IDs.

    Args:
        video_path:           Path to .mp4 video file.
        seg_dict:             Output of run_mask_seg.
        rf_model:             Trained RF colour classifier.
        possible_ids:         List of valid bird ID strings (e.g. ['ABLM', ...]).
        train_image_features: Training features for RF normalisation.

    Returns:
        {track_id: bird_id_or_"unringed"}
    """
    all_classes       = ['A', 'B', 'C', 'G', 'L', 'M', 'O', 'P', 'R', 'S', 'W', 'Y', 'U']
    class_pair_names  = np.array([f"{a}{b}" for a, b in itertools.product(all_classes, repeat=2)])

    cam = 'cam'

    # Collect frames per track
    track_frames = {}
    for frame_idx in seg_dict:
        for cam_name in seg_dict[frame_idx]:
            for bird_id in seg_dict[frame_idx][cam_name]:
                track_frames.setdefault(bird_id, []).append(frame_idx)

    id_scores_dict = {}

    for track, frames in tqdm(track_frames.items(), desc='Running CORVID matching'):
        tracklet_cum_list = []

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

        for i in range(min(frames), max(frames)):
            ret, img = cap.read()
            if not ret:
                break
            if i not in frames:
                continue
            if i not in seg_dict or cam not in seg_dict[i] or track not in seg_dict[i][cam]:
                continue

            rings = seg_dict[i][cam][track]
            colour_predict_dict, _ = run_rf(rings, rf_model, img, train_image_features)

            if len(rings) == 0:
                continue
            elif len(rings) == 1:
                pred_arr   = np.append(list(colour_predict_dict.values())[0], 0)
                ring2_pred = np.zeros((len(all_classes),))
                sum_matrix = np.add.outer(pred_arr, ring2_pred) / 2
                folded     = sum_matrix + sum_matrix.T - np.diag(sum_matrix.diagonal())
                tracklet_cum_list.append(folded.flatten().tolist())
            else:
                ring_mid_points = {ring: np.median(seg, axis=0)
                                   for ring, seg in rings.items()}
                for comb in itertools.combinations(rings.keys(), 2):
                    euc_dist = get_euc_dist(ring_mid_points[comb[0]], ring_mid_points[comb[1]])
                    if euc_dist < 38:
                        ring1_arr  = np.append(colour_predict_dict[comb[0]], 0)
                        ring2_arr  = np.append(colour_predict_dict[comb[1]], 0)
                        sum_matrix = np.add.outer(ring1_arr, ring2_arr) / 2
                        folded     = sum_matrix + sum_matrix.T - np.diag(sum_matrix.diagonal())
                        tracklet_cum_list.append(folded.flatten().tolist())

        cap.release()

        if len(tracklet_cum_list) == 0:
            id_scores_dict[track] = {}
            continue

        tracklet_arr  = np.array(tracklet_cum_list)
        ring_pair_sum = np.sum(tracklet_arr, axis=0) / len(frames)

        sorted_idx        = np.argsort(ring_pair_sum)[::-1]
        sorted_pair_names = class_pair_names[sorted_idx]
        sorted_pred_scores = ring_pair_sum[sorted_idx]

        per_bird_scores = {bird: [] for bird in possible_ids}
        for bird in possible_ids:
            if len(bird) != 4:
                continue
            ring1 = bird[0:2].upper()
            ring2 = bird[2:4].upper()
            per_bird_scores[bird].append(sorted_pred_scores[sorted_pair_names.tolist().index(ring1)])
            per_bird_scores[bird].append(sorted_pred_scores[sorted_pair_names.tolist().index(ring2)])

        id_scores_dict[track] = {bird: np.mean(scores)
                                 for bird, scores in per_bird_scores.items()
                                 if scores}

    # Initial best-score assignment
    out_convert_dict = {}
    for track in id_scores_dict:
        if len(id_scores_dict[track]) == 0:
            out_convert_dict[track] = 'unringed'
        else:
            out_convert_dict[track] = max(id_scores_dict[track], key=id_scores_dict[track].get)

    # Conflict resolution: tracks visible at the same time should have different IDs
    frame_overlap = {}
    for t1, f1 in track_frames.items():
        for t2, f2 in track_frames.items():
            if t1 == t2:
                continue
            frame_overlap[(t1, t2)] = len(set(f1).intersection(f2))

    filter_frame = 30
    filtered_overlap = {k: v for k, v in frame_overlap.items() if v > filter_frame}
    unique_pairs = [list(x) for x in set(tuple(sorted(p)) for p in filtered_overlap)]

    conflicts_per_track = {t: set() for t in out_convert_dict}
    for t1, t2 in unique_pairs:
        if t1 in conflicts_per_track and t2 in conflicts_per_track:
            conflicts_per_track[t1].add(t2)
            conflicts_per_track[t2].add(t1)

    # Sort tracks by top score descending
    track_scores = []
    for track in out_convert_dict:
        if track not in id_scores_dict or not id_scores_dict[track]:
            continue
        track_scores.append((track, max(id_scores_dict[track].values())))
    track_scores.sort(key=lambda x: x[1], reverse=True)

    tracks_to_reassign = set()
    for t1, t2 in unique_pairs:
        if t1 in out_convert_dict and t2 in out_convert_dict:
            if out_convert_dict[t1] == out_convert_dict[t2]:
                tracks_to_reassign.add(t1)
                tracks_to_reassign.add(t2)

    for track, _ in track_scores:
        if track not in tracks_to_reassign:
            continue
        sorted_ids = sorted(id_scores_dict[track].items(), key=lambda x: x[1], reverse=True)
        assigned = False
        for bird_id, _ in sorted_ids:
            conflict = any(
                out_convert_dict.get(ot) == bird_id
                for ot in conflicts_per_track[track]
            )
            if not conflict:
                out_convert_dict[track] = bird_id
                assigned = True
                break
        if not assigned and sorted_ids:
            out_convert_dict[track] = sorted_ids[0][0]

    return out_convert_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    trackers = discover_trackers(args.bbox_dir)
    if not trackers:
        print(f'ERROR: no TrackedBBox CSVs found in {args.bbox_dir}')
        sys.exit(1)
    print(f'Found trackers: {trackers}')

    # Auto-discover weights
    seg_config  = os.path.join(args.weights_dir, 'Segmentation', 'RingSegMask2Former.py')
    seg_weights = os.path.join(args.weights_dir, 'Segmentation', 'RingSegMask2Former.pth')
    rf_model_path    = os.path.join(args.weights_dir, 'CORVID', 'RandomForestModel.p')
    rf_features_path = os.path.join(args.weights_dir, 'CORVID', 'TrainImagesFeatures.p')

    for path in [seg_config, seg_weights, rf_model_path, rf_features_path]:
        if not os.path.exists(path):
            print(f'ERROR: missing weight file: {path}')
            sys.exit(1)

    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)
    with open(rf_features_path, 'rb') as f:
        train_features = pickle.load(f)

    meta_df = pd.read_csv(args.metadata)
    meta_df['PossibleBirds'] = meta_df['PossibleBirds'].apply(literal_eval)

    os.makedirs(args.output_dir, exist_ok=True)

    for vid in ValidationVideos:
        video_path = os.path.join(args.video_dir, vid + '.mp4')
        if not os.path.exists(video_path):
            print(f'[SKIP] video not found: {video_path}')
            continue

        row = meta_df[meta_df['Video'] == vid]
        if len(row) == 0:
            print(f'[SKIP] {vid} not in metadata')
            continue
        possible_ids = row['PossibleBirds'].values[0]

        for tracker in trackers:
            id_match_path = os.path.join(args.output_dir, f'{vid}_CORVID_IDMatch_{tracker}.p')
            if os.path.exists(id_match_path):
                print(f'[SKIP] already exists: {id_match_path}')
                continue

            bbox_path = os.path.join(args.bbox_dir, f'{vid}_TrackedBBox_{tracker}.csv')
            if not os.path.exists(bbox_path):
                print(f'[SKIP] TrackedBBox not found: {bbox_path}')
                continue

            print(f'\n=== {vid} | {tracker} ===')
            tracked_bbox = pd.read_csv(bbox_path)
            bbox_dict    = convert_bbox_df(tracked_bbox)

            # Ring segmentation (cached)
            ringseg_path = os.path.join(args.output_dir, f'{vid}_RingSeg_{tracker}.p')
            if os.path.exists(ringseg_path):
                print(f'  Loading cached RingSeg: {ringseg_path}')
                with open(ringseg_path, 'rb') as f:
                    seg_dict = pickle.load(f)
            else:
                print('  Running ring segmentation...')
                seg_dict = run_mask_seg(video_path, (seg_config, seg_weights), bbox_dict)
                with open(ringseg_path, 'wb') as f:
                    pickle.dump(seg_dict, f)
                print(f'  Saved RingSeg: {ringseg_path}')

            # CORVID matching
            print('  Running CORVID matching...')
            id_match = run_corvid(video_path, seg_dict, rf_model, possible_ids, train_features)

            with open(id_match_path, 'wb') as f:
                pickle.dump(id_match, f)
            print(f'  Saved IDMatch: {id_match_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CORVID ring ID matching for CHIRP benchmark')
    parser.add_argument('--video_dir',   required=True, help='Directory with {video}.mp4 files')
    parser.add_argument('--bbox_dir',    required=True, help='Directory with TrackedBBox CSVs')
    parser.add_argument('--output_dir',  required=True, help='Directory to save IDMatch pickles')
    parser.add_argument('--weights_dir', required=True, help='Root weights directory')
    parser.add_argument('--metadata',    required=True, help='Path to MetaData.csv')
    args = parser.parse_args()
    main(args)
