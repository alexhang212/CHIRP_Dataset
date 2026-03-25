"""
Run C3D action recognition to produce BehavDict pickle files
compatible with ComputeMetrics.py.

Output: {video}_BehavDict_C3D_{tracker}.p  per video/tracker combination

Usage:
    python RunBehav.py \
        --video_dir  /path/to/videos \
        --bbox_dir   /path/to/InferenceData \
        --output_dir /path/to/InferenceData \
        --weights_dir /media/alexchan/DataSSD/CHIRP_Dataset/Weights
"""

import argparse
import cv2
import numpy as np
import os
import pandas as pd
import pickle
import sys
import tempfile
from tqdm import tqdm


ValidationVideos = [
    '20210909_OiFHBq_1', '20210909_CFfo1Z', '20210914_h0h9CW', '20210922_GWDswk',
    '20211018_WS3GWi', '20220816_RB15Aq', '20220829_BrmjDE', '20220901_WS3GWi',
    '20220922_ebqkf4', '20221004_ukRCut', '20221009_WS3GWi', '20221015_csG6Ba_2'
]


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

def discover_trackers(bbox_dir):
    """Scan bbox_dir for TrackedBBox CSVs and return the set of tracker names found."""
    trackers = set()
    for fname in os.listdir(bbox_dir):
        if '_TrackedBBox_' in fname and fname.endswith('.csv'):
            tracker = fname.split('_TrackedBBox_')[1].replace('.csv', '')
            trackers.add(tracker)
    return sorted(trackers)


def process_crop(crop, target_size):
    """Pad crop to square then resize to target_size (w, h)."""
    h, w = crop.shape[:2]
    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    padded[:h, :w] = crop
    resized = cv2.resize(padded, target_size)
    return resized


def get_cropped_video(video_path, track_bbox_df, crop_size=(480, 480)):
    """
    Extract a cropped video clip for a single track.

    Args:
        video_path:    Path to source .mp4 video.
        track_bbox_df: DataFrame rows for this track (columns: Frame, Xmin, Ymin, Xmax, Ymax).
        crop_size:     Output (width, height) in pixels.
        scale_bbox:    Bounding box scale factor (unused scaling kept for parity).

    Returns:
        Path to a temporary .mp4 file with the cropped clip.
    """
    start = int(track_bbox_df['Frame'].min())
    end   = int(track_bbox_df['Frame'].max())

    tmp_path = tempfile.mktemp(suffix='.mp4')
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(
        filename=tmp_path,
        apiPreference=cv2.CAP_FFMPEG,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=25,
        frameSize=crop_size,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for i in range(end - start + 1):
        frame_num = start + i
        row = track_bbox_df[track_bbox_df['Frame'] == frame_num]

        ret, frame = cap.read()
        if not ret:
            break

        if len(row) > 0:
            xmin = int(row['Xmin'].values[0])
            ymin = int(row['Ymin'].values[0])
            xmax = int(row['Xmax'].values[0])
            ymax = int(row['Ymax'].values[0])
            # Clamp to frame bounds
            xmin = max(0, xmin); ymin = max(0, ymin)
            xmax = min(frame.shape[1], xmax); ymax = min(frame.shape[0], ymax)
            crop = frame[ymin:ymax, xmin:xmax]
            if crop.size > 0:
                crop = process_crop(crop, crop_size)
            else:
                crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)
        else:
            crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)

        out.write(crop)

    out.release()
    cap.release()
    return tmp_path


# ---------------------------------------------------------------------------
# Behaviour inference
# ---------------------------------------------------------------------------

def behaviour_inference(video_path, model, bbox_df, time_window=25):
    """
    Classify bird behaviour for each track in time_window-frame chunks.

    Args:
        video_path:  Path to source .mp4 video.
        model:       Initialised mmaction2 recogniser.
        bbox_df:     DataFrame with columns Frame, ID, Xmin, Ymin, Xmax, Ymax.
        time_window: Number of frames per classification window (default 25 = 1 second at 25 fps).

    Returns:
        {track_id: {(start_frame, end_frame): label}}
        where label is one of "Peck", "Submissive", "Other".
    """
    from mmaction.apis import inference_recognizer

    class_list = ['Peck', 'Submissive', 'Other']

    all_tracks    = bbox_df['ID'].unique()
    all_track_dict = {}

    for track in tqdm(all_tracks, desc='Classifying behaviours'):
        track_df     = bbox_df[bbox_df['ID'] == track].reset_index(drop=True)
        predict_dict = {}

        for x in range(0, len(track_df), time_window):
            window_df = track_df.iloc[x:x + time_window]
            if len(window_df) == 0:
                continue

            tmp_video = get_cropped_video(video_path, window_df)

            result    = inference_recognizer(model, tmp_video)
            pred_scores = list(result.pred_score.cpu().numpy())
            predicted   = class_list[pred_scores.index(max(pred_scores))]

            start_frame = int(window_df['Frame'].min())
            end_frame   = int(window_df['Frame'].max())
            predict_dict[(start_frame, end_frame)] = predicted

            # Clean up temp file
            try:
                os.remove(tmp_video)
            except OSError:
                pass

        all_track_dict[track] = predict_dict

    return all_track_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    from mmaction.apis import init_recognizer

    trackers = discover_trackers(args.bbox_dir)
    if not trackers:
        print(f'ERROR: no TrackedBBox CSVs found in {args.bbox_dir}')
        sys.exit(1)
    print(f'Found trackers: {trackers}')

    # Auto-discover weights
    config_path  = os.path.join(args.weights_dir, 'ActionRecognition', 'c3d_CHIRP-rgb.py')
    weights_path = os.path.join(args.weights_dir, 'ActionRecognition', 'c3d_weights.pth')

    for path in [config_path, weights_path]:
        if not os.path.exists(path):
            print(f'ERROR: missing weight file: {path}')
            sys.exit(1)

    print(f'Loading C3D model from {weights_path}...')
    model = init_recognizer(config_path, weights_path, device=args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    for vid in ValidationVideos:
        video_path = os.path.join(args.video_dir, vid + '.mp4')
        if not os.path.exists(video_path):
            print(f'[SKIP] video not found: {video_path}')
            continue

        for tracker in trackers:
            out_path = os.path.join(args.output_dir, f'{vid}_BehavDict_C3D_{tracker}.p')
            if os.path.exists(out_path):
                print(f'[SKIP] already exists: {out_path}')
                continue

            bbox_path = os.path.join(args.bbox_dir, f'{vid}_TrackedBBox_{tracker}.csv')
            if not os.path.exists(bbox_path):
                print(f'[SKIP] TrackedBBox not found: {bbox_path}')
                continue

            print(f'\n=== {vid} | {tracker} ===')
            bbox_df    = pd.read_csv(bbox_path)
            track_dict = behaviour_inference(video_path, model, bbox_df)

            with open(out_path, 'wb') as f:
                pickle.dump(track_dict, f)
            print(f'  Saved BehavDict: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run C3D behaviour inference for CHIRP benchmark')
    parser.add_argument('--video_dir',   required=True, help='Directory with {video}.mp4 files')
    parser.add_argument('--bbox_dir',    required=True, help='Directory with TrackedBBox CSVs')
    parser.add_argument('--output_dir',  required=True, help='Directory to save BehavDict pickles')
    parser.add_argument('--weights_dir', required=True, help='Root weights directory')
    parser.add_argument('--device',      default='cuda:0',
                        help='CUDA device string (default: cuda:0)')
    args = parser.parse_args()
    main(args)
