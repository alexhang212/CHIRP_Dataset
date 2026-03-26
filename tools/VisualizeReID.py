"""Visualize ReID tracklets as a strip of sampled frames"""

import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import sys


N_DISPLAY_FRAMES = 8   # frames to sample from each 25-frame tracklet
RING_COLOURS = [
    (255,   0, 255),   # Ring-0: magenta
    (  0, 255, 255),   # Ring-1: cyan
    (  0, 165, 255),   # Ring-2: orange
    (255, 255,   0),   # Ring-3: yellow
]
ColourDictionary = {
    'hd_bill_tip': (31, 119, 180), 'hd_bill_tiplower': (255, 127, 14),
    'hd_bill_base': (44, 160, 44), 'hd_eye_right': (214, 39, 40),
    'bd_shoulder_right': (148, 103, 189), 'bd_tail_base': (140, 86, 75),
    'bd_tail_tip': (227, 119, 194), 'bd_ankle_right': (127, 127, 127),
    'bd_feet_right': (188, 189, 34), 'hd_eye_left': (23, 190, 207),
    'bd_shoulder_left': (31, 119, 180), 'bd_ankle_left': (255, 127, 14),
    'bd_feet_left': (44, 160, 44)
}


def collect_tracklets(data_dir):
    """Walk data_dir and return list of (bird_id, video_territory, tracklet_name, path)."""
    tracklets = []
    for bird_id in sorted(os.listdir(data_dir)):
        bird_dir = os.path.join(data_dir, bird_id)
        if not os.path.isdir(bird_dir):
            continue
        for video_territory in sorted(os.listdir(bird_dir)):
            vt_dir = os.path.join(bird_dir, video_territory)
            if not os.path.isdir(vt_dir):
                continue
            for tracklet in sorted(os.listdir(vt_dir)):
                t_dir = os.path.join(vt_dir, tracklet)
                if os.path.isdir(t_dir):
                    tracklets.append((bird_id, video_territory, tracklet, t_dir))
    return tracklets


def normalize_path_key(path):
    return os.path.normpath(path).replace("\\", "/")


def load_keypoints_csv(csv_path):
    """Load video-level keypoints CSV indexed by image path."""
    df = pd.read_csv(csv_path)
    kp_data = {}
    for img_path, group in df.groupby("img"):
        key = normalize_path_key(img_path)
        kp_data[key] = list(zip(group["Keypoint"], group["x"], group["y"], group["conf"]))
    return kp_data


def iter_mask_contours(mask_obj):
    """Flatten nested mask JSON structures into contour coordinate lists."""
    if not isinstance(mask_obj, list):
        return

    if mask_obj and all(isinstance(v, (int, float)) for v in mask_obj):
        if len(mask_obj) >= 6:
            yield mask_obj
        return

    for item in mask_obj:
        yield from iter_mask_contours(item)


def load_masks_csv(csv_path):
    """Load video-level masks CSV indexed by image path."""
    df = pd.read_csv(csv_path)
    mask_data = {}
    for img_path, group in df.groupby("img"):
        contours = []
        for seg_json in group["mask"]:
            for contour in iter_mask_contours(json.loads(seg_json)):
                pts = np.array(contour, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                contours.append(pts)
        mask_data[normalize_path_key(img_path)] = contours
    return mask_data


def load_ring_masks_csv(csv_path):
    """Load ring masks CSV (columns: img, Class, mask) indexed by image path.
    Returns {img_path: [(class_label, contour_array), ...]}."""
    df = pd.read_csv(csv_path)
    ring_data = {}
    for img_path, group in df.groupby("img"):
        rings = []
        for _, row in group.iterrows():
            for contour in iter_mask_contours(json.loads(row["mask"])):
                pts = np.array(contour, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                rings.append((row["Class"], pts))
        ring_data[normalize_path_key(img_path)] = rings
    return ring_data


def draw_keypoints(img, keypoints):
    """Draw keypoints from a list of (keypoint_name, x, y, conf) tuples."""
    for kp_name, x, y, conf in keypoints:
        if conf > 0 and kp_name in ColourDictionary:
            cv2.circle(img, (int(x), int(y)), 5, ColourDictionary[kp_name], -1)
    return img


def draw_ring_masks(img, ring_contours):
    """Draw ring segmentation masks as coloured polygon outlines."""
    for class_label, pts in ring_contours:
        idx = int(class_label.split('-')[-1]) if '-' in class_label else 0
        colour = RING_COLOURS[idx % len(RING_COLOURS)]
        cv2.polylines(img, [pts], isClosed=True, color=colour, thickness=2)
    return img


def draw_masks(img, contours):
    """Draw segmentation mask contours as a semi-transparent overlay."""
    overlay = img.copy()
    for pts in contours:
        cv2.fillPoly(overlay, [pts], color=(0, 255, 255))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for pts in contours:
        cv2.polylines(img, [pts], isClosed=True, color=(0, 200, 200), thickness=1)
    return img


def load_video_overlays(video_dir, show_kp, show_mask):
    """Load optional video-level keypoint, mask, and ring mask overlays."""
    kp_data = {}
    mask_data = {}
    ring_mask_data = {}

    if show_kp:
        kp_path = os.path.join(video_dir, "keypoints.csv")
        if os.path.exists(kp_path):
            kp_data = load_keypoints_csv(kp_path)

    if show_mask:
        mask_path = os.path.join(video_dir, "masks.csv")
        if os.path.exists(mask_path):
            mask_data = load_masks_csv(mask_path)
        ring_path = os.path.join(video_dir, "masks_ring.csv")
        if os.path.exists(ring_path):
            ring_mask_data = load_ring_masks_csv(ring_path)

    return kp_data, mask_data, ring_mask_data


def load_tracklet_frames(tracklet_dir, reid_dir, kp_data=None, mask_data=None, ring_mask_data=None, n=N_DISPLAY_FRAMES):
    """Load n evenly-spaced frames from a tracklet directory, with optional overlays."""
    frame_paths = sorted([
        os.path.join(tracklet_dir, f)
        for f in os.listdir(tracklet_dir)
        if f.lower().endswith(".jpg")
    ])
    if not frame_paths:
        return []
    indices = np.linspace(0, len(frame_paths) - 1, min(n, len(frame_paths)), dtype=int)
    frames = []
    for idx in indices:
        frame_path = frame_paths[idx]
        img = cv2.imread(frame_path)
        if img is not None:
            frame_key = normalize_path_key(os.path.relpath(frame_path, reid_dir))
            if mask_data and frame_key in mask_data:
                img = draw_masks(img, mask_data[frame_key])
            if ring_mask_data and frame_key in ring_mask_data:
                img = draw_ring_masks(img, ring_mask_data[frame_key])
            if kp_data and frame_key in kp_data:
                img = draw_keypoints(img, kp_data[frame_key])
            frames.append(img)
    return frames


def make_strip(frames, target_h=240):
    """Stack frames horizontally into a single strip image."""
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(f, (int(w * scale), target_h)))
    return np.concatenate(resized, axis=1)


def visualize_tracklet(bird_id, video_territory, tracklet_name, tracklet_dir, reid_dir,
                       annot_df, kp_data, mask_data, ring_mask_data):
    frames = load_tracklet_frames(tracklet_dir, reid_dir, kp_data, mask_data, ring_mask_data)
    if not frames:
        print(f"No frames in {tracklet_dir}")
        return "next"

    strip = make_strip(frames)

    # Look up split/territory metadata from annotation CSV if provided
    territory = ""
    split_meta = ""
    if annot_df is not None:
        match = annot_df[
            (annot_df["id"] == bird_id)
            & (annot_df["Video"] == video_territory)
            & (annot_df["Tracklet"] == tracklet_name)
        ]
        if match.empty:
            match = annot_df[annot_df["id"] == bird_id]
        if not match.empty:
            row = match.iloc[0]
            territory = row.get("Territory", "")
            closed_split = row.get("ClosedSetSplit", "")
            disjointed_split = row.get("DisjointedSetSplit", "")
            open_split = row.get("OpenSetSplit", "")
            split_meta = (
                f"Closed: {closed_split}  "
                f"Disjointed: {disjointed_split}  "
                f"Open: {open_split}"
            )

    # Add info banner above strip
    banner_h = 70
    banner = np.zeros((banner_h, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, f"ID: {bird_id}  |  {video_territory}  |  {tracklet_name}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(banner, f"Territory: {territory}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(banner, split_meta, (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(banner, "n=next  q=quit", (strip.shape[1] - 200, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    display = np.concatenate([banner, strip], axis=0)
    cv2.imshow("ReID", display)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            return "quit"
        elif key == ord('n') or key == 13:  # n or Enter
            return "next"


def main(reid_dir, show_kp, show_mask):
    data_dir = os.path.join(reid_dir, "data")
    annot_path = os.path.join(reid_dir, "Annotation.csv")

    if not os.path.isdir(data_dir):
        print(f"ReID data directory not found: {data_dir}")
        sys.exit(1)

    tracklets = collect_tracklets(data_dir)
    if not tracklets:
        print(f"No tracklets found in {data_dir}")
        sys.exit(1)

    annot_df = None
    if os.path.exists(annot_path):
        annot_df = pd.read_csv(annot_path)
        print(f"Loaded annotations: {len(annot_df)} rows")

    print(f"Found {len(tracklets)} tracklets. Controls: n/Enter=next, q=quit")
    cv2.namedWindow("ReID", cv2.WINDOW_NORMAL)

    current_video_dir = None
    kp_data = {}
    mask_data = {}
    ring_mask_data = {}

    for bird_id, video_territory, tracklet_name, tracklet_dir in tracklets:
        video_dir = os.path.dirname(tracklet_dir)
        if video_dir != current_video_dir:
            kp_data, mask_data, ring_mask_data = load_video_overlays(video_dir, show_kp, show_mask)
            current_video_dir = video_dir

        result = visualize_tracklet(bird_id, video_territory, tracklet_name,
                                    tracklet_dir, reid_dir, annot_df, kp_data, mask_data, ring_mask_data)
        if result == "quit":
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ReID tracklets")
    parser.add_argument("--data", required=True, help="Path to ReID root directory")
    parser.add_argument("--keypoints", action="store_true",
                        help="Overlay per-frame keypoint annotations")
    parser.add_argument("--masks", action="store_true",
                        help="Overlay per-frame segmentation masks")
    args = parser.parse_args()
    main(args.data, args.keypoints, args.masks)
