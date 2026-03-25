"""Visualize Action Recognition video clips with optional keypoint overlay"""

import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import sys


ColourDictionary = {
    'hd_bill_tip': (31, 119, 180), 'hd_bill_tiplower': (255, 127, 14),
    'hd_bill_base': (44, 160, 44), 'hd_eye_right': (214, 39, 40),
    'bd_shoulder_right': (148, 103, 189), 'bd_tail_base': (140, 86, 75),
    'bd_tail_tip': (227, 119, 194), 'bd_ankle_right': (127, 127, 127),
    'bd_feet_right': (188, 189, 34), 'hd_eye_left': (23, 190, 207),
    'bd_shoulder_left': (31, 119, 180), 'bd_ankle_left': (255, 127, 14),
    'bd_feet_left': (44, 160, 44)
}
KP_NAMES = list(ColourDictionary.keys())


def load_keypoints_csv(csv_path):
    """Load per-frame keypoints CSV (long format: Frame, Cam, ID, Keypoint, x, y, Conf).
    Returns dict {frame_idx: list of (keypoint_name, x, y, conf)}."""
    df = pd.read_csv(csv_path)
    kp_data = {}
    for frame_idx, group in df.groupby("Frame"):
        kp_data[int(frame_idx)] = list(zip(group["Keypoint"], group["x"], group["y"], group["Conf"]))
    return kp_data


def draw_keypoints(img, keypoints):
    """Draw keypoints from a list of (keypoint_name, x, y, conf) tuples."""
    for kp_name, x, y, conf in keypoints:
        if conf > 0 and kp_name in ColourDictionary:
            cv2.circle(img, (int(x), int(y)), 5, ColourDictionary[kp_name], -1)
    return img


def load_masks_csv(csv_path):
    """Load per-frame mask CSV (long format: Frame, Cam, ID, Class, Segmentation).
    Returns dict {frame_idx: list of contour arrays}."""
    df = pd.read_csv(csv_path)
    mask_data = {}
    for frame_idx, group in df.groupby("Frame"):
        contours = []
        for seg_json in group["Segmentation"]:
            # seg_json is a nested list: birds -> contours -> flat [x, y, x, y, ...]
            birds = json.loads(seg_json)
            for bird in birds:
                for contour in bird:
                    pts = np.array(contour, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                    contours.append(pts)
        mask_data[int(frame_idx)] = contours
    return mask_data


def draw_masks(img, contours):
    """Draw segmentation mask contours as a semi-transparent overlay."""
    overlay = img.copy()
    for pts in contours:
        cv2.fillPoly(overlay, [pts], color=(0, 255, 255))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for pts in contours:
        cv2.polylines(img, [pts], isClosed=True, color=(0, 200, 200), thickness=1)
    return img


def visualize_clip(data_dir, video_path, behaviour, split, show_kp, show_mask):
    """Play a single video clip, overlaying keypoints and masks if enabled."""
    stem = os.path.splitext(os.path.basename(video_path))[0]

    kp_data = {}
    if show_kp:
        csv_path = os.path.join(data_dir, "Videos_Keypoints", stem + "_keypoints.csv")
        if os.path.exists(csv_path):
            kp_data = load_keypoints_csv(csv_path)
        # else:
            # import ipdb; ipdb.set_trace()  

    mask_data = {}
    if show_mask:
        csv_path = os.path.join(data_dir, "Videos_Masks", stem + "_masks.csv")
        if os.path.exists(csv_path):
            mask_data = load_masks_csv(csv_path)

    cap = cv2.VideoCapture(os.path.join(data_dir, video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return "next"

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                ret, frame = cap.read()

            if frame_idx in mask_data:
                frame = draw_masks(frame, mask_data[frame_idx])
            if frame_idx in kp_data:
                frame = draw_keypoints(frame, kp_data[frame_idx])

            cv2.putText(frame, stem, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Behaviour: {behaviour}  Split: {split}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            cv2.putText(frame, "n=next  q=quit  space=pause", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("ActionRecognition", frame)
            frame_idx += 1

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            cap.release()
            return "quit"
        elif key == ord('n'):
            cap.release()
            return "next"
        elif key == ord(' '):
            paused = not paused


def main(data_dir, show_kp, show_mask):
    annot_path = os.path.join(data_dir, "BehaviourAnnotations.csv")
    if not os.path.exists(annot_path):
        print(f"BehaviourAnnotations.csv not found in {data_dir}")
        sys.exit(1)

    df = pd.read_csv(annot_path)
    df = df[df["Corrupted"] == False].reset_index(drop=True)
    print(f"Loaded {len(df)} clips from BehaviourAnnotations.csv. Controls: n=next, q=quit, space=pause/resume")

    cv2.namedWindow("ActionRecognition", cv2.WINDOW_NORMAL)

    for _, row in df.iterrows():
        print(row["VideoPath"])
        result = visualize_clip(data_dir, row["VideoPath"], row["Behaviour"], row["Split"],
                                show_kp, show_mask)
        if result == "quit":
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Action Recognition clips")
    parser.add_argument("--data", required=True,
                        help="Path to ActionRecognition/ root directory")
    parser.add_argument("--keypoints", action="store_true",
                        help="Overlay keypoint annotations")
    parser.add_argument("--masks", action="store_true",
                        help="Overlay segmentation masks")
    args = parser.parse_args()
    main(args.data, args.keypoints, args.masks)
