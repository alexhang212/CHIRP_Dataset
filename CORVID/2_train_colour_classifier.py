"""
Train a Random Forest colour classifier for bird rings.

Input: COCO-format JSON(s) with **polygon instance masks** for rings, where
each annotation's category name is a single-character colour code (e.g. "R"
for red) — the same annotation format used by 1_train_ring_segmentation.py,
just with per-colour categories instead of a single generic "ring" class.

    --coco_train / --coco_val : separate COCO JSONs for the train/test split.
        Image `file_name` paths are resolved relative to the directory
        containing each JSON (so the dataset can be moved/shared as a single
        self-contained folder).
    --colours_json : JSON file with an ordered list of valid colour class
        names, e.g. ["A", "B", "C", "G", "L", "M", "O", "P", "R", "S", "W", "Y"].
        Annotations whose category name is not in this list are skipped
        (e.g. a generic "ring" category used for segmentation-only data).

Preprocessing: each ring polygon is perspective-warped to a 20x20 BGR crop
(same crop geometry as run_rf() in 4_run_inference.py), then converted to a
30-dim HSV histogram feature (10 bins per channel H/S/V).

Outputs saved to --output_dir:
    RandomForestModel.p       — trained RF model (RandomForestRegressor)
    TrainImagesFeatures.p     — (N_train, 30) raw feature array, needed at inference
                                for StandardScaler re-fitting
    Classes.p                 — ordered list of colour class names
    ConfusionMatrix_RF.png    — confusion matrix on test set
    Accuracy.csv              — per-class and overall top-1/top-3 accuracy

Usage:
    python 2_train_colour_classifier.py \
        --coco_train   /path/to/train_rings.json \
        --coco_val     /path/to/val_rings.json \
        --colours_json /path/to/colours.json \
        --output_dir   /path/to/save/models

    python 2_train_colour_classifier.py \
        --coco_train   /path/to/train_rings.json \
        --coco_val     /path/to/val_rings.json \
        --colours_json /path/to/colours.json \
        --output_dir   /path/to/save/models \
        --tune   # run RandomizedSearchCV hyperparameter search (slower)
"""

import argparse
import json
import os
import pickle


def _lazy_imports():
    global cv2, np, pd, plt, RandomForestRegressor, confusion_matrix, ConfusionMatrixDisplay
    global top_k_accuracy_score, RandomizedSearchCV, StandardScaler
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# COCO ring-crop extraction
# ---------------------------------------------------------------------------

def crop_ring_from_polygon(img, polygon, out_dim=(20, 20)):
    """Perspective-warp a single ring polygon into a fixed-size crop.

    Mirrors the crop geometry used at inference time in run_rf()
    (4_run_inference.py), so train/test features stay comparable.
    """
    poly = np.array([[round(x), round(y)] for x, y in polygon])
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    cropped = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(poly)
    src = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    dst = np.array([[0, 0], [out_dim[0] - 1, 0],
                     [out_dim[0] - 1, out_dim[1] - 1], [0, out_dim[1] - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(cropped, M, out_dim)


def load_ring_crops(coco_json_path, colours):
    """
    Load ring crops + colour labels from a COCO JSON with polygon masks.

    Images are resolved relative to the directory containing `coco_json_path`.
    Annotations whose category name is not in `colours` are skipped.

    Returns:
        images: list of (20, 20, 3) uint8 BGR crops
        labels: list of colour class names (same length as images)
    """
    with open(coco_json_path) as f:
        data = json.load(f)

    image_root = os.path.dirname(os.path.abspath(coco_json_path))
    img_info_by_id = {img['id']: img for img in data['images']}
    cat_name_by_id = {cat['id']: cat['name'] for cat in data['categories']}
    colour_set = set(colours)

    anns_by_img = {}
    for ann in data['annotations']:
        if cat_name_by_id.get(ann['category_id']) not in colour_set:
            continue
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    images, labels = [], []
    for image_id, anns in anns_by_img.items():
        img_path = os.path.join(image_root, img_info_by_id[image_id]['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f'WARNING: could not read image {img_path}, skipping its {len(anns)} annotation(s)')
            continue

        for ann in anns:
            if isinstance(ann['segmentation'], dict):
                raise ValueError(
                    f"Annotation {ann['id']} uses RLE segmentation; only polygon "
                    f"segmentation (list of [x1,y1,x2,y2,...]) is supported.")
            flat = ann['segmentation'][0]
            polygon = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
            images.append(crop_ring_from_polygon(img, polygon))
            labels.append(cat_name_by_id[ann['category_id']])

    return images, labels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_hsv_features(images):
    """Convert BGR ring crops to 30-dim HSV histogram features."""
    features = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [10], [0, 256]).flatten()
        s = cv2.calcHist([hsv], [1], None, [10], [0, 256]).flatten()
        v = cv2.calcHist([hsv], [2], None, [10], [0, 256]).flatten()
        features.append(np.concatenate([h, s, v]))
    return np.array(features)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rf(X_train, y_onehot, tune=False):
    """Train (or hyperparameter-tune) a RandomForestRegressor."""
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

    if tune:
        param_grid = {
            'n_estimators': [int(x) for x in np.linspace(10, 2000, 20)],
            'max_features': ['log2', 'sqrt'],
            'max_depth': [int(x) for x in np.linspace(10, 110, 11)] + [None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
        }
        search = RandomizedSearchCV(
            rf, param_grid, n_iter=50, cv=3,
            verbose=2, random_state=42, n_jobs=1
        )
        search.fit(X_train, y_onehot)
        print('Best params:', search.best_params_)
        return search.best_estimator_

    rf.fit(X_train, y_onehot)
    return rf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test_idx, classes, output_dir):
    """Compute top-1/top-3 accuracy, per-class breakdown, and confusion matrix."""
    pred = model.predict(X_test)

    top1 = top_k_accuracy_score(y_test_idx, pred, k=1)
    top3 = top_k_accuracy_score(y_test_idx, pred, k=3)

    per_class_top1 = {}
    per_class_top3 = {}
    for i, cls in enumerate(classes):
        subset = [pred[j] for j in range(len(pred)) if y_test_idx[j] == i]
        if not subset:
            per_class_top1[cls] = float('nan')
            per_class_top3[cls] = float('nan')
            continue
        labels_i = [i] * len(subset)
        per_class_top1[cls] = top_k_accuracy_score(
            labels_i, np.array(subset), k=1, labels=list(range(len(classes))))
        per_class_top3[cls] = top_k_accuracy_score(
            labels_i, np.array(subset), k=3, labels=list(range(len(classes))))

    per_class_top1['Overall'] = top1
    per_class_top3['Overall'] = top3

    print(f'Top-1 accuracy: {top1:.4f}')
    print(f'Top-3 accuracy: {top3:.4f}')

    df = pd.DataFrame({
        'Class': list(per_class_top1.keys()),
        'Top1Accuracy': list(per_class_top1.values()),
        'Top3Accuracy': list(per_class_top3.values()),
    })
    df.to_csv(os.path.join(output_dir, 'Accuracy.csv'), index=False)

    pred_idx = np.argmax(pred, axis=1)
    cm = confusion_matrix(y_test_idx, pred_idx)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Random Forest — Colour Classification')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ConfusionMatrix_RF.png'))
    plt.close()

    print(f'Saved Accuracy.csv and ConfusionMatrix_RF.png to {output_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    _lazy_imports()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.colours_json) as f:
        colours = json.load(f)

    print(f'Loading training crops from {args.coco_train}')
    train_images, train_labels = load_ring_crops(args.coco_train, colours)
    print(f'Loading test crops from {args.coco_val}')
    test_images, test_labels = load_ring_crops(args.coco_val, colours)

    # Preserve the order given in --colours_json; drop colours never seen in training.
    classes = [c for c in colours if c in set(train_labels)]
    print(f'Classes ({len(classes)}): {classes}')

    all_images = train_images + test_images
    all_features = extract_hsv_features(all_images)

    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_features)

    n_train = len(train_images)
    X_train = all_scaled[:n_train]
    X_test  = all_scaled[n_train:]

    y_train_idx = [classes.index(c) for c in train_labels]
    y_test_idx  = [classes.index(c) for c in test_labels]

    y_train_onehot = np.eye(len(classes))[y_train_idx]

    print(f'Train: {n_train} samples | Test: {len(test_images)} samples')
    print('Training Random Forest...')
    model = train_rf(X_train, y_train_onehot, tune=args.tune)

    # Save raw (unscaled) train features for use at inference time (scaler re-fit)
    train_features_raw = all_features[:n_train]

    pickle.dump(model,             open(os.path.join(args.output_dir, 'RandomForestModel.p'), 'wb'))
    pickle.dump(train_features_raw, open(os.path.join(args.output_dir, 'TrainImagesFeatures.p'), 'wb'))
    pickle.dump(classes,           open(os.path.join(args.output_dir, 'Classes.p'), 'wb'))
    print(f'Saved model, features, and classes to {args.output_dir}')

    evaluate(model, X_test, y_test_idx, classes, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Random Forest colour classifier for bird rings')
    parser.add_argument('--coco_train',   required=True,
                        help='COCO JSON with training ring polygon masks (category name = colour code)')
    parser.add_argument('--coco_val',     required=True,
                        help='COCO JSON with test/validation ring polygon masks')
    parser.add_argument('--colours_json', required=True,
                        help='JSON file with an ordered list of valid colour class names, '
                             'e.g. ["A","B","C","G","L","M","O","P","R","S","W","Y"]')
    parser.add_argument('--output_dir',   required=True,
                        help='Directory to save trained model and evaluation outputs')
    parser.add_argument('--tune', action='store_true',
                        help='Run RandomizedSearchCV hyperparameter tuning (slower but may improve accuracy)')
    args = parser.parse_args()
    main(args)
