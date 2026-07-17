"""
Train a YOLOv8 model for whole-bird detection.

Converts COCO bounding box annotations to YOLO format, then trains with
the ultralytics library. Suitable for detecting birds in video frames as
the first stage of the CORVID pipeline.

Usage:
    python 3_train_yolo.py \
        --coco_train  /path/to/train_birds.json \
        --coco_val    /path/to/val_birds.json \
        --output_dir  /path/to/yolo_dataset \
        --save_dir    /path/to/save/weights \
        --model       yolov8n.pt \
        --epochs      100

Image `file_name` paths in each COCO JSON are resolved relative to the
directory containing that JSON (train and val can live in different folders).

If your COCO file also contains other annotation types (e.g. ring polygon
masks from tools/label_studio_to_coco.py's combined output), pass
--categories to restrict training to the bird class only, e.g.
--categories "bird". Annotations for any other category are ignored.

After training, the best weights will be at:
    {save_dir}/weights/best.pt
"""

import argparse
import json
import os
import shutil


# ---------------------------------------------------------------------------
# COCO → YOLO conversion
# ---------------------------------------------------------------------------

def get_classes_from_coco(coco_json):
    """Return category names from a COCO file, ordered by category id."""
    with open(coco_json) as f:
        data = json.load(f)
    return [c['name'] for c in sorted(data['categories'], key=lambda c: c['id'])]


def parse_categories_arg(categories, coco_json):
    """Parse a comma-separated --categories string, or auto-detect every
    category (ordered by id) from the COCO file if not given."""
    if categories is None:
        return get_classes_from_coco(coco_json)

    requested = [c.strip() for c in categories.split(',') if c.strip()]
    available = set(get_classes_from_coco(coco_json))
    missing = [c for c in requested if c not in available]
    if missing:
        raise ValueError(f"--categories {missing} not found in {coco_json}. Available: {sorted(available)}")
    return requested


def coco_to_yolo(coco_json, output_dir, split, classes):
    """
    Convert a COCO bbox JSON to YOLO label files.

    Args:
        coco_json:  Path to COCO annotation JSON (bbox only, no masks needed).
                    Image `file_name` paths are resolved relative to the
                    directory containing this JSON.
        output_dir: Root YOLO dataset directory.
        split:      'train' or 'val' — determines output subdirectory.
        classes:    Ordered list of class names to emit. Annotations whose
                    category isn't in this list are skipped (e.g. non-bird
                    categories in a combined COCO file).
    """
    with open(coco_json) as f:
        data = json.load(f)

    images_dir = os.path.dirname(os.path.abspath(coco_json))
    cat_id_to_idx = {c['id']: classes.index(c['name']) for c in data['categories'] if c['name'] in classes}

    # Group annotations by image id
    anns_by_img = {}
    for ann in data['annotations']:
        if ann['category_id'] not in cat_id_to_idx:
            continue
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    img_out_dir   = os.path.join(output_dir, 'images', split)
    label_out_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(img_out_dir,   exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    for img_info in data['images']:
        img_id   = img_info['id']
        filename = img_info['file_name']
        W = img_info['width']
        H = img_info['height']

        src = os.path.join(images_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(img_out_dir, os.path.basename(filename)))

        label_path = os.path.join(
            label_out_dir,
            os.path.splitext(os.path.basename(filename))[0] + '.txt'
        )

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cls_idx = cat_id_to_idx[ann['category_id']]
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f'{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))


def create_yaml(output_dir, classes):
    """Write dataset.yaml for ultralytics YOLO training."""
    import yaml
    yaml_data = {
        'path':  os.path.abspath(output_dir),
        'train': 'images/train',
        'val':   'images/val',
        'nc':    len(classes),
        'names': classes,
    }
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    return yaml_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_yolo(yaml_path, save_dir, model_name, epochs, imgsz, batch):
    import yaml
    from ultralytics import YOLO
    model = YOLO(model_name)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=save_dir,
        name='bird_detection',
        exist_ok=True,
    )
    best_weights = os.path.join(save_dir, 'bird_detection', 'weights', 'best.pt')
    print(f'\nTraining complete. Best weights: {best_weights}')
    return best_weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    classes = parse_categories_arg(args.categories, args.coco_train)
    print(f'Classes ({len(classes)}): {classes}')

    print('Converting COCO → YOLO format...')
    coco_to_yolo(args.coco_train, args.output_dir, 'train', classes)
    coco_to_yolo(args.coco_val,   args.output_dir, 'val',   classes)

    yaml_path = create_yaml(args.output_dir, classes)
    print(f'Dataset YAML: {yaml_path}')

    os.makedirs(args.save_dir, exist_ok=True)
    train_yolo(yaml_path, args.save_dir, args.model, args.epochs, args.imgsz, args.batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 for whole-bird detection')
    parser.add_argument('--coco_train',  required=True,
                        help='COCO JSON with bird bounding box annotations (training split)')
    parser.add_argument('--coco_val',    required=True,
                        help='COCO JSON (validation split)')
    parser.add_argument('--categories',  default=None,
                        help='Comma-separated list of category names to train on (e.g. "bird"). '
                             'Restricts training to these classes when the COCO file also contains '
                             'other annotation types. Default: use every category present.')
    parser.add_argument('--output_dir',  required=True,
                        help='Directory to write YOLO-format dataset and YAML config')
    parser.add_argument('--save_dir',    required=True,
                        help='Directory to save YOLO training outputs and weights')
    parser.add_argument('--model',       default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                        help='Base YOLO model size (default: yolov8n.pt)')
    parser.add_argument('--epochs',      type=int, default=100)
    parser.add_argument('--imgsz',       type=int, default=640,
                        help='Input image size for training (default: 640)')
    parser.add_argument('--batch',       type=int, default=16)
    args = parser.parse_args()
    main(args)
