"""
Convert a Label Studio JSON-MIN export directly to a single COCO file.

Expects the labelling template documented in CORVID/TUTORIAL.md:
    <Image name="image" value="$img" .../>
    <RectangleLabels name="label" toName="image">  ...bird bboxes ("bird")
    <PolygonLabels   name="mask"  toName="image">  ...ring colour masks ("R","G",...)

Each task in the JSON-MIN export is expected to look like:
    {
      "img": "<upload_hash>-frame_001.jpg",
      "label": [{"x":.., "y":.., "width":.., "height":.., "rotation":0,
                 "original_width":W, "original_height":H, "rectanglelabels":["bird"]}, ...],
      "mask":  [{"points":[[x,y], ...], "original_width":W, "original_height":H,
                 "polygonlabels":["R"]}, ...]
    }
All x/y/width/height/points are stored by Label Studio as percentages (0-100)
of original_width/original_height and are converted here to absolute pixels.
A task can contain any number of "label"/"mask" entries, including several of
the same class (e.g. two birds, three red rings) — every entry becomes its
own COCO annotation.

The output is a single COCO JSON containing both bird bounding-box
annotations (category "bird", no "segmentation" field) and ring polygon
segmentation annotations (category = colour letter, with "segmentation" +
a bbox derived from the polygon extent). Since this mixes categories that
1_train_ring_segmentation.py and 3_train_yolo.py each only want a subset of,
both scripts accept a --categories filter to pick out the relevant classes
from this combined file (see their docstrings / --help).

`file_name` in the output COCO JSON is the resolved image filename (any
Label Studio upload hash prefix, e.g. "8f3a2b1c-frame_001.jpg", is stripped
back to "frame_001.jpg"). Place your images next to wherever you put the
output JSON, since every CORVID training script resolves `file_name`
relative to the COCO JSON's own directory.

Usage:
    python tools/label_studio_to_coco.py \
        --ls_json  /path/to/label_studio_export.json \
        --out_coco /data/annotations/all.json

    # Optionally split into train/val (by image, shuffled with --seed):
    python tools/label_studio_to_coco.py \
        --ls_json   /path/to/label_studio_export.json \
        --out_coco  /data/annotations/all.json \
        --val_ratio 0.1
    # writes all_train.json / all_val.json
"""

import argparse
import json
import os
import random
import re
from urllib.parse import unquote, urlsplit, parse_qs

_UPLOAD_PREFIX_RE = re.compile(r'^[0-9a-fA-F]{8}-')


# ---------------------------------------------------------------------------
# Label Studio JSON-MIN parsing
# ---------------------------------------------------------------------------

def resolve_image_filename(task):
    """Resolve the on-disk image filename from a Label Studio JSON-MIN task."""
    raw = task.get('img', task.get('image'))
    if raw is None:
        raise KeyError(f"Task {task.get('id')} has no 'img'/'image' field")

    if '?d=' in raw:
        query = parse_qs(urlsplit(raw).query)
        raw = unquote(query['d'][0])

    filename = os.path.basename(raw)
    return _UPLOAD_PREFIX_RE.sub('', filename)


def get_image_size(task):
    """Read image width/height from whichever annotation type is present."""
    for key in ('label', 'mask'):
        items = task.get(key, [])
        if items:
            return items[0]['original_width'], items[0]['original_height']
    raise ValueError(f"Task {task.get('id')} has no 'label' or 'mask' entries to read image size from")


def extract_bboxes(task):
    """Return [(label, [x, y, w, h]), ...] in absolute pixel XYWH, one entry per
    rectangle regardless of how many rectangles share the same class."""
    boxes = []
    for item in task.get('label', []):
        x_ratio = item['original_width'] / 100
        y_ratio = item['original_height'] / 100
        x = item['x'] * x_ratio
        y = item['y'] * y_ratio
        w = item['width'] * x_ratio
        h = item['height'] * y_ratio
        boxes.append((item['rectanglelabels'][0], [x, y, w, h]))
    return boxes


def extract_polygons(task):
    """Return [(label, [x1, y1, x2, y2, ...]), ...] in absolute pixel coordinates,
    one entry per polygon regardless of how many polygons share the same class."""
    polygons = []
    for item in task.get('mask', []):
        x_ratio = item['original_width'] / 100
        y_ratio = item['original_height'] / 100
        flat = []
        for px, py in item['points']:
            flat.append(px * x_ratio)
            flat.append(py * y_ratio)
        polygons.append((item['polygonlabels'][0], flat))
    return polygons


def polygon_bbox_area(flat_points):
    """Bounding box [x,y,w,h] and shoelace area for a flat [x1,y1,x2,y2,...] polygon."""
    xs = flat_points[0::2]
    ys = flat_points[1::2]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    area = abs(area) / 2

    return [x0, y0, x1 - x0, y1 - y0], area


# ---------------------------------------------------------------------------
# COCO builder
# ---------------------------------------------------------------------------

def build_coco(tasks):
    """Build a single COCO dict with both bird bboxes and ring polygon masks."""
    images, annotations, categories = [], [], []
    cat_id_by_name = {}
    ann_id = 1

    def ensure_category(name):
        if name not in cat_id_by_name:
            cat_id_by_name[name] = len(cat_id_by_name) + 1
            categories.append({'id': cat_id_by_name[name], 'name': name})
        return cat_id_by_name[name]

    for image_id, task in enumerate(tasks, start=1):
        boxes = extract_bboxes(task)
        polygons = extract_polygons(task)
        if not boxes and not polygons:
            continue

        w, h = get_image_size(task)
        images.append({'id': image_id, 'file_name': resolve_image_filename(task), 'width': w, 'height': h})

        for label, bbox in boxes:
            annotations.append({
                'id': ann_id, 'image_id': image_id, 'category_id': ensure_category(label),
                'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0,
            })
            ann_id += 1

        for label, flat_points in polygons:
            bbox, area = polygon_bbox_area(flat_points)
            annotations.append({
                'id': ann_id, 'image_id': image_id, 'category_id': ensure_category(label),
                'segmentation': [flat_points], 'bbox': bbox, 'area': area, 'iscrowd': 0,
            })
            ann_id += 1

    return {'images': images, 'annotations': annotations, 'categories': categories}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def split_suffix(path, suffix):
    root, ext = os.path.splitext(path)
    return f'{root}_{suffix}{ext}'


def write_coco(coco, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(coco, f)
    n_bbox = sum(1 for a in coco['annotations'] if 'segmentation' not in a)
    n_seg = len(coco['annotations']) - n_bbox
    print(f'{len(coco["images"])} images, {n_bbox} bbox + {n_seg} polygon annotations -> {out_path}')


def main(args):
    with open(args.ls_json) as f:
        tasks = json.load(f)

    if args.val_ratio is None:
        write_coco(build_coco(tasks), args.out_coco)
        return

    shuffled = list(tasks)
    random.Random(args.seed).shuffle(shuffled)
    n_val = round(len(shuffled) * args.val_ratio)
    val_tasks, train_tasks = shuffled[:n_val], shuffled[n_val:]

    write_coco(build_coco(train_tasks), split_suffix(args.out_coco, 'train'))
    write_coco(build_coco(val_tasks), split_suffix(args.out_coco, 'val'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a Label Studio JSON-MIN export to a single COCO file')
    parser.add_argument('--ls_json',   required=True, help='Path to Label Studio JSON-MIN export')
    parser.add_argument('--out_coco',  required=True, help='Output COCO JSON path (both bboxes and polygon masks)')
    parser.add_argument('--val_ratio', type=float, default=None,
                        help='If set, randomly split by image into train/val COCO files '
                             '(e.g. 0.1 for 10%% validation) instead of writing a single file')
    parser.add_argument('--seed',      type=int, default=534202,
                        help='Random seed for the train/val split (default: 534202)')
    args = parser.parse_args()
    main(args)
