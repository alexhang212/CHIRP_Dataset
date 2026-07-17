"""
Train an MMDetection instance segmentation model to detect and segment colour rings.

Supports MaskRCNN (ResNet50/101), Cascade Mask R-CNN (ResNet50), and Mask2Former.
Reads COCO-format annotations with polygon instance masks.

Usage:
    python 1_train_ring_segmentation.py \
        --coco_train /path/to/train_rings.json \
        --coco_val   /path/to/val_rings.json \
        --save_dir   /path/to/save/weights \
        --model_type maskrcnn50 \
        --epochs 50 \
        --batch 2

Image `file_name` paths in each COCO JSON are resolved relative to the
directory containing that JSON (train and val can live in different folders).

Cropping to the bird box before training:
    At inference (see mmdet_mask_inference() in 4_run_inference.py), this
    model runs on a crop of just the tracked bird, not the full video frame.
    So if --coco_train/--coco_val contain bird bounding boxes (category name
    given by --bird_category, default "bird") alongside ring polygon masks —
    e.g. the combined output of tools/label_studio_to_coco.py — this script
    automatically crops each bird instance out of its source image before
    training, keeping only the ring masks that fall inside that bird's box
    (a mask inside more than one bird box goes to the smallest/tightest
    box; masks inside no box are dropped, with a count printed). Cropped
    images + a remapped COCO file are cached under
    {save_dir}/ring_crops_train and {save_dir}/ring_crops_val.

    If the COCO file has no --bird_category category at all (e.g. a
    ring-only dataset that's already pre-cropped, like CHIRP RingMask),
    it's used as-is with no cropping step.

If your COCO file also contains other annotation types, pass --categories
to restrict training to the ring/colour classes only, e.g. --categories
"R,G,B,Y" or --categories "ring". This also controls which ring categories
are kept during the crop step above.
"""

import argparse
import colorsys
import json
import os

import cv2


# ---------------------------------------------------------------------------
# Config helpers (standalone versions of MAAP3D equivalents)
# ---------------------------------------------------------------------------

def get_classes_from_coco(json_path):
    """Return sorted list of category names from a COCO annotation file."""
    with open(json_path) as f:
        data = json.load(f)
    return sorted([cat['name'] for cat in data['categories']])


def parse_categories_arg(categories, json_path):
    """Parse a comma-separated --categories string, or auto-detect all
    categories from the COCO file if not given."""
    if categories is None:
        return get_classes_from_coco(json_path)

    requested = [c.strip() for c in categories.split(',') if c.strip()]
    available = set(get_classes_from_coco(json_path))
    missing = [c for c in requested if c not in available]
    if missing:
        raise ValueError(f"--categories {missing} not found in {json_path}. Available: {sorted(available)}")
    return requested


def image_root_for(coco_json_path):
    """Directory that image `file_name` paths in the COCO JSON are relative to."""
    return os.path.dirname(os.path.abspath(coco_json_path))


# ---------------------------------------------------------------------------
# Crop-to-bird preprocessing
# ---------------------------------------------------------------------------

def polygon_centroid(flat_points):
    xs = flat_points[0::2]
    ys = flat_points[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def point_in_box(point, box):
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def assign_ring_to_box(flat_points, boxes):
    """Index of the smallest bird box whose area contains the polygon centroid, or None."""
    centroid = polygon_centroid(flat_points)
    candidates = [i for i, box in enumerate(boxes) if point_in_box(centroid, box)]
    if not candidates:
        return None
    return min(candidates, key=lambda i: boxes[i][2] * boxes[i][3])


def remap_polygon_to_crop(flat_points, crop_box):
    """Shift a polygon into crop-local coordinates and clip to the crop bounds."""
    bx, by, bw, bh = crop_box
    out = []
    for i in range(0, len(flat_points), 2):
        x = min(max(flat_points[i] - bx, 0), bw)
        y = min(max(flat_points[i + 1] - by, 0), bh)
        out.extend([x, y])
    return out


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


def prepare_ring_crops(coco_json_path, out_dir, bird_category, ring_categories):
    """
    Crop each bird instance out of `coco_json_path`'s images and remap the
    ring polygon masks that fall inside its box, writing cropped images and a
    new COCO JSON to `out_dir`.

    If `bird_category` isn't a category in the file at all, the dataset is
    assumed to already be bird-cropped and is returned unchanged.

    Returns:
        Path to the COCO JSON to actually train on (either the new cropped
        one, or the original `coco_json_path` if no cropping was needed).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    cat_names = {c['name'] for c in coco['categories']}
    if bird_category not in cat_names:
        return coco_json_path

    image_root = image_root_for(coco_json_path)
    os.makedirs(out_dir, exist_ok=True)

    cat_name_by_id = {c['id']: c['name'] for c in coco['categories']}
    anns_by_img = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    out_images, out_annotations, out_categories = [], [], []
    cat_id_by_name = {}
    out_image_id = 1
    ann_id = 1
    n_dropped_rings = 0

    def ensure_category(name):
        if name not in cat_id_by_name:
            cat_id_by_name[name] = len(cat_id_by_name) + 1
            out_categories.append({'id': cat_id_by_name[name], 'name': name})
        return cat_id_by_name[name]

    for img_info in coco['images']:
        anns = anns_by_img.get(img_info['id'], [])
        boxes = [a['bbox'] for a in anns if cat_name_by_id[a['category_id']] == bird_category]
        if not boxes:
            continue

        rings = []
        for a in anns:
            name = cat_name_by_id[a['category_id']]
            if name == bird_category or 'segmentation' not in a:
                continue
            if ring_categories is not None and name not in ring_categories:
                continue
            rings.append((name, a['segmentation'][0]))

        src_path = os.path.join(image_root, img_info['file_name'])
        img = cv2.imread(src_path)
        if img is None:
            print(f'WARNING: could not read {src_path}, skipping')
            continue
        h_full, w_full = img.shape[:2]

        rings_by_box = {i: [] for i in range(len(boxes))}
        for name, flat_points in rings:
            box_idx = assign_ring_to_box(flat_points, boxes)
            if box_idx is None:
                n_dropped_rings += 1
                continue
            rings_by_box[box_idx].append((name, flat_points))

        stem, ext = os.path.splitext(os.path.basename(img_info['file_name']))

        for box_idx, box in enumerate(boxes):
            x, y, w, h = box
            x0, y0 = max(0, round(x)), max(0, round(y))
            x1, y1 = min(w_full, round(x + w)), min(h_full, round(y + h))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1]

            crop_name = f'{stem}_bird{box_idx}{ext}'
            cv2.imwrite(os.path.join(out_dir, crop_name), crop)

            out_images.append({
                'id': out_image_id, 'file_name': crop_name,
                'width': x1 - x0, 'height': y1 - y0,
            })

            for name, flat_points in rings_by_box[box_idx]:
                remapped = remap_polygon_to_crop(flat_points, [x0, y0, x1 - x0, y1 - y0])
                bbox, area = polygon_bbox_area(remapped)
                out_annotations.append({
                    'id': ann_id, 'image_id': out_image_id, 'category_id': ensure_category(name),
                    'segmentation': [remapped], 'bbox': bbox, 'area': area, 'iscrowd': 0,
                })
                ann_id += 1

            out_image_id += 1

    out_coco = {'images': out_images, 'annotations': out_annotations, 'categories': out_categories}
    out_json_path = os.path.join(out_dir, 'cropped.json')
    with open(out_json_path, 'w') as f:
        json.dump(out_coco, f)

    print(f'[{os.path.basename(coco_json_path)}] {len(out_images)} bird crops -> {out_dir} '
          f'({len(out_annotations)} ring annotations remapped)')
    if n_dropped_rings:
        print(f'  WARNING: {n_dropped_rings} ring annotation(s) had no containing bird box and were dropped')

    return out_json_path


def generate_palette(num_classes):
    palette = []
    for i in range(num_classes):
        rgb = colorsys.hsv_to_rgb(i / num_classes, 0.9, 0.9)
        palette.append([int(c * 255) for c in rgb])
    return palette


def set_num_classes(cfg, model_type, num_classes):
    if model_type.startswith('maskrcnn'):
        cfg.model.roi_head.bbox_head.num_classes = num_classes
        cfg.model.roi_head.mask_head.num_classes = num_classes
    elif model_type.startswith('cascade'):
        for bbox_head in cfg.model.roi_head.bbox_head:
            bbox_head.num_classes = num_classes
        cfg.model.roi_head.mask_head.num_classes = num_classes
    elif model_type.startswith('mask2former'):
        cfg.model.panoptic_head.num_things_classes = num_classes
        cfg.model.panoptic_head.num_stuff_classes = 0
        cfg.model.panoptic_fusion_head.num_things_classes = num_classes
        cfg.model.panoptic_fusion_head.num_stuff_classes = 0
        cfg.model.panoptic_head.loss_cls.class_weight = [1.0] * num_classes + [0.1]


def update_lr_milestones(cfg, epochs):
    """Scale LR milestones proportionally to total epochs (default schedule is 12 epochs)."""
    default_epochs = 12
    default_milestones = [8, 11]
    for scheduler in cfg.param_scheduler:
        if scheduler.get('type') == 'MultiStepLR':
            scale = epochs / default_epochs
            milestones = [int(m * scale) for m in default_milestones]
            milestones = [m for m in milestones if m < epochs]
            if not milestones:
                milestones = [int(epochs * 0.67), int(epochs * 0.92)]
            scheduler['milestones'] = milestones
            scheduler['end'] = epochs


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    from mmengine.config import Config
    from mmengine.runner import Runner

    configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    config_map = {
        'maskrcnn50':        'maskrcnn50.py',
        'maskrcnn101':       'maskrcnn101.py',
        'cascade_maskrcnn50': 'cascade_maskrcnn50.py',
        'mask2former':       'mask2former.py',
    }

    if args.model_type not in config_map:
        raise ValueError(f"Unknown model_type '{args.model_type}'. "
                         f"Choose from: {', '.join(sorted(config_map))}")

    cfg_path = os.path.join(configs_dir, config_map[args.model_type])
    cfg = Config.fromfile(cfg_path)

    if args.pretrained:
        cfg.load_from = args.pretrained

    os.makedirs(args.save_dir, exist_ok=True)

    ring_categories = None
    if args.categories is not None:
        ring_categories = {c.strip() for c in args.categories.split(',') if c.strip()}

    coco_train = prepare_ring_crops(
        args.coco_train, os.path.join(args.save_dir, 'ring_crops_train'), args.bird_category, ring_categories)
    coco_val = prepare_ring_crops(
        args.coco_val, os.path.join(args.save_dir, 'ring_crops_val'), args.bird_category, ring_categories)

    classes = parse_categories_arg(args.categories, coco_train)
    num_classes = len(classes)
    print(f'Classes ({num_classes}): {classes}')

    set_num_classes(cfg, args.model_type, num_classes)

    palette = generate_palette(num_classes)
    metainfo = dict(classes=tuple(classes), palette=palette)
    cfg.metainfo = metainfo

    cfg.data_root = image_root_for(coco_train)
    cfg.work_dir = args.save_dir

    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = args.save_interval
    cfg.default_hooks.checkpoint.interval = args.save_interval
    cfg.default_hooks.checkpoint.save_best = 'coco/segm_mAP'
    cfg.resume = args.resume

    update_lr_milestones(cfg, args.epochs)

    for loader_key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        loader = getattr(cfg, loader_key)
        ann_file = coco_train if loader_key == 'train_dataloader' else coco_val
        loader.batch_size = args.batch if loader_key == 'train_dataloader' else 1
        loader.num_workers = args.num_workers
        loader.dataset.data_root = image_root_for(ann_file)
        loader.dataset.ann_file = ann_file
        loader.dataset.data_prefix = dict(img='')
        loader.dataset.metainfo = metainfo
        if args.num_workers == 0:
            loader.persistent_workers = False

    cfg.val_evaluator.ann_file = coco_val
    cfg.test_evaluator.ann_file = coco_val

    print(cfg.pretty_text)

    runner = Runner.from_cfg(cfg)
    runner.train()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MMDetection ring instance segmentation model')
    parser.add_argument('--coco_train',    required=True, help='COCO JSON with training instance masks')
    parser.add_argument('--coco_val',      required=True, help='COCO JSON for validation')
    parser.add_argument('--categories',    default=None,
                        help='Comma-separated list of ring/colour category names to train on (e.g. '
                             '"R,G,B,Y" or "ring"). Also controls which ring masks are kept during the '
                             'crop-to-bird step. Default: use every non-bird category present.')
    parser.add_argument('--bird_category', default='bird',
                        help='Category name for bird bounding boxes (default: "bird"). If the COCO file '
                             'has no such category, no crop-to-bird step is performed.')
    parser.add_argument('--save_dir',      required=True, help='Output directory for weights and logs')
    parser.add_argument('--model_type',    default='maskrcnn50',
                        choices=['maskrcnn50', 'maskrcnn101', 'cascade_maskrcnn50', 'mask2former'],
                        help='Model architecture (default: maskrcnn50)')
    parser.add_argument('--epochs',        type=int, default=50)
    parser.add_argument('--batch',         type=int, default=2)
    parser.add_argument('--num_workers',   type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume',        action='store_true',
                        help='Resume training from existing checkpoint in save_dir')
    parser.add_argument('--pretrained',    default=None,
                        help='Path to pretrained weights (.pth) to fine-tune from')
    args = parser.parse_args()
    train(args)
