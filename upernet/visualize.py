"""使用训练好的 checkpoint 生成分割预览图。

从验证集选取若干样本进行推理，每张输出包含「输入 / 真值 / 预测 / 叠加」的四格 PNG，
配色遵循 ade_palette.py 中的 mmsegmentation ADE20K 调色板。

    python upernet/visualize.py --ckpt checkpoints/best.pth \
        --output-dir docs/images --num-images 4
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ade_palette import CLASSES, IGNORE_COLOR, PALETTE  # noqa: E402
from vavae_upernet import VAVAEUPerNet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORE_INDEX = 255


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize VAVAE+UPerNet predictions on ADE20K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "best.pth"))
    parser.add_argument("--vavae-ckpt", default=os.path.join(ROOT, "vavae-imagenet256-f16d32-dinov2.pt"))
    parser.add_argument("--data-root", default=os.path.join(ROOT, "data", "ADEChallengeData2016"))
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--images", nargs="*", default=None,
                        help="explicit sample ids such as ADE_val_00000001")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--scan", type=int, default=300,
                        help="how many samples to scan when picking diverse images")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "docs", "images"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def colorize(labels):
    """将类别索引（0..149，255 表示忽略）映射为 RGB 彩色图像。"""
    palette = np.asarray(PALETTE, dtype=np.uint8)
    colors = np.vstack([palette, np.asarray(IGNORE_COLOR, dtype=np.uint8)[None, :]])
    ids = np.asarray(labels, dtype=np.int64).copy()
    ids[ids == IGNORE_INDEX] = len(PALETTE)
    return Image.fromarray(colors[ids])


def remap(mask):
    labels = np.asarray(mask, dtype=np.int64) - 1
    labels[labels < 0] = IGNORE_INDEX
    return labels


def image_miou(pred, label, num_classes=150):
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (label >= 0) & (label < num_classes)
    idx = num_classes * label[valid] + pred[valid]
    hist += np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    diag = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - diag
    iou = diag / np.maximum(union, 1.0)
    present = hist.sum(1) > 0
    return float(iou[present].mean()) if present.any() else 0.0


def pick_samples(args):
    image_dir = os.path.join(args.data_root, "images", args.split)
    ann_dir = os.path.join(args.data_root, "annotations", args.split)
    if args.images:
        return [(name, os.path.join(image_dir, name + ".jpg"), os.path.join(ann_dir, name + ".png"))
                for name in args.images]

    ids = sorted(
        os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(".jpg")
    )[:args.scan]
    scored = []
    for name in ids:
        ann_path = os.path.join(ann_dir, name + ".png")
        img_path = os.path.join(image_dir, name + ".jpg")
        if not os.path.isfile(ann_path):
            continue
        with Image.open(img_path) as image:
            width, height = image.size
        ratio = width / max(height, 1)
        if not 0.6 <= ratio <= 1.8:
            continue
        labels = remap(Image.open(ann_path).resize((128, 128), Image.NEAREST))
        class_count = len(np.unique(labels[labels != IGNORE_INDEX]))
        scored.append((class_count, name, img_path, ann_path))
    scored.sort(reverse=True)
    return [(name, img, ann) for _, name, img, ann in scored[:args.num_images]]


def build_model(args, device):
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = VAVAEUPerNet(args.vavae_ckpt, num_classes=150)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.freeze_backbone()
    model.to(device).eval()
    return model


def compose(images, titles, panel_size, footer):
    label_height, footer_height = 24, 26
    width = panel_size * len(images)
    height = panel_size + label_height + footer_height
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(images, titles)):
        x = index * panel_size
        canvas.paste(panel, (x, label_height))
        draw.text((x + 6, 6), title, fill=(20, 20, 20))
        if index:
            draw.line([(x, label_height), (x, height - footer_height)], fill=(210, 210, 210))
    draw.text((6, height - footer_height + 6), footer, fill=(70, 70, 70))
    return canvas


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    samples = pick_samples(args)
    if not samples:
        raise SystemExit("no usable samples found")
    print(f"device={device} samples={len(samples)}")

    model = build_model(args, device)
    rows = []
    for index, (name, image_path, ann_path) in enumerate(samples, 1):
        original = Image.open(image_path).convert("RGB")
        target_size = original.size
        panel_size = args.panel_size

        x = np.asarray(original.resize((args.image_size, args.image_size), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        tensor = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred_small = logits.argmax(1)[0].cpu().numpy().astype(np.int64)

        gt_small = remap(Image.open(ann_path).resize(
            (args.image_size, args.image_size), Image.NEAREST
        ))
        miou = image_miou(pred_small, gt_small)

        gt_display = colorize(remap(Image.open(ann_path)))
        pred_display = colorize(
            Image.fromarray(pred_small.astype(np.uint8)).resize(target_size, Image.NEAREST)
        )
        resized_original = original.resize((panel_size, panel_size), Image.BILINEAR)
        overlay = Image.blend(
            resized_original, pred_display.resize((panel_size, panel_size), Image.NEAREST), 0.5
        )

        counts = np.bincount(pred_small.ravel(), minlength=len(CLASSES))
        classes = sorted(np.unique(pred_small).tolist(), key=lambda c: -counts[c])
        main_classes = ", ".join(CLASSES[c] for c in classes[:4])
        footer = f"{name} | image mIoU {miou:.3f} | largest predicted: {main_classes}"
        canvas = compose(
            [
                resized_original,
                gt_display.resize((panel_size, panel_size), Image.NEAREST),
                pred_display.resize((panel_size, panel_size), Image.NEAREST),
                overlay,
            ],
            ["Input", "Ground truth", "Prediction", "Overlay"],
            panel_size,
            footer,
        )
        out_path = os.path.join(args.output_dir, f"preview_{index:02d}_{name}.png")
        canvas.save(out_path)
        print(f"[{index}/{len(samples)}] {out_path} mIoU={miou:.4f} classes={len(classes)}")
        rows.append([os.path.basename(out_path), name, f"{miou:.4f}", len(classes), main_classes])

    csv_path = os.path.join(args.output_dir, "preview_info.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "sample", "image_miou", "predicted_classes", "main_classes"])
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
