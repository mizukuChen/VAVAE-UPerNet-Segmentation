"""在 ADE20K 上微调 VA-VAE + UPerNet 语义分割模型。

CPU 冒烟测试（16 张图，1 轮）：
    python upernet/train.py --subset 16 --epochs 1 --batch-size 2 --device cpu

冻结 encoder，仅训练 UPerNet head：
    python upernet/train.py --hf-head data/upernet-swin-tiny-pytorch_model.bin \
        --freeze-backbone --epochs 20 --lr 1e-3 --batch-size 4

在验证集上单独评测已保存权重：
    python upernet/train.py --resume checkpoints/best.pth --eval-only
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vavae_upernet import VAVAEUPerNet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORE_INDEX = 255


class ADE20KSegmentation(Dataset):
    """ADEChallengeData2016 数据集：images/<split>/*.jpg + annotations/<split>/*.png。

    标注 PNG 中 1..150 为语义类别、0 为未标注像素；加载时映射为 0..149，
    并将未标注像素置为 IGNORE_INDEX (255)，在计算 loss 和 mIoU 时忽略。
    """

    def __init__(self, root, split, image_size=256, augment=False, subset=None):
        self.image_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")

        ids = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(self.image_dir)
            if name.lower().endswith(".jpg")
        )
        self.ids = [
            name for name in ids if os.path.isfile(os.path.join(self.ann_dir, name + ".png"))
        ]
        if subset is not None:
            self.ids = self.ids[:subset]
        if not self.ids:
            raise RuntimeError(f"No image/mask pairs found under {self.image_dir}")

        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        image = Image.open(os.path.join(self.image_dir, name + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.ann_dir, name + ".png"))

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - 0.5) / 0.5  # VA-VAE 要求输入归一化到 [-1, 1]
        image_tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())

        labels = np.asarray(mask, dtype=np.int64) - 1
        labels[labels < 0] = IGNORE_INDEX
        return image_tensor, torch.from_numpy(labels.copy())


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, label):
        valid = label != IGNORE_INDEX
        pred = pred[valid].astype(np.int64)
        label = label[valid].astype(np.int64)
        index = self.num_classes * label + pred
        self.mat += np.bincount(index, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def mean_iou(self):
        diagonal = np.diag(self.mat).astype(np.float64)
        union = self.mat.sum(axis=1) + self.mat.sum(axis=0) - diagonal
        iou = diagonal / np.maximum(union, 1.0)
        present = self.mat.sum(axis=1) > 0
        return float(iou[present].mean()) if present.any() else 0.0

    def pixel_accuracy(self):
        total = self.mat.sum()
        return float(np.diag(self.mat).sum() / total) if total else 0.0


class Logger:
    """同时向标准输出打印并将相同内容追加写入日志文件。"""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")

    def __call__(self, message):
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune UPerNet on ADE20K with a VAVAE encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 路径配置
    parser.add_argument("--data-root", default=os.path.join(ROOT, "data", "ADEChallengeData2016"))
    parser.add_argument("--vavae-ckpt", default=os.path.join(ROOT, "vavae-imagenet256-f16d32-dinov2.pt"))
    parser.add_argument("--hf-head", default=None,
                        help="openmmlab/upernet-* pytorch_model.bin used to initialize the head")
    parser.add_argument("--resume", default=None, help="checkpoint saved by this script")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "checkpoints"))
    # 数据配置
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset", type=int, default=None,
                        help="only use the first N images of each split (debugging)")
    # 优化器与学习率配置
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for the head")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1,
                        help="encoder lr = lr * scale, only used without --freeze-backbone")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument("--scheduler", choices=["poly", "cosine", "none"], default="poly")
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="freeze the VAVAE encoder and train the UPerNet head only")
    # 运行时配置
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="mixed precision on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true", help="evaluate --resume and exit")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    minutes = seconds / 60.0
    return f"{minutes / 60:.1f}h" if minutes >= 60 else f"{minutes:.1f}m"


def amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (TypeError, RuntimeError):
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_loaders(args):
    train_set = ADE20KSegmentation(
        args.data_root, "training", args.image_size, augment=True, subset=args.subset
    )
    val_set = ADE20KSegmentation(
        args.data_root, "validation", args.image_size, augment=False, subset=args.subset
    )
    pin_memory = args.device.startswith("cuda")
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def build_model(args, device):
    model = VAVAEUPerNet(args.vavae_ckpt, num_classes=150)
    if args.hf_head:
        model.load_hf_head(args.hf_head)
    total = sum(p.numel() for p in model.parameters())
    if args.eval_only:
        print(f"evaluation only; total params: {total / 1e6:.2f}M")
        return model.to(device)
    if args.freeze_backbone:
        model.freeze_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"frozen VAVAE encoder; trainable params: {trainable / 1e6:.2f}M of {total / 1e6:.2f}M")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"training full model; trainable params: {trainable / 1e6:.2f}M")
    return model.to(device)


def _split_decay(named_params):
    decay, no_decay = [], []
    for name, param in named_params:
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def configure_optimizer(model, args):
    """配置 AdamW 优化器，对 norm 层和 bias 不施加权重衰减。"""
    if args.freeze_backbone:
        scopes = [([(n, p) for n, p in model.named_parameters() if p.requires_grad], args.lr)]
    else:
        head = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
        encoder = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
        scopes = [(head, args.lr), (encoder, args.lr * args.backbone_lr_scale)]

    groups = []
    for named_params, lr in scopes:
        decay, no_decay = _split_decay(named_params)
        groups.append({"params": decay, "lr": lr, "weight_decay": args.weight_decay})
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr)


def build_scheduler(optimizer, args, total_steps):
    if args.scheduler == "none":
        return None
    warmup = max(1, args.warmup_iters)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        if args.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, (1.0 - progress) ** args.poly_power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate(model, loader, device, num_classes=150, amp=False):
    model.eval()
    confusion = ConfusionMatrix(num_classes)
    loss_sum, pixel_count = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_context(device, amp):
                logits = model(images)
                loss = F.cross_entropy(
                    logits.float(), labels, ignore_index=IGNORE_INDEX, reduction="sum"
                )
            loss_sum += loss.item()
            pixel_count += int((labels != IGNORE_INDEX).sum().item())
            confusion.update(logits.argmax(1).cpu().numpy(), labels.cpu().numpy())
    return confusion.mean_iou(), confusion.pixel_accuracy(), loss_sum / max(pixel_count, 1)


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    epoch=0, best_miou=-1.0, args=None):
    """保存训练状态。

    `last.pth` 保留优化器/调度器/scaler 状态以便断点续训；
    `best.pth` 仅保存模型参数以减小文件体积（约 381 MB vs 610 MB）。
    """
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "args": vars(args) if args is not None else {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_miou", -1.0)


def main():
    args = parse_args()
    if args.image_size % 32 != 0:
        raise SystemExit("--image-size must be a multiple of 32 (features go down to 1/32)")
    if args.eval_only and not args.resume:
        raise SystemExit("--eval-only requires --resume <checkpoint>")
    if not args.eval_only and args.batch_size < 2:
        raise SystemExit(
            "training needs --batch-size >= 2: the PPM branch pools to 1x1 and "
            "BatchNorm cannot compute statistics from a single sample"
        )
    seed_everything(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    log = Logger(os.path.join(args.output_dir, "train.log"))
    log(f"device={device} amp={args.amp and device.type == 'cuda'}")
    log("args=" + json.dumps(vars(args), ensure_ascii=False, sort_keys=True))

    train_loader, val_loader = build_loaders(args)
    log(f"train images={len(train_loader.dataset)} val images={len(val_loader.dataset)}")
    log(f"steps per epoch={len(train_loader)} batch_size={args.batch_size} "
        f"grad_accum={args.grad_accum}")

    model = build_model(args, device)
    optimizer = configure_optimizer(model, args)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    scheduler = build_scheduler(optimizer, args, args.epochs * steps_per_epoch)
    use_amp = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    start_epoch, best_miou = 1, -1.0
    if args.resume:
        last_epoch, best_miou = load_checkpoint(
            args.resume, model, None if args.eval_only else optimizer,
            None if args.eval_only else scheduler,
            None if args.eval_only else scaler,
        )
        start_epoch = last_epoch + 1
        log(f"resumed {args.resume} (epoch={last_epoch}, best_mIoU={best_miou:.4f})")

    if args.eval_only:
        miou, pixel_acc, val_loss = evaluate(model, val_loader, device, amp=use_amp)
        log(f"eval only | val_loss={val_loss:.4f} mIoU={miou:.4f} pixel_acc={pixel_acc:.4f}")
        log.close()
        return

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    if not os.path.exists(metrics_path) or start_epoch == 1:
        with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["epoch", "lr", "train_loss", "val_loss", "miou", "pixel_acc", "seconds"]
            )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp_context(device, use_amp):
                loss = F.cross_entropy(model(images), labels, ignore_index=IGNORE_INDEX)
            scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

            should_step = step % args.grad_accum == 0 or step == len(train_loader)
            if should_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            if step % args.print_freq == 0 or step == len(train_loader):
                elapsed = time.time() - epoch_start
                eta = elapsed / step * (len(train_loader) - step)
                lr = optimizer.param_groups[0]["lr"]
                log(f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={running_loss / max(seen, 1):.4f} lr={lr:.2e} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}")

        miou, pixel_acc, val_loss = evaluate(model, val_loader, device, amp=use_amp)
        seconds = time.time() - epoch_start
        log(f"epoch {epoch} done in {format_duration(seconds)} | train_loss={running_loss / max(seen, 1):.4f} "
            f"val_loss={val_loss:.4f} mIoU={miou:.4f} pixel_acc={pixel_acc:.4f}")

        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch, f"{optimizer.param_groups[0]['lr']:.6e}",
                 f"{running_loss / max(seen, 1):.6f}", f"{val_loss:.6f}",
                 f"{miou:.6f}", f"{pixel_acc:.6f}", f"{seconds:.1f}"]
            )

        save_checkpoint(
            os.path.join(args.output_dir, "last.pth"),
            model, optimizer, scheduler, scaler, epoch, max(best_miou, miou), args,
        )
        if miou > best_miou:
            best_miou = miou
            best_path = os.path.join(args.output_dir, "best.pth")
            save_checkpoint(best_path, model, epoch=epoch, best_miou=best_miou, args=args)
            log(f"saved {best_path} (mIoU={miou:.4f})")

    log(f"finished. best mIoU={best_miou:.4f}")
    log.close()


if __name__ == "__main__":
    main()
