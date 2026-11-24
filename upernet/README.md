# VAVAE + UPerNet 语义分割训练指南

本文档面向训练复现，介绍训练所需的**硬件配置**、**环境与依赖要求**、**数据/权重准备**以及**启动训练与断点续训方法**。模型对接方案与实验结果总览请见根目录 [`README.md`](../README.md)。

---

## 1. 硬件配置要求

| 硬件项 | 最低 / 推荐配置 | 说明 |
|---|---|---|
| **GPU（显存）** | 8 GB 以上（推荐 12–24 GB） | `256×256`、batch 4、冻结 encoder + AMP 显存峰值约 **2–3 GB** |
| **系统内存** | 16 GB 以上 | `num_workers=4` 时各工作进程会缓存图像数据 |
| **磁盘空间** | 至少 8 GB 可用空间 | ADE20K 解压后 ~2.3 GB + 预训练权重 ~2.4 GB + 训练产出（`last.pth` 610 MB、`best.pth` 381 MB） |
| **多卡支持** | 单卡训练 | 当前脚本为单卡实现，未启用 DDP |

**训练耗时参考**：在单卡 GPU 上以 `256×256`、batch 4、冻结 encoder 跑满 20 轮（每轮 20,210 张训练图 + 2,000 张验证图），实测每轮约 **14.1 分钟**，20 轮总计约 **4.7 小时**。时间开销主要集中在 encoder 前向计算。

---

## 2. 环境与依赖要求

训练仅需以下依赖，**不需要 `transformers` 或 `mmsegmentation`**（UPerNet head 为本仓库独立实现）：

- **Python**：3.9+（本地验证版本 3.12）
- **PyTorch / torchvision**：CUDA 版 `torch>=2.0`、`torchvision>=0.15`（需与本机 CUDA 驱动版本匹配；`LightningDiT/tokenizer/autoencoder.py` 顶层导入了 `torchvision`）
- **其他依赖**（见 `upernet/requirements.txt`）：`numpy>=1.24`、`pillow>=9.0`、`requests>=2.28`、`tqdm>=4.65`

```bash
pip install -r upernet/requirements.txt
```

---

## 3. 训练前文件准备

由于数据集与权重文件未上传至 Git 仓库，启动训练前请先手动下载并按如下结构放置：

```text
vavae(upernet)/
├── vavae-imagenet256-f16d32-dinov2.pt             # 必需，2,148,559,914 字节
├── data/
│   ├── ADEChallengeData2016/                      # 必需，解压自 ADEChallengeData2016.zip
│   │   ├── images/training                        # 20,210 张 jpg
│   │   ├── images/validation                      # 2,000 张 jpg
│   │   ├── annotations/training                   # 20,210 张 png
│   │   └── annotations/validation                 # 2,000 张 png
│   └── upernet-swin-tiny-pytorch_model.bin        # 可选，240,155,657 字节
└── LightningDiT/
    └── tokenizer/autoencoder.py                   # 随仓库提供，切勿移动或改名
```

| 本地路径 | 说明 | 资源链接 |
|---|---|---|
| `vavae-imagenet256-f16d32-dinov2.pt` | VA-VAE encoder 预训练权重（必需） | [HuggingFace](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2) |
| `data/ADEChallengeData2016/` | ADE20K 数据集（必需，解压至此目录） | [ADEChallengeData2016.zip](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) |
| `data/upernet-swin-tiny-pytorch_model.bin` | UPerNet 预训练头（可选，不传 `--hf-head` 则随机初始化） | [HuggingFace](https://huggingface.co/openmmlab/upernet-swin-tiny)（下载 `pytorch_model.bin` 后重命名） |

---

## 4. 启动训练方法

以下所有命令均在项目根目录 `vavae(upernet)/` 下执行。

### 4.1 小样本环境冒烟测试（CPU 可跑，约 1 分钟）

```bash
python upernet/train.py --freeze-backbone --subset 8 --epochs 1 --batch-size 2 --num-workers 0 --device cpu
```

若能正常打印 loss、mIoU 并在 `checkpoints/` 下生成权重文件，说明数据路径与依赖环境均配置正确。

### 4.2 正式训练：冻结 VA-VAE Encoder，仅训练 UPerNet Head

```bash
python upernet/train.py \
  --hf-head data/upernet-swin-tiny-pytorch_model.bin \
  --freeze-backbone \
  --batch-size 4 \
  --epochs 20 \
  --lr 1e-3 \
  --image-size 256 \
  --num-workers 4 \
  --output-dir checkpoints
```

- `--freeze-backbone`：冻结 VA-VAE encoder（69.84M），仅训练 head（29.97M），大幅节省显存与时间。
- `--lr 1e-3`：head 从部分随机初始化起步，建议使用 `1e-3`（默认 `1e-4` 收敛较慢）。
- 若在 Windows 上遇到 DataLoader 多进程报错或卡顿，可将 `--num-workers` 设为 `0`。


### 4.3 单独评测与生成预览图

```bash
# 验证集评测（2000 张验证图，支持使用从 GitHub Release 下载的 best.pth）
python upernet/train.py --resume checkpoints/best.pth --eval-only --batch-size 4

# 生成四格分割预览图（输入 / 真值 / 预测 / 叠加）
python upernet/visualize.py --ckpt checkpoints/best.pth --output-dir docs/images \
  --images ADE_val_00000127 ADE_val_00000526 ADE_val_00000291 ADE_val_00000345
```