import argparse
import csv
import json
import math
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "voxel_cache_32"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
VOXEL_SIZE = 32
IMAGE_SIZE = 64


@dataclass
class TrainConfig:
    name: str
    model_type: str
    use_attention: bool
    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    max_samples: int = 96


MODEL_CONFIGS = [
    TrainConfig(name="model_1_baseline", model_type="baseline", use_attention=False),
    TrainConfig(name="model_2_attention", model_type="attention", use_attention=True),
    TrainConfig(name="model_3_residual_attention", model_type="residual_attention", use_attention=True),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paired_samples(dataset_dir: Path = DATASET_DIR) -> list[tuple[Path, Path]]:
    pngs = {p.stem: p for p in dataset_dir.glob("*.png")}
    stls = {p.stem: p for p in dataset_dir.glob("*.stl")}
    stems = sorted(set(pngs) & set(stls))
    return [(pngs[stem], stls[stem]) for stem in stems]


def read_binary_stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL file is too small: {path}")

    n_triangles = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + n_triangles * 50
    if expected > len(data):
        raise ValueError(f"Unsupported or truncated STL file: {path}")

    triangles = np.empty((n_triangles, 3, 3), dtype=np.float32)
    offset = 84
    for i in range(n_triangles):
        # 12 bytes normal, then 3 vertices x 3 float32, then 2 bytes attr.
        values = struct.unpack_from("<12fH", data, offset)
        triangles[i] = np.asarray(values[3:12], dtype=np.float32).reshape(3, 3)
        offset += 50
    return triangles


def normalize_vertices(triangles: np.ndarray) -> np.ndarray:
    points = triangles.reshape(-1, 3)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    scale = float(np.max(maxs - mins))
    if scale <= 0:
        raise ValueError("Degenerate STL geometry")
    return (triangles - mins) / scale


def rasterize_triangles_to_voxels(triangles: np.ndarray, size: int = VOXEL_SIZE) -> np.ndarray:
    triangles = normalize_vertices(triangles)
    vox = np.zeros((size, size, size), dtype=np.uint8)

    for tri in triangles:
        v0, v1, v2 = tri
        edge = max(np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v0), np.linalg.norm(v2 - v1))
        steps = max(2, min(14, int(math.ceil(edge * size * 1.5))))
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                a = i / steps
                b = j / steps
                c = 1.0 - a - b
                p = a * v0 + b * v1 + c * v2
                idx = np.clip(np.round(p * (size - 1)).astype(np.int32), 0, size - 1)
                vox[idx[2], idx[1], idx[0]] = 1

    # One-voxel thickening stabilizes training on sparse triangle surfaces.
    thick = vox.copy()
    thick[:-1] |= vox[1:]
    thick[1:] |= vox[:-1]
    thick[:, :-1] |= vox[:, 1:]
    thick[:, 1:] |= vox[:, :-1]
    thick[:, :, :-1] |= vox[:, :, 1:]
    thick[:, :, 1:] |= vox[:, :, :-1]
    return thick


def cache_path_for(stl_path: Path) -> Path:
    return CACHE_DIR / f"{stl_path.stem}_vox{VOXEL_SIZE}.npy"


def load_or_create_voxel(stl_path: Path) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for(stl_path)
    if cache_path.exists():
        return np.load(cache_path)
    triangles = read_binary_stl_triangles(stl_path)
    vox = rasterize_triangles_to_voxels(triangles, VOXEL_SIZE)
    np.save(cache_path, vox)
    return vox


def prepare_cache(max_samples: int | None = None) -> dict:
    pairs = paired_samples()
    if max_samples is not None:
        pairs = pairs[:max_samples]
    stats = []
    for _, stl_path in pairs:
        vox = load_or_create_voxel(stl_path)
        stats.append(float(vox.mean()))
    return {
        "dataset_dir": str(DATASET_DIR),
        "num_pairs": len(paired_samples()),
        "cached_pairs": len(pairs),
        "voxel_size": VOXEL_SIZE,
        "image_size": IMAGE_SIZE,
        "mean_occupancy": float(np.mean(stats)) if stats else 0.0,
    }


class ImageVoxelDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int = IMAGE_SIZE):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, stl_path = self.pairs[idx]
        image = Image.open(img_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        img = np.asarray(image, dtype=np.float32) / 255.0
        vox = load_or_create_voxel(stl_path).astype(np.float32)
        return torch.from_numpy(img[None, :, :]), torch.from_numpy(vox[None, :, :, :])


class SpatialSelfAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ConvEncoder2d(nn.Module):
    def __init__(self, use_attention: bool = False, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.c1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.c4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.attn = SpatialSelfAttention2d(256, heads=4) if use_attention else nn.Identity()

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        out = self.attn(x)
        return x + out if self.residual and not isinstance(self.attn, nn.Identity) else out


class VoxelDecoder3d(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, 3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z).view(z.shape[0], 256, 4, 4, 4)
        return self.net(x)


class ImageToVoxelNet(nn.Module):
    def __init__(self, use_attention: bool = False, residual: bool = False):
        super().__init__()
        self.encoder = ConvEncoder2d(use_attention=use_attention, residual=residual)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.decoder = VoxelDecoder3d(256)

    def forward(self, x):
        features = self.encoder(x)
        latent = self.pool(features).flatten(1)
        return self.decoder(latent)


def make_model(model_type: str) -> ImageToVoxelNet:
    if model_type == "baseline":
        return ImageToVoxelNet(use_attention=False, residual=False)
    if model_type == "attention":
        return ImageToVoxelNet(use_attention=True, residual=False)
    if model_type == "residual_attention":
        return ImageToVoxelNet(use_attention=True, residual=True)
    raise ValueError(model_type)


def dice_loss(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3, 4))
    den = probs.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4)) + eps
    return 1 - (num / den).mean()


def combined_loss(logits, targets):
    pos = targets.sum().clamp_min(1.0)
    neg = targets.numel() - pos
    pos_weight = (neg / pos).clamp(1.0, 30.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    return bce + dice_loss(logits, targets)


@torch.no_grad()
def metrics_from_logits(logits, targets, thr: float = 0.35) -> dict:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * targets).sum(dim=(1, 2, 3, 4))
    union = ((pred + targets) > 0).float().sum(dim=(1, 2, 3, 4)).clamp_min(1)
    pred_sum = pred.sum(dim=(1, 2, 3, 4))
    target_sum = targets.sum(dim=(1, 2, 3, 4)).clamp_min(1)
    dice_den = (pred_sum + target_sum).clamp_min(1)
    return {
        "iou": float((inter / union).mean().item()),
        "dice": float(((2 * inter) / dice_den).mean().item()),
        "pred_occupancy": float(pred.mean().item()),
        "target_occupancy": float(targets.mean().item()),
    }


def evaluate(model, loader, device) -> dict:
    model.eval()
    losses = []
    agg = []
    with torch.no_grad():
        for images, voxels in loader:
            images = images.to(device)
            voxels = voxels.to(device)
            logits = model(images)
            losses.append(float(combined_loss(logits, voxels).item()))
            agg.append(metrics_from_logits(logits, voxels))
    keys = agg[0].keys()
    out = {key: float(np.mean([item[key] for item in agg])) for key in keys}
    out["loss"] = float(np.mean(losses))
    return out


def split_loaders(max_samples: int, batch_size: int, seed: int, num_workers: int, pin_memory: bool):
    pairs = paired_samples()[:max_samples]
    ds = ImageVoxelDataset(pairs)
    train_len = max(1, int(0.8 * len(ds)))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, {"total": len(ds), "train": train_len, "val": val_len}


def train_one(config: TrainConfig, device, seed: int, num_workers: int, amp_enabled: bool) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pin_memory = device.type == "cuda"
    train_loader, val_loader, split_info = split_loaders(
        config.max_samples,
        config.batch_size,
        seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model = make_model(config.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for images, voxels in train_loader:
            images = images.to(device, non_blocking=pin_memory)
            voxels = voxels.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = combined_loss(logits, voxels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        val = evaluate(model, val_loader, device)
        train_loss = float(np.mean(train_losses))
        print(f"{config.name}: epoch {epoch}/{config.epochs}, train_loss={train_loss:.4f}, val_iou={val['iou']:.4f}, val_dice={val['dice']:.4f}")
        if best is None or val["loss"] < best["val_loss"]:
            best = {"val_loss": val["loss"], "val_iou": val["iou"], "val_dice": val["dice"], "epoch": epoch}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "split": split_info,
                    "metrics": best,
                    "voxel_size": VOXEL_SIZE,
                    "image_size": IMAGE_SIZE,
                },
                CHECKPOINT_DIR / f"{config.name}.pt",
            )

    return {
        **asdict(config),
        **split_info,
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_iou": best["val_iou"],
        "val_dice": best["val_dice"],
        "checkpoint": str(CHECKPOINT_DIR / f"{config.name}.pt"),
    }


def train_all(args) -> None:
    seed_everything(args.seed)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    cache_info = prepare_cache(args.max_samples)
    (ARTIFACTS_DIR / "dataset_summary.json").write_text(json.dumps(cache_info, indent=2), encoding="utf-8")

    device = resolve_device(args)
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"AMP enabled: {amp_enabled}")
    rows = []
    for base in MODEL_CONFIGS:
        config = TrainConfig(
            name=base.name,
            model_type=base.model_type,
            use_attention=base.use_attention,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_samples=args.max_samples,
        )
        rows.append(
            train_one(
                config,
                device,
                args.seed,
                num_workers=args.num_workers,
                amp_enabled=amp_enabled,
            )
        )

    csv_path = ARTIFACTS_DIR / "training_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (ARTIFACTS_DIR / "training_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved results to {csv_path}")


def inspect_dataset(args) -> None:
    summary = prepare_cache(args.max_samples)
    png_path, stl_path = paired_samples()[0]
    img = Image.open(png_path)
    vox = load_or_create_voxel(stl_path)
    summary.update(
        {
            "first_png": str(png_path),
            "first_stl": str(stl_path),
            "first_image_size": img.size,
            "first_image_mode": img.mode,
            "first_voxel_shape": list(vox.shape),
            "first_voxel_occupancy": float(vox.mean()),
        }
    )
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def resolve_device(args) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if args.device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if args.require_gpu:
        raise RuntimeError("GPU is required, but CUDA is not available")
    return torch.device("cpu")


def build_argparser():
    parser = argparse.ArgumentParser(description="Sem 4 image-to-voxel training pipeline.")
    parser.add_argument("--mode", choices=["inspect", "train"], default="inspect")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.mode == "inspect":
        inspect_dataset(args)
    else:
        train_all(args)


if __name__ == "__main__":
    main()
