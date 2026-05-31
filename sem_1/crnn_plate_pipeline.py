import argparse
import csv
import glob
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = 0
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(ALPHABET)}
IDX2CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1

ROOT = Path(__file__).resolve().parent
OCR_ROOT = ROOT / "autoriaNumberplateOcrRu"
DETECTOR_PATH = ROOT / "data_car_number_labels" / "runs" / "detect" / "car_number_1003263" / "weights" / "best.pt"
ARTIFACTS_DIR = ROOT / "artifacts"


@dataclass
class ExperimentConfig:
    name: str
    img_height: int = 32
    img_width: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    lr: float = 1e-3
    augment: bool = False


EXPERIMENTS = [
    ExperimentConfig(name="baseline_32x128"),
    ExperimentConfig(name="wide_input_32x160", img_width=160),
    ExperimentConfig(name="augmented_32x128", augment=True),
]


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(config: ExperimentConfig, train: bool) -> T.Compose:
    transforms = [T.Grayscale(num_output_channels=1)]
    if train and config.augment:
        transforms.extend(
            [
                T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.5),
                T.RandomAffine(degrees=3, translate=(0.03, 0.05), shear=2),
            ]
        )
    transforms.extend(
        [
            T.Resize((config.img_height, config.img_width)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )
    return T.Compose(transforms)


class NumberplateOCRDataset(Dataset):
    def __init__(self, root: Path, split: str, transforms=None, max_samples: int | None = None):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.ann_paths = sorted(glob.glob(str(self.root / split / "ann" / "*.json")))
        if max_samples is not None:
            self.ann_paths = self.ann_paths[:max_samples]

    def __len__(self) -> int:
        return len(self.ann_paths)

    def encode_text(self, text: str) -> list[int]:
        return [CHAR2IDX[char] for char in str(text).upper() if char in CHAR2IDX]

    def __getitem__(self, idx: int):
        ann_path = Path(self.ann_paths[idx])
        with ann_path.open("r", encoding="utf-8") as file:
            ann = json.load(file)

        text = str(ann["description"]).upper()
        img_path = self.root / self.split / "img" / ann["name"]
        if not img_path.exists():
            matches = sorted((self.root / self.split / "img").glob(f"{ann['name']}.*"))
            if not matches:
                raise FileNotFoundError(img_path)
            img_path = matches[0]
        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)

        encoded = torch.tensor(self.encode_text(text), dtype=torch.long)
        return image, encoded, text


def collate_fn(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat(labels)
    return images, targets, target_lengths, list(texts)


class CNNBackbone(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
        )

    def forward(self, x):
        return self.net(x)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.backbone = CNNBackbone(1)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features).squeeze(2)
        features = features.permute(2, 0, 1)
        features, _ = self.rnn(features)
        return self.classifier(features).log_softmax(2)


def decode_sequence(sequence: torch.Tensor) -> str:
    values = sequence.detach().cpu().numpy().tolist()
    previous = None
    output = []
    for value in values:
        if value != BLANK_IDX and value != previous and value in IDX2CHAR:
            output.append(IDX2CHAR[value])
        previous = value
    return "".join(output)


def edit_distance(left: str, right: str) -> int:
    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        cur = [i]
        for j, right_char in enumerate(right, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (left_char != right_char),
                )
            )
        prev = cur
    return prev[-1]


def train_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for images, targets, target_lengths, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(images)
        time_steps, batch_size, _ = logits.size()
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)

        loss = criterion(logits, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device: torch.device, max_batches: int | None = None) -> dict:
    model.eval()
    total_seq = 0
    exact_seq = 0
    total_edits = 0
    total_chars = 0
    samples = []

    for batch_idx, (images, _, _, texts) in enumerate(loader):
        images = images.to(device)
        logits = model(images)
        predictions = logits.permute(1, 0, 2).argmax(2)
        for idx in range(images.size(0)):
            pred = decode_sequence(predictions[idx])
            gt = str(texts[idx]).upper()
            total_seq += 1
            exact_seq += int(pred == gt)
            total_edits += edit_distance(gt, pred)
            total_chars += max(1, len(gt))
            if len(samples) < 20:
                samples.append({"gt": gt, "pred": pred})
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    return {
        "exact_accuracy": exact_seq / max(1, total_seq),
        "cer": total_edits / max(1, total_chars),
        "samples": samples,
        "num_samples": total_seq,
    }


def make_loaders(config: ExperimentConfig, batch_size: int, max_train_samples: int | None, max_test_samples: int | None):
    train_dataset = NumberplateOCRDataset(
        OCR_ROOT,
        "train",
        transforms=build_transforms(config, train=True),
        max_samples=max_train_samples,
    )
    test_dataset = NumberplateOCRDataset(
        OCR_ROOT,
        "test",
        transforms=build_transforms(config, train=False),
        max_samples=max_test_samples,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return train_loader, test_loader


def run_training(args) -> None:
    seed_everything(args.seed)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows = []
    best = None

    for config in EXPERIMENTS:
        train_loader, test_loader = make_loaders(config, args.batch_size, args.max_train_samples, args.max_test_samples)
        model = CRNN(VOCAB_SIZE, hidden_size=config.hidden_size, num_layers=config.num_layers).to(device)
        criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        losses = []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            losses.append(loss)
            print(f"{config.name}: epoch {epoch}/{args.epochs}, loss={loss:.4f}")

        metrics = evaluate(model, test_loader, device, max_batches=args.eval_batches)
        row = {
            **asdict(config),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "final_train_loss": losses[-1] if losses else None,
            "exact_accuracy": metrics["exact_accuracy"],
            "cer": metrics["cer"],
            "num_eval_samples": metrics["num_samples"],
        }
        rows.append(row)
        print(f"{config.name}: CER={metrics['cer']:.4f}, exact_accuracy={metrics['exact_accuracy']:.4f}")

        if best is None or metrics["cer"] < best["metrics"]["cer"]:
            best = {"config": config, "metrics": metrics, "state_dict": model.state_dict(), "row": row}

    csv_path = ARTIFACTS_DIR / "crnn_experiments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    alphabet_path = ARTIFACTS_DIR / "alphabet.json"
    alphabet_path.write_text(json.dumps({"alphabet": ALPHABET, "blank_idx": BLANK_IDX}, indent=2), encoding="utf-8")

    checkpoint_path = ARTIFACTS_DIR / "crnn_final.pt"
    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "config": asdict(best["config"]),
            "metrics": best["metrics"],
            "alphabet": ALPHABET,
            "blank_idx": BLANK_IDX,
        },
        checkpoint_path,
    )

    summary_path = ARTIFACTS_DIR / "crnn_final_summary.json"
    summary = {"best_config": asdict(best["config"]), "best_metrics": best["metrics"], "experiments": rows}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved experiments: {csv_path}")
    print(f"Saved alphabet: {alphabet_path}")
    print(f"Saved checkpoint: {checkpoint_path}")


def load_crnn(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ExperimentConfig(**checkpoint["config"])
    model = CRNN(VOCAB_SIZE, hidden_size=config.hidden_size, num_layers=config.num_layers).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def recognize_crop(model: CRNN, config: ExperimentConfig, crop_rgb: np.ndarray, device: torch.device) -> str:
    transform = build_transforms(config, train=False)
    image = Image.fromarray(crop_rgb)
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    prediction = logits.permute(1, 0, 2).argmax(2)[0]
    return decode_sequence(prediction)


def run_pipeline(args) -> None:
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, config, checkpoint = load_crnn(Path(args.crnn_checkpoint), device)
    detector = YOLO(args.detector)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(args.image)

    result = detector.predict(args.image, conf=args.conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("Detector did not find a license plate")

    boxes = result.boxes
    confs = boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].detach().cpu().numpy().tolist())
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    crop_bgr = image_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    prediction = recognize_crop(model, config, crop_rgb, device)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    crop_path = ARTIFACTS_DIR / "pipeline_last_crop.png"
    cv2.imwrite(str(crop_path), crop_bgr)
    output = {
        "image": str(Path(args.image).resolve()),
        "detector": str(Path(args.detector).resolve()),
        "crnn_checkpoint": str(Path(args.crnn_checkpoint).resolve()),
        "bbox_xyxy": [x1, y1, x2, y2],
        "detector_confidence": float(confs[best_idx]),
        "recognized_plate": prediction,
        "crop_path": str(crop_path),
        "crnn_metrics": checkpoint.get("metrics", {}),
    }
    output_path = ARTIFACTS_DIR / "pipeline_last_result.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def run_smoke(args) -> None:
    config = EXPERIMENTS[0]
    train_loader, test_loader = make_loaders(config, batch_size=4, max_train_samples=8, max_test_samples=8)
    images, targets, target_lengths, texts = next(iter(train_loader))
    model = CRNN(VOCAB_SIZE, hidden_size=config.hidden_size, num_layers=config.num_layers)
    logits = model(images)
    print(
        json.dumps(
            {
                "train_batch_shape": list(images.shape),
                "logits_shape": list(logits.shape),
                "target_lengths": target_lengths.tolist(),
                "sample_texts": texts[:3],
                "test_batches": len(test_loader),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CRNN training and end-to-end license plate pipeline.")
    parser.add_argument("--mode", choices=["train", "pipeline", "smoke"], default="smoke")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-test-samples", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--detector", default=str(DETECTOR_PATH))
    parser.add_argument("--crnn-checkpoint", default=str(ARTIFACTS_DIR / "crnn_final.pt"))
    parser.add_argument("--image", default=str(ROOT / "data_car_number_labels" / "images" / "test" / "6.jpg"))
    parser.add_argument("--conf", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "pipeline":
        run_pipeline(args)
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()
