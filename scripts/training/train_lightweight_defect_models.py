#!/usr/bin/env python3
"""Train and evaluate class-conditioned MobileNetV3 and EfficientNet-B0."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from conditional_lightweight_models import SUPPORTED_BACKBONES, build_model
from conditional_resnet import (
    Config,
    best_f1_threshold,
    build_transforms,
    predict_loader,
    save_reports,
    seed_everything,
    train_model,
)


RIDAC_DIR = Path(__file__).resolve().parent
BALANCED_ROOT = (
    RIDAC_DIR
    / "object_defect_dataset"
    / "balanced_augmented_full_r50_cap8_seed42"
)
REFERENCE_ROOT = BALANCED_ROOT / "resnet_training_results"
SPLIT_MANIFEST = REFERENCE_ROOT / "balanced_training_split_manifest.csv"
OUTPUT_ROOT = BALANCED_ROOT / "lightweight_model_results"

MODEL_SETTINGS = {
    "mobilenet_v3_large": {
        "batch_size": 160,
        "learning_rate": 3e-4,
        "backbone_lr_multiplier": 0.20,
    },
    "efficientnet_b0": {
        "batch_size": 128,
        "learning_rate": 3e-4,
        "backbone_lr_multiplier": 0.20,
    },
}



def _decode_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


class CachedDefectDataset(Dataset):
    # Decoded RGB crops are shared by forked loader workers.
    def __init__(self, table: pd.DataFrame, transform, workers: int = 16):
        self.table = table.reset_index(drop=True)
        self.transform = transform
        paths = self.table["crop_path"].astype(str).tolist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            self.images = list(tqdm(
                pool.map(_decode_rgb, paths), total=len(paths),
                desc="Caching decoded crops", unit="image",
            ))

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        image = Image.fromarray(self.images[index], mode="RGB")
        return (
            self.transform(image),
            torch.tensor(int(row["class_id"]), dtype=torch.long),
            torch.tensor(float(row["target"]), dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


def cache_split_datasets(table: pd.DataFrame, image_size: int):
    train_transform, eval_transform = build_transforms(image_size)
    subsets = {
        split: table.loc[table["split"].eq(split)].copy()
        for split in ("train", "val", "test")
    }
    return {
        split: CachedDefectDataset(
            subset, train_transform if split == "train" else eval_transform
        )
        for split, subset in subsets.items()
    }


def cached_loaders(datasets: dict[str, Dataset], config: Config):
    train_table = datasets["train"].table
    group_counts = train_table.groupby(["source_class", "target"]).size()
    sample_weights = train_table.apply(
        lambda row: 1.0 / group_counts.loc[(row["source_class"], row["target"])],
        axis=1,
    ).to_numpy()
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights), replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        common["prefetch_factor"] = 3
    return {
        "train": DataLoader(datasets["train"], sampler=sampler, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def load_reference_split() -> tuple[pd.DataFrame, dict[str, int]]:
    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing {SPLIT_MANIFEST}. Run the balanced ResNet workflow first."
        )
    table = pd.read_csv(SPLIT_MANIFEST, low_memory=False)
    required = {
        "source_image",
        "source_class",
        "assigned_label",
        "crop_path",
        "class_id",
        "target",
        "split",
    }
    missing = required - set(table)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    exists = table["crop_path"].map(lambda value: Path(str(value)).is_file())
    if not exists.all():
        examples = table.loc[~exists, "crop_path"].head().tolist()
        raise FileNotFoundError(
            f"{int((~exists).sum())} crop files are missing. Examples: {examples}"
        )
    class_map = (
        table[["source_class", "class_id"]]
        .drop_duplicates()
        .sort_values("class_id")
    )
    if class_map["source_class"].duplicated().any():
        raise ValueError("A source class has multiple class IDs.")
    class_to_idx = {
        str(row.source_class): int(row.class_id)
        for row in class_map.itertuples(index=False)
    }
    return table, class_to_idx


def make_config(backbone: str) -> Config:
    settings = MODEL_SETTINGS[backbone]
    return Config(
        ridac_dir=RIDAC_DIR,
        output_dir_name=str(
            (BALANCED_ROOT / "lightweight_model_results" / backbone).relative_to(
                RIDAC_DIR / "object_defect_dataset"
            )
        ),
        backbone="resnet18",  # Data/training helper compatibility only.
        pretrained=True,
        image_size=224,
        batch_size=settings["batch_size"],
        epochs=20,
        early_stopping_patience=5,
        learning_rate=settings["learning_rate"],
        backbone_lr_multiplier=settings["backbone_lr_multiplier"],
        weight_decay=1e-4,
        dropout=0.30,
        embedding_dim=128,
        label_smoothing=0.02,
        num_workers=min(16, os.cpu_count() or 2),
        seed=42,
        compile_model=False,
    )


def save_checkpoint(
    model,
    backbone: str,
    config: Config,
    class_to_idx: dict[str, int],
    threshold: float,
    best_validation_pr_auc: float,
    destination: Path,
) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "architecture": "class_conditioned_binary_classifier",
        "backbone": backbone,
        "class_to_idx": class_to_idx,
        "threshold": float(threshold),
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "backbone": backbone,
        },
        "best_validation_pr_auc": float(best_validation_pr_auc),
        "split_manifest": str(SPLIT_MANIFEST.resolve()),
    }
    torch.save(checkpoint, destination)


def train_one(
    backbone: str,
    table: pd.DataFrame,
    class_to_idx: dict[str, int],
    datasets: dict[str, Dataset],
    device: torch.device,
) -> dict[str, float | str]:
    output_dir = OUTPUT_ROOT / backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(backbone)
    seed_everything(config.seed)
    loaders = cached_loaders(datasets, config)
    model = build_model(
        backbone=backbone,
        num_classes=len(class_to_idx),
        device=device,
        pretrained=True,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    )
    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"\n=== {backbone} ===\n"
        f"Device: {device}; parameters: {parameters:,}; "
        f"batch size: {config.batch_size}"
    )
    model, history, best_validation_pr_auc = train_model(
        model, loaders, config, device
    )

    validation_predictions = predict_loader(model, loaders["val"], device)
    threshold = best_f1_threshold(
        validation_predictions["target"].to_numpy(),
        validation_predictions["probability_defective"].to_numpy(),
    )
    validation_predictions["predicted_label"] = np.where(
        validation_predictions["probability_defective"].ge(threshold),
        "defective",
        "normal",
    )
    validation_predictions.to_csv(
        output_dir / "validation_predictions.csv", index=False
    )

    test_predictions = predict_loader(model, loaders["test"], device)
    overall, per_class = save_reports(
        test_predictions, history, threshold, output_dir
    )
    checkpoint_path = output_dir / f"best_conditional_{backbone}.pt"
    save_checkpoint(
        model,
        backbone,
        config,
        class_to_idx,
        threshold,
        best_validation_pr_auc,
        checkpoint_path,
    )
    with (output_dir / "class_to_idx.json").open("w") as handle:
        json.dump(class_to_idx, handle, indent=2)
    summary = overall.loc["overall"].to_dict()
    summary.update(
        {
            "model": backbone,
            "parameters": parameters,
            "best_validation_pr_auc": best_validation_pr_auc,
            "epochs_completed": int(history["epoch"].max()),
            "checkpoint": str(checkpoint_path.resolve()),
        }
    )
    del model, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def reference_resnet_row() -> dict[str, float | str] | None:
    metrics_path = REFERENCE_ROOT / "overall_metrics.csv"
    checkpoint_path = REFERENCE_ROOT / "best_conditional_resnet.pt"
    history_path = REFERENCE_ROOT / "training_history.csv"
    if not metrics_path.is_file():
        return None
    row = pd.read_csv(metrics_path, index_col=0).loc["overall"].to_dict()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    history = pd.read_csv(history_path)
    parameters = sum(
        value.numel()
        for key, value in checkpoint["model_state_dict"].items()
        if key.endswith("weight") or key.endswith("bias")
    )
    row.update(
        {
            "model": "resnet18",
            "parameters": parameters,
            "best_validation_pr_auc": checkpoint["best_validation_pr_auc"],
            "epochs_completed": int(history["epoch"].max()),
            "checkpoint": str(checkpoint_path.resolve()),
        }
    )
    return row


def run(backbones: list[str]) -> pd.DataFrame:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table, class_to_idx = load_reference_split()
    print(
        "Using the exact ResNet split:",
        table.groupby(["split", "assigned_label"]).size().to_dict(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Preloading decoded crops once to eliminate per-epoch disk stalls...")
    datasets = cache_split_datasets(table, image_size=224)
    summaries = []
    reference = reference_resnet_row()
    if reference is not None:
        summaries.append(reference)
    for backbone in backbones:
        summaries.append(train_one(backbone, table, class_to_idx, datasets, device))
    comparison = pd.DataFrame(summaries)
    preferred = [
        "model",
        "parameters",
        "epochs_completed",
        "best_validation_pr_auc",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "negative_predictive_value",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "tn",
        "fp",
        "fn",
        "tp",
        "checkpoint",
    ]
    comparison = comparison[
        [column for column in preferred if column in comparison]
    ]
    comparison.to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)
    print("\nModel comparison")
    print(comparison.round(6).to_string(index=False))
    return comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_BACKBONES,
        default=list(SUPPORTED_BACKBONES),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.models)

