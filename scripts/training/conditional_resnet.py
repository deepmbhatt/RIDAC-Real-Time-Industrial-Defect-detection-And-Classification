"""Class-conditioned ResNet training for normal/defective object crops.

The model is shared across every object category. A known object-class label is
embedded and used to create spatial attention and feature-wise modulation, so
the same backbone can focus on class-specific regions before predicting the
binary defect label.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)
from tqdm.auto import tqdm


@dataclass
class Config:
    ridac_dir: Path = field(default_factory=lambda: find_ridac_dir())
    output_dir_name: str = "conditional_resnet_results"
    backbone: str = "resnet18"  # resnet18, resnet34, or resnet50
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 64
    epochs: int = 30
    early_stopping_patience: int = 7
    learning_rate: float = 3e-4
    backbone_lr_multiplier: float = 0.20
    weight_decay: float = 1e-4
    dropout: float = 0.30
    embedding_dim: int = 128
    label_smoothing: float = 0.02
    num_workers: int = min(8, os.cpu_count() or 2)
    seed: int = 42
    test_folds: int = 5  # One fold is test: 20%.
    val_folds: int = 5  # One fold of train+val is val: 16% overall.
    compile_model: bool = False

    @property
    def dataset_root(self) -> Path:
        return self.ridac_dir / "object_defect_dataset"

    @property
    def output_dir(self) -> Path:
        return self.dataset_root / self.output_dir_name


def find_ridac_dir() -> Path:
    here = Path.cwd().resolve()
    candidates = [here, here / "misc" / "ridac"]
    candidates.extend(parent / "misc" / "ridac" for parent in here.parents)
    for candidate in dict.fromkeys(candidates):
        if (candidate / "extracted_dataset").is_dir():
            return candidate
    raise FileNotFoundError("Could not find misc/ridac/extracted_dataset.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _resolve_crop_path(row: pd.Series, dataset_root: Path) -> Path:
    saved = Path(str(row["saved_path"]))
    if saved.is_file():
        return saved.resolve()
    fallback = dataset_root / "binary_dataset" / row["assigned_label"] / saved.name
    return fallback.resolve()


def load_metadata(config: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the full table when available, otherwise combine completed phases."""
    full_csv = config.dataset_root / "full_results.csv"
    if full_csv.is_file():
        table = pd.read_csv(full_csv)
        source_tables = [full_csv.name]
    else:
        candidates = [
            config.dataset_root / "sample_results.csv",
            config.dataset_root / "remaining_results.csv",
        ]
        available = [path for path in candidates if path.is_file()]
        if not available:
            raise FileNotFoundError(
                f"No result CSV found in {config.dataset_root}. Run the segmentation notebook first."
            )
        table = pd.concat((pd.read_csv(path) for path in available), ignore_index=True)
        source_tables = [path.name for path in available]

    required = {
        "source_image", "source_class", "status", "assigned_label", "saved_path"
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Result table is missing columns: {sorted(missing)}")

    table = table.loc[
        table["status"].eq("ok")
        & table["assigned_label"].isin(["normal", "defective"])
        & table["saved_path"].notna()
    ].copy()
    table["crop_path"] = table.apply(
        _resolve_crop_path, axis=1, dataset_root=config.dataset_root
    ).astype(str)
    exists = table["crop_path"].map(lambda value: Path(value).is_file())
    if not exists.all():
        missing_examples = table.loc[~exists, "crop_path"].head(5).tolist()
        raise FileNotFoundError(
            f"{int((~exists).sum())} crop files are missing. Examples: {missing_examples}"
        )

    table = table.drop_duplicates(subset=["crop_path"]).reset_index(drop=True)
    table["target"] = table["assigned_label"].map({"normal": 0, "defective": 1})
    class_names = sorted(table["source_class"].unique())
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    table["class_id"] = table["source_class"].map(class_to_idx)
    table["stratum"] = table["source_class"] + "__" + table["assigned_label"]
    print(f"Loaded {len(table):,} crops from {', '.join(source_tables)}")
    return table, class_to_idx


def make_grouped_splits(table: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Stratify class/label combinations while keeping a source image in one split."""
    outer = StratifiedGroupKFold(
        n_splits=config.test_folds, shuffle=True, random_state=config.seed
    )
    train_val_idx, test_idx = next(
        outer.split(table, table["stratum"], groups=table["source_image"])
    )
    train_val = table.iloc[train_val_idx]
    inner = StratifiedGroupKFold(
        n_splits=config.val_folds, shuffle=True, random_state=config.seed + 1
    )
    train_rel, val_rel = next(
        inner.split(
            train_val,
            train_val["stratum"],
            groups=train_val["source_image"],
        )
    )

    split_table = table.copy()
    split_table["split"] = "test"
    split_table.loc[train_val.index[val_rel], "split"] = "val"
    split_table.loc[train_val.index[train_rel], "split"] = "train"

    group_sets = {
        name: set(group["source_image"])
        for name, group in split_table.groupby("split")
    }
    assert group_sets["train"].isdisjoint(group_sets["val"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["val"].isdisjoint(group_sets["test"])
    return split_table


class SquarePad:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(
            image,
            border=(left, top, side - width - left, side - height - top),
            fill=0,
        )


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8, fill=0),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


class DefectDataset(Dataset):
    def __init__(self, table: pd.DataFrame, transform: transforms.Compose):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        with Image.open(row["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            torch.tensor(int(row["class_id"]), dtype=torch.long),
            torch.tensor(float(row["target"]), dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


def build_loaders(split_table: pd.DataFrame, config: Config):
    train_transform, eval_transform = build_transforms(config.image_size)
    subsets = {
        split: split_table.loc[split_table["split"].eq(split)].copy()
        for split in ("train", "val", "test")
    }
    datasets = {
        "train": DefectDataset(subsets["train"], train_transform),
        "val": DefectDataset(subsets["val"], eval_transform),
        "test": DefectDataset(subsets["test"], eval_transform),
    }

    # Equal expected contribution from every (object class, binary label) group.
    group_counts = subsets["train"].groupby(
        ["source_class", "target"]
    ).size()
    sample_weights = subsets["train"].apply(
        lambda row: 1.0 / group_counts.loc[(row["source_class"], row["target"])],
        axis=1,
    ).to_numpy()
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        common["prefetch_factor"] = 2
    loaders = {
        "train": DataLoader(datasets["train"], sampler=sampler, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, datasets, eval_transform


def _resnet_backbone(name: str, pretrained: bool):
    choices = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
        "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
        "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
    }
    if name not in choices:
        raise ValueError(f"Unsupported backbone {name!r}; choose from {sorted(choices)}")
    constructor, default_weights, feature_dim = choices[name]
    network = constructor(weights=default_weights if pretrained else None)
    features = nn.Sequential(
        network.conv1,
        network.bn1,
        network.relu,
        network.maxpool,
        network.layer1,
        network.layer2,
        network.layer3,
        network.layer4,
    )
    return features, feature_dim


class ClassConditionedResNet(nn.Module):
    """One ResNet with class-conditioned spatial attention and FiLM features."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.features, feature_dim = _resnet_backbone(backbone, pretrained)
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.attention_query = nn.Linear(embedding_dim, feature_dim)
        self.film = nn.Linear(embedding_dim, feature_dim * 2)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        hidden_dim = 512 if feature_dim >= 1024 else 256
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor, class_ids: torch.Tensor):
        feature_map = self.features(images)
        embedding = self.class_embedding(class_ids)

        keys = F.normalize(feature_map, dim=1)
        query = F.normalize(self.attention_query(embedding), dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        attention_logits = torch.einsum("bchw,bc->bhw", keys, query) * scale
        attention = attention_logits.flatten(1).softmax(dim=1).view_as(attention_logits)
        attended = torch.einsum("bchw,bhw->bc", feature_map, attention)

        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        gamma, beta = self.film(embedding).chunk(2, dim=1)
        conditioned = pooled * (1.0 + torch.tanh(gamma)) + beta
        logits = self.classifier(torch.cat((attended, conditioned), dim=1)).squeeze(1)
        return logits, attention


def build_model(config: Config, num_classes: int, device: torch.device):
    model = ClassConditionedResNet(
        num_classes=num_classes,
        backbone=config.backbone,
        pretrained=config.pretrained,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device, memory_format=torch.channels_last)
    return model


def _smoothed_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    label_smoothing: float = 0.0,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets, all_probabilities = [], []
    progress = tqdm(loader, leave=False, desc="train" if training else "eval")
    for images, class_ids, targets, _ in progress:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        class_ids = class_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits, _ = model(images, class_ids)
            loss_targets = _smoothed_targets(targets, label_smoothing) if training else targets
            loss = criterion(logits, loss_targets)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(targets)
        all_targets.append(targets.detach().cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_targets),
        np.concatenate(all_probabilities),
    )


def safe_auc(metric, targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(metric(targets, probabilities)) if np.unique(targets).size == 2 else float("nan")


def binary_metrics(
    targets: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {
        "samples": int(len(targets)),
        "threshold": float(threshold),
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall_sensitivity": recall_score(targets, predictions, zero_division=0),
        "specificity": specificity,
        "negative_predictive_value": npv,
        "f1": f1_score(targets, predictions, zero_division=0),
        "mcc": matthews_corrcoef(targets, predictions),
        "roc_auc": safe_auc(roc_auc_score, targets, probabilities),
        "pr_auc": safe_auc(average_precision_score, targets, probabilities),
        "brier_score": brier_score_loss(targets, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


@torch.inference_mode()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> pd.DataFrame:
    model.eval()
    probabilities, targets, indices = [], [], []
    for images, class_ids, batch_targets, batch_indices in tqdm(
        loader, leave=False, desc="predict"
    ):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        class_ids = class_ids.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits, _ = model(images, class_ids)
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        targets.extend(batch_targets.tolist())
        indices.extend(batch_indices.tolist())
    output = loader.dataset.table.iloc[indices].copy().reset_index(drop=True)
    output["target"] = np.asarray(targets, dtype=np.int64)
    output["probability_defective"] = probabilities
    return output


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    config: Config,
    device: torch.device,
):
    backbone_parameters = list(model.features.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("features.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": config.learning_rate * config.backbone_lr_multiplier,
            },
            {"params": head_parameters, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_score = -float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_targets, train_probs = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            label_smoothing=config.label_smoothing,
        )
        with torch.inference_mode():
            val_loss, val_targets, val_probs = run_epoch(
                model, loaders["val"], criterion, device
            )
        scheduler.step()
        val_pr_auc = safe_auc(average_precision_score, val_targets, val_probs)
        train_pr_auc = safe_auc(average_precision_score, train_targets, train_probs)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_pr_auc": train_pr_auc,
            "val_pr_auc": val_pr_auc,
            "learning_rate": optimizer.param_groups[-1]["lr"],
        })
        print(
            f"Epoch {epoch:02d}/{config.epochs}: "
            f"loss={train_loss:.4f}/{val_loss:.4f}, "
            f"PR-AUC={train_pr_auc:.4f}/{val_pr_auc:.4f}"
        )
        if val_pr_auc > best_score + 1e-5:
            best_score = val_pr_auc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}; best validation PR-AUC={best_score:.4f}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    return model, pd.DataFrame(history), best_score


def _draw_confusion(ax, matrix: np.ndarray, title: str, fmt: str) -> None:
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.figure.colorbar(image, ax=ax, fraction=0.046)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["normal", "defective"],
        yticklabels=["normal", "defective"],
        xlabel="Predicted", ylabel="True", title=title,
    )
    for row in range(2):
        for column in range(2):
            ax.text(column, row, format(matrix[row, column], fmt), ha="center", va="center")


def save_reports(
    test_predictions: pd.DataFrame,
    history: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = test_predictions["target"].to_numpy()
    probabilities = test_predictions["probability_defective"].to_numpy()
    predictions = (probabilities >= threshold).astype(int)
    test_predictions = test_predictions.copy()
    test_predictions["predicted_label"] = np.where(predictions == 1, "defective", "normal")
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)

    overall = pd.DataFrame([binary_metrics(targets, probabilities, threshold)], index=["overall"])
    per_class_rows = []
    for class_name, group in test_predictions.groupby("source_class", sort=True):
        values = binary_metrics(
            group["target"], group["probability_defective"], threshold
        )
        values["source_class"] = class_name
        per_class_rows.append(values)
    per_class = pd.DataFrame(per_class_rows).set_index("source_class")
    overall.to_csv(output_dir / "overall_metrics.csv")
    per_class.to_csv(output_dir / "per_class_metrics.csv")

    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    _draw_confusion(axes[0, 0], matrix, "Test confusion matrix", "d")
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    _draw_confusion(axes[0, 1], normalized, "Row-normalized confusion matrix", ".2f")
    fpr, tpr, _ = roc_curve(targets, probabilities)
    axes[1, 0].plot(fpr, tpr, label=f"AUC = {overall.loc['overall', 'roc_auc']:.3f}")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[1, 0].legend()
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    axes[1, 1].plot(recall, precision, label=f"AP = {overall.loc['overall', 'pr_auc']:.3f}")
    axes[1, 1].axhline(targets.mean(), linestyle="--", color="gray", label="Prevalence")
    axes[1, 1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "test_evaluation.png", dpi=180, bbox_inches="tight")
    plt.show()

    history_figure, history_axes = plt.subplots(1, 2, figsize=(13, 4.5))
    history_axes[0].plot(history["epoch"], history["train_loss"], label="train")
    history_axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    history_axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Loss")
    history_axes[0].legend()
    history_axes[1].plot(history["epoch"], history["train_pr_auc"], label="train")
    history_axes[1].plot(history["epoch"], history["val_pr_auc"], label="validation")
    history_axes[1].set(xlabel="Epoch", ylabel="PR-AUC", title="PR-AUC")
    history_axes[1].legend()
    history_figure.tight_layout()
    history_figure.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.show()
    return overall, per_class


@torch.inference_mode()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    class_name: str,
    class_to_idx: dict[str, int],
    transform: transforms.Compose,
    device: torch.device,
    threshold: float,
    show_attention: bool = True,
) -> dict[str, float | str]:
    if class_name not in class_to_idx:
        raise ValueError(f"Unknown class {class_name!r}; choose from {sorted(class_to_idx)}")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0)
    class_id = torch.tensor([class_to_idx[class_name]], dtype=torch.long)
    model.eval()
    logits, attention = model(
        tensor.to(device, memory_format=torch.channels_last), class_id.to(device)
    )
    probability = float(torch.sigmoid(logits).item())
    predicted_label = "defective" if probability >= threshold else "normal"

    if show_attention:
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        display_image = (tensor[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        heatmap = F.interpolate(
            attention[:, None].float().cpu(),
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        heatmap = (heatmap - heatmap.min()) / max(float(np.ptp(heatmap)), 1e-8)
        plt.figure(figsize=(6, 6))
        plt.imshow(display_image)
        plt.imshow(heatmap, cmap="jet", alpha=0.45)
        plt.axis("off")
        plt.title(f"{class_name}: {predicted_label} ({probability:.3f})")
        plt.show()
    return {
        "class_name": class_name,
        "predicted_label": predicted_label,
        "probability_defective": probability,
        "threshold": threshold,
    }


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
):
    """Restore a saved model and everything required by ``predict_image``."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_values = dict(checkpoint["config"])
    config_values["ridac_dir"] = Path(config_values["ridac_dir"])
    config = Config(**config_values)
    class_to_idx = {
        str(name): int(index) for name, index in checkpoint["class_to_idx"].items()
    }
    model = ClassConditionedResNet(
        num_classes=len(class_to_idx),
        backbone=config.backbone,
        pretrained=False,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device, memory_format=torch.channels_last).eval()
    _, eval_transform = build_transforms(config.image_size)
    return {
        "model": model,
        "config": config,
        "device": device,
        "class_to_idx": class_to_idx,
        "threshold": float(checkpoint["threshold"]),
        "eval_transform": eval_transform,
    }


def run_training(config: Config | None = None):
    config = config or Config()
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    table, class_to_idx = load_metadata(config)
    split_table = make_grouped_splits(table, config)
    split_table.to_csv(config.output_dir / "split_manifest.csv", index=False)
    print(pd.crosstab(
        [split_table["split"], split_table["source_class"]],
        split_table["assigned_label"],
    ))

    loaders, datasets, eval_transform = build_loaders(split_table, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, len(class_to_idx), device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Device: {device}; trainable parameters: {trainable:,}")
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    model, history, best_val_pr_auc = train_model(model, loaders, config, device)
    val_predictions = predict_loader(model, loaders["val"], device)
    threshold = best_f1_threshold(
        val_predictions["target"].to_numpy(),
        val_predictions["probability_defective"].to_numpy(),
    )
    test_predictions = predict_loader(model, loaders["test"], device)
    overall, per_class = save_reports(
        test_predictions, history, threshold, config.output_dir
    )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "class_to_idx": class_to_idx,
        "threshold": threshold,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "best_validation_pr_auc": best_val_pr_auc,
    }
    torch.save(checkpoint, config.output_dir / "best_conditional_resnet.pt")
    with (config.output_dir / "class_to_idx.json").open("w", encoding="utf-8") as handle:
        json.dump(class_to_idx, handle, indent=2)
    print(f"Best model and reports saved to {config.output_dir}")
    return {
        "model": raw_model,
        "config": config,
        "device": device,
        "class_to_idx": class_to_idx,
        "threshold": threshold,
        "eval_transform": eval_transform,
        "history": history,
        "overall_metrics": overall,
        "per_class_metrics": per_class,
        "test_predictions": test_predictions,
        "datasets": datasets,
    }


if __name__ == "__main__":
    run_training()
