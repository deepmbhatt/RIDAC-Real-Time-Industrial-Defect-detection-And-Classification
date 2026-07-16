#!/usr/bin/env python3
"""Leakage-safe pseudo-label fine-tuning for the VisA YOLO segmentation model.

The teacher is the newest trained YOLO segmentation checkpoint. Predictions from
``object_defect_dataset/full_results.csv`` identify source images on which the
teacher already produced the expected product class. Those images are predicted
again to recover polygons, filtered by confidence, and used only as pseudo-labels.
The original validation and test sets are never added to training.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


RIDAC_DIR = Path(__file__).resolve().parent
SOURCE_DATASET = RIDAC_DIR / "extracted_dataset"
SUPERVISED_DATASET = RIDAC_DIR / "VisA_Object_Segmentation.v3i.yolov11"
SOURCE_RESULTS = RIDAC_DIR / "object_defect_dataset" / "full_results.csv"
OUTPUT_ROOT = RIDAC_DIR / "pseudo_label_segmentation_retraining"
DATASET_ROOT = OUTPUT_ROOT / "dataset"
RUNS_ROOT = OUTPUT_ROOT / "runs"

SEED = 42
IMAGE_SIZE = 640
PSEUDO_CONFIDENCE = 0.90
PREDICTION_IOU = 0.70
PSEUDO_BATCH = 16
TRAIN_BATCH = 24
EPOCHS = 30
PATIENCE = 8
WORKERS = min(8, os.cpu_count() or 1)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def newest_teacher() -> Path:
    roots = [
        RIDAC_DIR.parent.parent / "deep_trial" / "Red-Light-Violation-Detection" / "runs" / "segment",
        RIDAC_DIR.parent.parent / "runs" / "segment",
    ]
    checkpoints = [
        checkpoint
        for root in roots
        if root.is_dir()
        for checkpoint in root.glob("train*/weights/best.pt")
    ]
    if not checkpoints:
        raise FileNotFoundError("No runs/segment/train*/weights/best.pt checkpoint was found")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_dataset_config() -> tuple[list[str], Path]:
    yaml_path = SUPERVISED_DATASET / "data.yaml"
    config = yaml.safe_load(yaml_path.read_text())
    raw_names = config["names"]
    if isinstance(raw_names, dict):
        names = [str(raw_names[index]) for index in sorted(map(int, raw_names))]
    else:
        names = list(map(str, raw_names))
    return names, yaml_path


def canonical_name(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return {"wheel": "fryum"}.get(name, name)


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def first_label_class(label_path: Path) -> int | None:
    if not label_path.is_file():
        return None
    for line in label_path.read_text().splitlines():
        values = line.split()
        if values:
            return int(float(values[0]))
    return None


def roboflow_source_stem(path: Path) -> str:
    base = re.sub(r"_JPG\.rf\..*$", "", path.stem, flags=re.IGNORECASE)
    return base


def supervised_source_exclusions(names: list[str]) -> tuple[set[tuple[str, str, str]], set[tuple[str, str]]]:
    """Return exact and condition-agnostic source identities in supervised splits."""
    exact: set[tuple[str, str, str]] = set()
    wildcard: set[tuple[str, str]] = set()
    for split in ("train", "valid", "test"):
        image_dir = SUPERVISED_DATASET / split / "images"
        label_dir = SUPERVISED_DATASET / split / "labels"
        for image_path in image_files(image_dir):
            class_id = first_label_class(label_dir / f"{image_path.stem}.txt")
            if class_id is None or class_id >= len(names):
                continue
            class_name = canonical_name(names[class_id])
            base = roboflow_source_stem(image_path)
            prefix = f"{class_name}_"
            matched = re.match(rf"^{re.escape(prefix)}(Normal|Anomaly)_(.+)$", base, re.IGNORECASE)
            if matched:
                exact.add((class_name, matched.group(1).lower(), matched.group(2).lower()))
            else:
                # Some Roboflow filenames retain only the numeric source stem.
                # Excluding both conditions is conservative and prevents leakage.
                wildcard.add((class_name, base.lower()))
    return exact, wildcard


def is_supervised_source(
    class_name: str,
    condition: str,
    source_stem: str,
    exact: set[tuple[str, str, str]],
    wildcard: set[tuple[str, str]],
) -> bool:
    class_name = canonical_name(class_name)
    return (
        (class_name, condition.lower(), source_stem.lower()) in exact
        or (class_name, source_stem.lower()) in wildcard
    )


def eligible_sources(names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SOURCE_RESULTS.is_file():
        raise FileNotFoundError(f"Run test_best_segmentation first: {SOURCE_RESULTS}")
    raw = pd.read_csv(SOURCE_RESULTS)
    raw["confidence"] = pd.to_numeric(raw["confidence"], errors="coerce")
    accepted = raw.loc[raw["status"].eq("ok")].copy()
    grouped = (
        accepted.groupby(["source_image", "source_class", "condition"], as_index=False)
        .agg(previous_detections=("object_id", "count"), previous_max_confidence=("confidence", "max"))
    )
    grouped = grouped.loc[grouped["previous_max_confidence"].ge(PSEUDO_CONFIDENCE)].copy()
    exact, wildcard = supervised_source_exclusions(names)
    grouped["source_stem"] = grouped["source_image"].map(lambda value: Path(value).stem)
    grouped["excluded_supervised_source"] = grouped.apply(
        lambda row: is_supervised_source(
            row.source_class, row.condition, row.source_stem, exact, wildcard
        ),
        axis=1,
    )
    audit = grouped.copy()
    grouped = grouped.loc[~grouped["excluded_supervised_source"]].reset_index(drop=True)
    grouped = grouped.loc[grouped["source_image"].map(lambda value: Path(value).is_file())].copy()
    valid_names = {canonical_name(name) for name in names}
    grouped = grouped.loc[grouped["source_class"].map(canonical_name).isin(valid_names)].copy()
    return grouped.reset_index(drop=True), audit


def replace_link(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        destination.unlink()
    destination.symlink_to(source.resolve())


def copy_supervised_training_data() -> int:
    target_images = DATASET_ROOT / "train" / "images"
    target_labels = DATASET_ROOT / "train" / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in image_files(SUPERVISED_DATASET / "train" / "images"):
        label_path = SUPERVISED_DATASET / "train" / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        target_stem = f"gt__{image_path.stem}"
        replace_link(target_images / f"{target_stem}{image_path.suffix.lower()}", image_path)
        shutil.copy2(label_path, target_labels / f"{target_stem}.txt")
        copied += 1
    return copied


def normalized_polygon(points: np.ndarray, width: int, height: int) -> list[float] | None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return None
    points[:, 0] = np.clip(points[:, 0] / width, 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1] / height, 0.0, 1.0)
    if np.unique(np.round(points, 6), axis=0).shape[0] < 3:
        return None
    return points.reshape(-1).tolist()


def pseudo_target_stem(row) -> str:
    digest = hashlib.sha1(str(Path(row.source_image).resolve()).encode()).hexdigest()[:10]
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", row.source_stem)
    return f"pseudo__{canonical_name(row.source_class)}__{row.condition.lower()}__{safe}__{digest}"


def generate_pseudo_labels(
    teacher_path: Path,
    candidates: pd.DataFrame,
    names: list[str],
    device: int | str,
) -> pd.DataFrame:
    pseudo_images = DATASET_ROOT / "train" / "images"
    pseudo_labels = DATASET_ROOT / "train" / "labels"
    teacher = YOLO(str(teacher_path))
    model_names = teacher.names if isinstance(teacher.names, dict) else dict(enumerate(teacher.names))
    model_class_ids = {canonical_name(name): int(index) for index, name in model_names.items()}
    missing = {canonical_name(name) for name in names} - set(model_class_ids)
    if missing:
        raise ValueError(f"Teacher checkpoint is missing dataset classes: {sorted(missing)}")

    sources = candidates["source_image"].tolist()
    # Passing a Python list makes this Ultralytics release decode every image
    # eagerly. A newline-delimited source file is streamed with bounded RAM.
    source_list_path = OUTPUT_ROOT / "pseudo_inference_sources.txt"
    source_list_path.write_text("\n".join(sources) + "\n")
    predictions = teacher.predict(
        source=str(source_list_path),
        imgsz=IMAGE_SIZE,
        conf=PSEUDO_CONFIDENCE,
        iou=PREDICTION_IOU,
        max_det=100,
        device=device,
        batch=PSEUDO_BATCH,
        half=torch.cuda.is_available(),
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    records = []
    for number, (row, result) in enumerate(zip(candidates.itertuples(index=False), predictions), 1):
        expected_id = model_class_ids[canonical_name(row.source_class)]
        label_lines: list[str] = []
        confidences: list[float] = []
        if result.boxes is not None and result.masks is not None:
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            height, width = result.orig_shape
            for class_id, score, polygon in zip(class_ids, scores, result.masks.xy):
                if class_id != expected_id or float(score) < PSEUDO_CONFIDENCE:
                    continue
                values = normalized_polygon(polygon, width, height)
                if values is None:
                    continue
                coordinates = " ".join(f"{value:.6f}" for value in values)
                label_lines.append(f"{expected_id} {coordinates}")
                confidences.append(float(score))

        status = "no_high_confidence_polygon"
        target_image = ""
        target_label = ""
        if label_lines:
            stem = pseudo_target_stem(row)
            source_path = Path(row.source_image)
            image_path = pseudo_images / f"{stem}{source_path.suffix.lower()}"
            label_path = pseudo_labels / f"{stem}.txt"
            replace_link(image_path, source_path)
            label_path.write_text("\n".join(label_lines) + "\n")
            status = "included"
            target_image, target_label = str(image_path), str(label_path)
        records.append({
            "source_image": row.source_image,
            "source_class": canonical_name(row.source_class),
            "condition": row.condition,
            "status": status,
            "object_count": len(label_lines),
            "min_confidence": min(confidences) if confidences else np.nan,
            "max_confidence": max(confidences) if confidences else np.nan,
            "target_image": target_image,
            "target_label": target_label,
        })
        if number % 250 == 0 or number == len(candidates):
            print(f"Pseudo-labeling: {number:,}/{len(candidates):,} images", flush=True)
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame.from_records(records)


def write_training_yaml(names: list[str]) -> Path:
    config = {
        "path": str(DATASET_ROOT.resolve()),
        "train": "train/images",
        "val": str((SUPERVISED_DATASET / "valid" / "images").resolve()),
        "test": str((SUPERVISED_DATASET / "test" / "images").resolve()),
        "nc": len(names),
        "names": names,
    }
    destination = OUTPUT_ROOT / "pseudo_retrain_data.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    return destination


def standard_metrics(metrics, model_tag: str, names: list[str]) -> tuple[dict, pd.DataFrame]:
    overall = {
        "model": model_tag,
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
    }
    per_class = []
    for metric_index, class_id in enumerate(metrics.seg.ap_class_index):
        class_id = int(class_id)
        per_class.append({
            "model": model_tag,
            "class_id": class_id,
            "class_name": names[class_id],
            "mask_precision": float(metrics.seg.p[metric_index]),
            "mask_recall": float(metrics.seg.r[metric_index]),
            "mask_f1": float(metrics.seg.f1[metric_index]),
            "mask_map50": float(metrics.seg.ap50[metric_index]),
            "mask_map50_95": float(metrics.seg.ap[metric_index]),
        })
    return overall, pd.DataFrame(per_class)


def evaluate_standard(
    checkpoint: Path,
    data_yaml: Path,
    model_tag: str,
    names: list[str],
    device: int | str,
):
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=PSEUDO_BATCH,
        device=device,
        plots=True,
        project=str(RUNS_ROOT),
        name=f"{model_tag}_test",
        exist_ok=True,
        verbose=False,
    )
    overall, per_class = standard_metrics(metrics, model_tag, names)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return overall, per_class


def read_yolo_masks(label_path: Path, width: int, height: int) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    if not label_path.is_file():
        return masks
    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            continue
        class_id = int(float(values[0]))
        points = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygon = np.rint(points).astype(np.int32)
        mask = masks.setdefault(class_id, np.zeros((height, width), dtype=np.uint8))
        cv2.fillPoly(mask, [polygon], 1)
    return {key: value.astype(bool) for key, value in masks.items()}


def boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    target_edge = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    if not pred_edge.any() and not target_edge.any():
        return 1.0
    if not pred_edge.any() or not target_edge.any():
        return 0.0
    radius = 2 * tolerance + 1
    dilation = np.ones((radius, radius), np.uint8)
    target_near = cv2.dilate(target_edge.astype(np.uint8), dilation).astype(bool)
    pred_near = cv2.dilate(pred_edge.astype(np.uint8), dilation).astype(bool)
    precision = np.count_nonzero(pred_edge & target_near) / np.count_nonzero(pred_edge)
    recall = np.count_nonzero(target_edge & pred_near) / np.count_nonzero(target_edge)
    return 2 * precision * recall / (precision + recall + 1e-12)


def evaluate_pixels(checkpoint: Path, model_tag: str, names: list[str], device: int | str):
    image_dir = SUPERVISED_DATASET / "test" / "images"
    label_dir = SUPERVISED_DATASET / "test" / "labels"
    paths = image_files(image_dir)
    model = YOLO(str(checkpoint))
    predictions = model.predict(
        source=[str(path) for path in paths],
        imgsz=IMAGE_SIZE,
        conf=0.25,
        iou=PREDICTION_IOU,
        max_det=100,
        device=device,
        batch=PSEUDO_BATCH,
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    rows = []
    total_intersection = total_union = total_pred = total_target = 0
    for image_path, result in zip(paths, predictions):
        height, width = result.orig_shape
        target_masks = read_yolo_masks(label_dir / f"{image_path.stem}.txt", width, height)
        pred_masks: dict[int, np.ndarray] = {}
        if result.boxes is not None and result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            for class_id, mask in zip(class_ids, masks):
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                binary = mask > 0.5
                pred_masks[class_id] = pred_masks.get(class_id, np.zeros_like(binary)) | binary
        intersection = union = pred_area = target_area = 0
        boundary_scores = []
        for class_id in set(target_masks) | set(pred_masks):
            target = target_masks.get(class_id, np.zeros((height, width), dtype=bool))
            prediction = pred_masks.get(class_id, np.zeros((height, width), dtype=bool))
            intersection += np.count_nonzero(prediction & target)
            union += np.count_nonzero(prediction | target)
            pred_area += np.count_nonzero(prediction)
            target_area += np.count_nonzero(target)
            boundary_scores.append(boundary_f1(prediction, target))
        iou = intersection / union if union else 1.0
        dice = 2 * intersection / (pred_area + target_area) if pred_area + target_area else 1.0
        precision = intersection / pred_area if pred_area else float(target_area == 0)
        recall = intersection / target_area if target_area else float(pred_area == 0)
        rows.append({
            "model": model_tag,
            "image": str(image_path),
            "pixel_iou": iou,
            "pixel_dice": dice,
            "pixel_precision": precision,
            "pixel_recall": recall,
            "boundary_f1": float(np.mean(boundary_scores)) if boundary_scores else 1.0,
        })
        total_intersection += intersection
        total_union += union
        total_pred += pred_area
        total_target += target_area
    frame = pd.DataFrame(rows)
    overall = {
        "model": model_tag,
        "pixel_iou_micro": total_intersection / total_union if total_union else 1.0,
        "pixel_dice_micro": 2 * total_intersection / (total_pred + total_target)
        if total_pred + total_target else 1.0,
        "pixel_precision_micro": total_intersection / total_pred if total_pred else float(total_target == 0),
        "pixel_recall_micro": total_intersection / total_target if total_target else float(total_pred == 0),
        "pixel_iou_macro": float(frame["pixel_iou"].mean()),
        "pixel_dice_macro": float(frame["pixel_dice"].mean()),
        "boundary_f1_macro": float(frame["boundary_f1"].mean()),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return overall, frame


def train_model(teacher_path: Path, data_yaml: Path, device: int | str) -> Path:
    model = YOLO(str(teacher_path))
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMAGE_SIZE,
        batch=TRAIN_BATCH,
        device=device,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=2e-4,
        lrf=0.05,
        weight_decay=5e-4,
        amp=True,
        close_mosaic=5,
        seed=SEED,
        deterministic=True,
        cache=False,
        plots=True,
        project=str(RUNS_ROOT),
        name="pseudo_finetune",
        exist_ok=True,
    )
    checkpoint = RUNS_ROOT / "pseudo_finetune" / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training finished without best.pt: {checkpoint}")
    return checkpoint


def save_metric_tables(standard_rows, pixel_rows, class_frames, pixel_frames) -> None:
    standard = pd.DataFrame(standard_rows)
    pixels = pd.DataFrame(pixel_rows)
    overall = standard.merge(pixels, on="model", how="outer")
    model_order = {"teacher": 0, "pseudo_finetuned": 1}
    overall = (
        overall.assign(_order=overall["model"].map(model_order).fillna(len(model_order)))
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    overall.to_csv(OUTPUT_ROOT / "overall_segmentation_metrics.csv", index=False)
    pd.concat(class_frames, ignore_index=True).to_csv(
        OUTPUT_ROOT / "per_class_mask_metrics.csv", index=False
    )
    pd.concat(pixel_frames, ignore_index=True).to_csv(
        OUTPUT_ROOT / "per_image_pixel_metrics.csv", index=False
    )
    if {"teacher", "pseudo_finetuned"}.issubset(set(overall["model"])):
        numeric = overall.set_index("model").select_dtypes(include=np.number)
        delta = (numeric.loc["pseudo_finetuned"] - numeric.loc["teacher"]).rename("absolute_change")
        delta.to_csv(
            OUTPUT_ROOT / "metric_improvements.csv",
            header=True,
            index_label="metric",
        )
    print("\nFinal segmentation metrics")
    print(overall.round(5).to_string(index=False))


def run(skip_pseudo: bool = False, skip_train: bool = False) -> None:
    seed_everything()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    names, _ = load_dataset_config()
    teacher_path = newest_teacher()
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    print(f"Teacher: {teacher_path}")
    print(f"Device: {device}; classes: {len(names)}; pseudo confidence: {PSEUDO_CONFIDENCE}")

    candidates, audit = eligible_sources(names)
    audit.to_csv(OUTPUT_ROOT / "pseudo_source_audit.csv", index=False)
    supervised_count = copy_supervised_training_data()
    manifest_path = OUTPUT_ROOT / "pseudo_label_manifest.csv"
    if skip_pseudo:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot --skip-pseudo without {manifest_path}")
        pseudo_manifest = pd.read_csv(manifest_path)
    else:
        print(f"Eligible leakage-free pseudo-label sources: {len(candidates):,}")
        pseudo_manifest = generate_pseudo_labels(teacher_path, candidates, names, device)
        pseudo_manifest.to_csv(manifest_path, index=False)
    included = pseudo_manifest.loc[pseudo_manifest["status"].eq("included")]
    print(
        f"Training set: {supervised_count:,} ground-truth images + "
        f"{len(included):,} pseudo-labeled images; "
        f"{int(included['object_count'].sum()):,} pseudo masks"
    )
    if included.empty:
        raise RuntimeError("No pseudo-label images passed the quality filters")

    data_yaml = write_training_yaml(names)
    standard_rows, pixel_rows, class_frames, pixel_frames = [], [], [], []
    teacher_standard, teacher_classes = evaluate_standard(
        teacher_path, data_yaml, "teacher", names, device
    )
    teacher_pixels, teacher_images = evaluate_pixels(teacher_path, "teacher", names, device)
    standard_rows.append(teacher_standard)
    pixel_rows.append(teacher_pixels)
    class_frames.append(teacher_classes)
    pixel_frames.append(teacher_images)

    trained_path = RUNS_ROOT / "pseudo_finetune" / "weights" / "best.pt"
    if not skip_train:
        trained_path = train_model(teacher_path, data_yaml, device)
    elif not trained_path.is_file():
        raise FileNotFoundError(f"Cannot --skip-train without {trained_path}")

    trained_standard, trained_classes = evaluate_standard(
        trained_path, data_yaml, "pseudo_finetuned", names, device
    )
    trained_pixels, trained_images = evaluate_pixels(
        trained_path, "pseudo_finetuned", names, device
    )
    standard_rows.append(trained_standard)
    pixel_rows.append(trained_pixels)
    class_frames.append(trained_classes)
    pixel_frames.append(trained_images)
    save_metric_tables(standard_rows, pixel_rows, class_frames, pixel_frames)
    metadata = {
        "teacher": str(teacher_path),
        "trained_model": str(trained_path),
        "pseudo_confidence": PSEUDO_CONFIDENCE,
        "ground_truth_train_images": supervised_count,
        "pseudo_train_images": int(len(included)),
        "pseudo_masks": int(included["object_count"].sum()),
        "validation_images": len(image_files(SUPERVISED_DATASET / "valid" / "images")),
        "test_images": len(image_files(SUPERVISED_DATASET / "test" / "images")),
        "seed": SEED,
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pseudo", action="store_true", help="Reuse pseudo_label_manifest.csv and labels")
    parser.add_argument("--skip-train", action="store_true", help="Reuse the existing pseudo_finetune best.pt")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(skip_pseudo=arguments.skip_pseudo, skip_train=arguments.skip_train)
