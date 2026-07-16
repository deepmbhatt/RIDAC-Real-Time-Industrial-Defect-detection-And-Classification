"""Create a new class-balanced dataset using defective-image augmentation.

Original crops and result CSVs are never deleted or modified. The generated
dataset has ``images/<object_class>/<normal|defective>`` directories. Original
images are hard-linked when possible (copied as a fallback), while only the
additional defective samples consume new image storage.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class BalanceConfig:
    ridac_dir: Path = field(default_factory=lambda: find_ridac_dir())
    defective_to_normal_ratio: float = 0.50
    max_augmentations_per_defective: int = 8
    rotation_degrees: float = 15.0
    brightness_change: float = 0.15
    contrast_change: float = 0.10
    saturation_change: float = 0.10
    scale_change: float = 0.05
    max_noise_sigma: float = 2.0
    seed: int = 42
    output_name: str | None = None

    @property
    def source_root(self) -> Path:
        return self.ridac_dir / "object_defect_dataset"


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


def load_source_rows(config: BalanceConfig) -> tuple[pd.DataFrame, str]:
    """Prefer the final table, otherwise use every currently completed phase."""
    full_csv = config.source_root / "full_results.csv"
    if full_csv.is_file():
        table = pd.read_csv(full_csv)
        source_tag = "full"
        source_csvs = [full_csv]
    else:
        candidates = [
            config.source_root / "sample_results.csv",
            config.source_root / "remaining_results.csv",
        ]
        source_csvs = [path for path in candidates if path.is_file()]
        if not source_csvs:
            raise FileNotFoundError(
                f"No result CSV exists in {config.source_root}. Run segmentation first."
            )
        table = pd.concat((pd.read_csv(path) for path in source_csvs), ignore_index=True)
        source_tag = "_".join(path.stem.replace("_results", "") for path in source_csvs)

    required = {
        "source_image", "source_class", "status", "assigned_label", "saved_path"
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Source results are missing columns: {sorted(missing)}")

    table = table.loc[
        table["status"].eq("ok")
        & table["assigned_label"].isin(("normal", "defective"))
        & table["saved_path"].notna()
    ].copy()
    table["original_crop_path"] = table.apply(
        lambda row: resolve_crop(row, config.source_root), axis=1
    ).astype(str)
    exists = table["original_crop_path"].map(lambda value: Path(value).is_file())
    if not exists.all():
        examples = table.loc[~exists, "original_crop_path"].head(5).tolist()
        raise FileNotFoundError(
            f"{int((~exists).sum())} source crops are missing. Examples: {examples}"
        )
    table = table.drop_duplicates(subset=["original_crop_path"]).reset_index(drop=True)
    print(
        f"Loaded {len(table):,} valid crops from "
        + ", ".join(path.name for path in source_csvs)
    )
    return table, source_tag


def resolve_crop(row: pd.Series, source_root: Path) -> Path:
    path = Path(str(row["saved_path"]))
    if path.is_file():
        return path.resolve()
    fallback = source_root / "binary_dataset" / row["assigned_label"] / path.name
    return fallback.resolve()


def output_root(config: BalanceConfig, source_tag: str) -> Path:
    ratio_tag = int(round(config.defective_to_normal_ratio * 100))
    name = config.output_name or f"balanced_augmented_{source_tag}_r{ratio_tag}_cap{config.max_augmentations_per_defective}_seed{config.seed}"
    return config.source_root / name


def link_or_copy(source: Path, destination: Path) -> str:
    """Create an independent directory entry without changing the source file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def balanced_parent_indices(count: int, needed: int, rng: np.random.Generator):
    """Use each defective parent once per cycle before reusing any parent."""
    indices: list[int] = []
    base = np.arange(count)
    while len(indices) < needed:
        indices.extend(rng.permutation(base).tolist())
    return indices[:needed]


def augmentation_parameters(config: BalanceConfig, rng: np.random.Generator) -> dict:
    flip = rng.choice(("none", "horizontal", "vertical", "both"), p=(0.15, 0.45, 0.15, 0.25))
    return {
        "flip": str(flip),
        "rotation_degrees": float(rng.uniform(-config.rotation_degrees, config.rotation_degrees)),
        "scale": float(rng.uniform(1.0 - config.scale_change, 1.0 + config.scale_change)),
        "brightness": float(rng.uniform(1.0 - config.brightness_change, 1.0 + config.brightness_change)),
        "contrast": float(rng.uniform(1.0 - config.contrast_change, 1.0 + config.contrast_change)),
        "saturation": float(rng.uniform(1.0 - config.saturation_change, 1.0 + config.saturation_change)),
        "noise_sigma": float(rng.uniform(0.0, config.max_noise_sigma)),
    }


def augment_image(image: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    flip_codes = {"horizontal": 1, "vertical": 0, "both": -1}
    if params["flip"] in flip_codes:
        image = cv2.flip(image, flip_codes[params["flip"]])

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0),
        params["rotation_degrees"],
        params["scale"],
    )
    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    values = image.astype(np.float32) / 255.0
    values *= params["brightness"]
    # Gamma-like contrast keeps the black segmentation background black.
    values = np.power(np.clip(values, 0.0, 1.0), 1.0 / params["contrast"])
    image = np.clip(values * 255.0, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * params["saturation"], 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if params["noise_sigma"] > 0:
        foreground = np.any(image > 2, axis=2, keepdims=True)
        noise = rng.normal(0.0, params["noise_sigma"], image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise * foreground
        image = np.clip(noisy, 0, 255).astype(np.uint8)
    return image


def original_manifest_rows(
    table: pd.DataFrame, images_root: Path
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    methods: dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    for row in tqdm(table.itertuples(index=False), total=len(table), desc="Linking originals"):
        source = Path(row.original_crop_path)
        destination = (
            images_root
            / row.source_class
            / row.assigned_label
            / f"original_{source.name}"
        )
        method = link_or_copy(source, destination)
        methods[method] += 1
        record = row._asdict()
        record.update({
            "crop_path": str(destination.resolve()),
            "saved_path": str(destination.resolve()),
            "is_augmented": False,
            "parent_crop_path": str(source.resolve()),
            "augmentation_id": "",
            "augmentation_parameters": "",
            "storage_method": method,
        })
        rows.append(record)
    return rows, methods


def create_augmented_rows(
    table: pd.DataFrame,
    images_root: Path,
    config: BalanceConfig,
) -> tuple[list[dict], list[dict]]:
    augmented_rows: list[dict] = []
    report_rows: list[dict] = []
    class_names = sorted(table["source_class"].unique())
    for class_index, class_name in enumerate(class_names):
        class_rows = table.loc[table["source_class"].eq(class_name)]
        normal_count = int(class_rows["assigned_label"].eq("normal").sum())
        defective = class_rows.loc[class_rows["assigned_label"].eq("defective")].reset_index(drop=True)
        defective_count = len(defective)
        target_defective = int(np.ceil(normal_count * config.defective_to_normal_ratio))
        requested = max(0, target_defective - defective_count)
        augmentation_cap = defective_count * config.max_augmentations_per_defective
        needed = min(requested, augmentation_cap)
        if defective_count == 0 and needed > 0:
            raise ValueError(
                f"Cannot balance {class_name}: it has {normal_count} normal but no defective crops."
            )

        rng = np.random.default_rng(config.seed + class_index * 10_007)
        parent_indices = balanced_parent_indices(defective_count, needed, rng) if needed else []
        iterator = tqdm(
            enumerate(parent_indices),
            total=needed,
            desc=f"Augmenting {class_name}",
            leave=False,
        )
        for augmentation_index, parent_index in iterator:
            parent = defective.iloc[parent_index]
            parent_path = Path(parent["original_crop_path"])
            destination = (
                images_root
                / class_name
                / "defective"
                / f"aug_{augmentation_index:06d}_{parent_path.stem}.png"
            )
            # Per-image RNG makes interrupted runs exactly reproducible on resume.
            item_rng = np.random.default_rng(
                config.seed + class_index * 1_000_003 + augmentation_index
            )
            params = augmentation_parameters(config, item_rng)
            if not destination.exists():
                image = cv2.imread(str(parent_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Could not read defective crop: {parent_path}")
                augmented = augment_image(image, params, item_rng)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(destination), augmented, [cv2.IMWRITE_PNG_COMPRESSION, 2]
                ):
                    raise OSError(f"Could not write augmented crop: {destination}")

            record = parent.to_dict()
            record.update({
                "crop_path": str(destination.resolve()),
                "saved_path": str(destination.resolve()),
                "is_augmented": True,
                # Keep the original group so augmented siblings cannot leak across splits.
                "source_image": parent["source_image"],
                "parent_crop_path": str(parent_path.resolve()),
                "augmentation_id": f"{class_name}_{augmentation_index:06d}",
                "augmentation_parameters": json.dumps(params, sort_keys=True),
                "storage_method": "generated",
            })
            augmented_rows.append(record)

        report_rows.append({
            "source_class": class_name,
            "normal_original": normal_count,
            "defective_original": defective_count,
            "requested_defective_target": target_defective,
            "maximum_generated_from_cap": augmentation_cap,
            "defective_augmented": needed,
            "defective_final": defective_count + needed,
            "target_shortfall_after_cap": max(0, target_defective - defective_count - needed),
            "final_defective_to_normal_ratio": (
                (defective_count + needed) / normal_count if normal_count else np.nan
            ),
        })
    return augmented_rows, report_rows


def create_balanced_dataset(config: BalanceConfig | None = None):
    config = config or BalanceConfig()
    if not 0 < config.defective_to_normal_ratio <= 2.0:
        raise ValueError("defective_to_normal_ratio must be in (0, 2].")
    if config.max_augmentations_per_defective < 0:
        raise ValueError("max_augmentations_per_defective must be non-negative.")
    seed_everything(config.seed)
    source_table, source_tag = load_source_rows(config)
    destination_root = output_root(config, source_tag)
    manifest_path = destination_root / "balanced_manifest.csv"
    report_path = destination_root / "balance_report.csv"
    if manifest_path.is_file() and report_path.is_file():
        print(f"Balanced dataset already exists; nothing was overwritten: {destination_root}")
        return pd.read_csv(manifest_path, low_memory=False), pd.read_csv(report_path), destination_root

    images_root = destination_root / "images"
    destination_root.mkdir(parents=True, exist_ok=True)
    original_rows, storage_methods = original_manifest_rows(source_table, images_root)
    augmented_rows, report_rows = create_augmented_rows(source_table, images_root, config)
    manifest = pd.DataFrame.from_records(original_rows + augmented_rows)
    report = pd.DataFrame.from_records(report_rows)

    manifest_tmp = manifest_path.with_suffix(".csv.tmp")
    report_tmp = report_path.with_suffix(".csv.tmp")
    manifest.to_csv(manifest_tmp, index=False)
    report.to_csv(report_tmp, index=False)
    manifest_tmp.replace(manifest_path)
    report_tmp.replace(report_path)
    with (destination_root / "augmentation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(config).items()
            },
            handle,
            indent=2,
        )

    print(f"Original storage methods: {storage_methods}")
    print(f"Created {len(augmented_rows):,} augmented defective crops")
    print(f"Balanced dataset saved to {destination_root}")
    return manifest, report, destination_root


if __name__ == "__main__":
    _, summary, _ = create_balanced_dataset()
    print(summary.to_string(index=False))
