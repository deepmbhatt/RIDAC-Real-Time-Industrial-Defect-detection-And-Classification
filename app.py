#!/usr/bin/env python3
"""RIDAC Gradio application: YOLO11 segmentation -> MobileNetV3 classification."""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import YOLO

from src.models import build_model, build_transforms


HERE = Path(__file__).resolve().parent
DEFAULT_YOLO = HERE / "models" / "yolo11_seg_best.pt"
DEFAULT_MOBILENET = HERE / "models" / "mobilenet_v3_best.pt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RESULT_COLUMNS = [
    "image",
    "result",
    "objects",
    "defective_objects",
    "classes",
    "max_defect_probability",
    "yolo_ms",
    "mobilenet_ms",
    "total_inference_ms",
    "message",
]
MAX_IMAGES = 10_000
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 150_000_000


@dataclass
class SegmentedObject:
    crop: Image.Image
    class_name: str
    class_id: int
    yolo_confidence: float
    box: tuple[int, int, int, int]
    polygon: np.ndarray | None


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


class InspectionModels:
    """Models loaded once and reused for every Gradio request."""

    def __init__(self, yolo_path: Path, mobilenet_path: Path):
        for path in (yolo_path, mobilenet_path):
            if not path.is_file():
                raise FileNotFoundError(f"Model checkpoint not found: {path}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(str(yolo_path), task="segment")
        checkpoint = torch.load(mobilenet_path, map_location="cpu", weights_only=False)
        self.class_to_idx = {
            str(name): int(index) for name, index in checkpoint["class_to_idx"].items()
        }
        self.threshold = float(checkpoint["threshold"])
        config = checkpoint.get("config", {})
        self.image_size = int(config.get("image_size", 224))
        self.mobilenet = build_model(
            backbone="mobilenet_v3_large",
            num_classes=len(self.class_to_idx),
            device=self.device,
            pretrained=False,
            embedding_dim=int(config.get("embedding_dim", 128)),
            dropout=float(config.get("dropout", 0.30)),
        )
        self.mobilenet.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.mobilenet.eval()
        _, self.transform = build_transforms(self.image_size)

        yolo_names = _model_names(self.yolo.names)
        yolo_classes = set(yolo_names.values())
        mobile_classes = set(self.class_to_idx)
        if yolo_classes != mobile_classes:
            raise ValueError(
                "YOLO/MobileNet class mismatch. "
                f"Only YOLO: {sorted(yolo_classes - mobile_classes)}; "
                f"only MobileNet: {sorted(mobile_classes - yolo_classes)}"
            )
        self.yolo_names = yolo_names
        self._warm_up()

    def _warm_up(self) -> None:
        """Exclude one-time CUDA/kernel initialization from reported inference time."""
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(
            source=blank,
            imgsz=640,
            conf=0.25,
            device=str(self.device),
            verbose=False,
        )
        tensor = torch.zeros(
            (1, 3, self.image_size, self.image_size), device=self.device
        ).to(memory_format=torch.channels_last)
        class_id = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            self.mobilenet(tensor, class_id)
        _sync_cuda(self.device)

    def segment(
        self, image: Image.Image, confidence: float, iou: float, image_size: int
    ) -> tuple[list[SegmentedObject], float]:
        rgb = np.asarray(image.convert("RGB"))
        _sync_cuda(self.device)
        started = time.perf_counter()
        result = self.yolo.predict(
            source=rgb,
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            device=str(self.device),
            retina_masks=True,
            verbose=False,
        )[0]
        _sync_cuda(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if result.boxes is None or len(result.boxes) == 0:
            return [], elapsed_ms

        height, width = rgb.shape[:2]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)
        objects: list[SegmentedObject] = []

        for box, class_id, score, polygon in zip(
            boxes, class_ids, confidences, polygons
        ):
            x1, y1, x2, y2 = box
            padding = max(2, int(0.02 * max(x2 - x1, y2 - y1)))
            left = max(0, int(np.floor(x1)) - padding)
            top = max(0, int(np.floor(y1)) - padding)
            right = min(width, int(np.ceil(x2)) + padding)
            bottom = min(height, int(np.ceil(y2)) + padding)
            if right <= left or bottom <= top:
                continue

            polygon_array = None
            object_image = rgb
            if polygon is not None and len(polygon) >= 3:
                polygon_array = np.asarray(polygon, dtype=np.float32)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.rint(polygon_array).astype(np.int32)], 255)
                object_image = cv2.bitwise_and(rgb, rgb, mask=mask)
            crop = Image.fromarray(object_image[top:bottom, left:right]).convert("RGB")
            class_name = self.yolo_names.get(int(class_id), f"class_{class_id}")
            if class_name not in self.class_to_idx:
                continue
            objects.append(
                SegmentedObject(
                    crop=crop,
                    class_name=class_name,
                    class_id=self.class_to_idx[class_name],
                    yolo_confidence=float(score),
                    box=(left, top, right, bottom),
                    polygon=polygon_array,
                )
            )
        return objects, elapsed_ms

    def classify(
        self, objects: list[SegmentedObject]
    ) -> tuple[np.ndarray, float]:
        if not objects:
            return np.empty(0, dtype=np.float32), 0.0

        # Preprocessing is included in MobileNet pipeline time; model loading is not.
        _sync_cuda(self.device)
        started = time.perf_counter()
        images = torch.stack([self.transform(item.crop) for item in objects]).to(
            self.device, non_blocking=True, memory_format=torch.channels_last
        )
        class_ids = torch.tensor(
            [item.class_id for item in objects], dtype=torch.long, device=self.device
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            logits, _ = self.mobilenet(images, class_ids)
            probabilities = torch.sigmoid(logits)
        _sync_cuda(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return probabilities.float().cpu().numpy(), elapsed_ms


MODELS: InspectionModels | None = None


def annotate(
    image: Image.Image,
    objects: list[SegmentedObject],
    probabilities: np.ndarray,
    threshold: float,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, round(max(canvas.size) / 500))

    for index, (item, probability) in enumerate(zip(objects, probabilities), start=1):
        defective = float(probability) >= threshold
        color = (225, 45, 45) if defective else (35, 180, 90)
        label = (
            f"{index} {item.class_name} | "
            f"{'DEFECTIVE' if defective else 'NORMAL'} {float(probability):.1%}"
        )
        if item.polygon is not None:
            points = [tuple(map(float, point)) for point in item.polygon]
            draw.line(points + [points[0]], fill=color, width=line_width)
        draw.rectangle(item.box, outline=color, width=line_width)
        text_box = draw.textbbox((item.box[0], item.box[1]), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        tx = item.box[0]
        ty = max(0, item.box[1] - text_height - 6)
        draw.rectangle((tx, ty, tx + text_width + 6, ty + text_height + 6), fill=color)
        draw.text((tx + 3, ty + 3), label, fill="white", font=font)
    return canvas


def _zip_path(upload) -> Path:
    if upload is None:
        raise gr.Error("Upload a ZIP file first.")
    value = upload if isinstance(upload, (str, os.PathLike)) else upload.name
    path = Path(value)
    if path.suffix.lower() != ".zip" or not zipfile.is_zipfile(path):
        raise gr.Error("The uploaded file is not a valid ZIP archive.")
    return path


def _image_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    total_bytes = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or path.name.startswith(".") or "__MACOSX" in path.parts:
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if info.file_size > MAX_IMAGE_BYTES:
            raise gr.Error(f"Image is too large: {info.filename}")
        total_bytes += info.file_size
        if total_bytes > MAX_TOTAL_BYTES:
            raise gr.Error("Uncompressed image data exceeds the 4 GiB safety limit.")
        members.append(info)
    members.sort(key=lambda item: item.filename.casefold())
    if not members:
        raise gr.Error("No supported images were found in the ZIP file.")
    if len(members) > MAX_IMAGES:
        raise gr.Error(f"ZIP contains {len(members):,} images; limit is {MAX_IMAGES:,}.")
    return members


def _timing_markdown(
    processed: int,
    total: int,
    yolo_ms: float,
    mobile_ms: float,
    inference_ms: float,
) -> str:
    average = inference_ms / processed if processed else 0.0
    throughput = 1000.0 / average if average else 0.0
    return (
        "### Inference time (model loading excluded)\n"
        f"**Processed:** {processed:,}/{total:,} images  \n"
        f"**YOLO11:** {yolo_ms / 1000:.3f} s  \n"
        f"**MobileNetV3:** {mobile_ms / 1000:.3f} s  \n"
        f"**Combined inference:** {inference_ms / 1000:.3f} s  \n"
        f"**Average:** {average:.1f} ms/image · **Throughput:** {throughput:.2f} images/s"
    )


def inspect_zip(
    upload,
    yolo_confidence: float,
    yolo_iou: float,
    yolo_image_size: int,
) -> Iterator[tuple[str, Image.Image | None, list[list], str, str | None]]:
    if MODELS is None:
        raise gr.Error("Models are not loaded. Restart the application.")

    path = _zip_path(upload)
    rows: list[dict] = []
    cumulative_yolo_ms = 0.0
    cumulative_mobile_ms = 0.0
    cumulative_inference_ms = 0.0

    with zipfile.ZipFile(path) as archive:
        members = _image_members(archive)
        total = len(members)
        yield (
            f"### Starting inspection\nFound **{total:,}** images. Models are already loaded.",
            None,
            [],
            _timing_markdown(0, total, 0.0, 0.0, 0.0),
            None,
        )

        for position, info in enumerate(members, start=1):
            annotated = None
            try:
                # ZIP reading and image decoding happen before the inference timer.
                data = archive.read(info)
                with Image.open(io.BytesIO(data)) as opened:
                    image = opened.convert("RGB")

                pipeline_started = time.perf_counter()
                objects, yolo_ms = MODELS.segment(
                    image,
                    confidence=float(yolo_confidence),
                    iou=float(yolo_iou),
                    image_size=int(yolo_image_size),
                )
                probabilities, mobile_ms = MODELS.classify(objects)
                _sync_cuda(MODELS.device)
                total_ms = (time.perf_counter() - pipeline_started) * 1000.0

                cumulative_yolo_ms += yolo_ms
                cumulative_mobile_ms += mobile_ms
                cumulative_inference_ms += total_ms
                defective_count = int((probabilities >= MODELS.threshold).sum())
                if not objects:
                    decision = "NO OBJECTS"
                    max_probability = None
                    message = "YOLO11 found no objects at the selected confidence."
                else:
                    decision = "DEFECTIVE" if defective_count else "NORMAL"
                    max_probability = float(probabilities.max())
                    message = ""
                annotated = annotate(image, objects, probabilities, MODELS.threshold)
                rows.append(
                    {
                        "image": info.filename,
                        "result": decision,
                        "objects": len(objects),
                        "defective_objects": defective_count,
                        "classes": ", ".join(sorted({item.class_name for item in objects})),
                        "max_defect_probability": (
                            "" if max_probability is None else round(max_probability, 6)
                        ),
                        "yolo_ms": round(yolo_ms, 2),
                        "mobilenet_ms": round(mobile_ms, 2),
                        "total_inference_ms": round(total_ms, 2),
                        "message": message,
                    }
                )
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as error:
                rows.append(
                    {
                        "image": info.filename,
                        "result": "ERROR",
                        "objects": 0,
                        "defective_objects": 0,
                        "classes": "",
                        "max_defect_probability": "",
                        "yolo_ms": "",
                        "mobilenet_ms": "",
                        "total_inference_ms": "",
                        "message": str(error),
                    }
                )

            normal = sum(row["result"] == "NORMAL" for row in rows)
            defective = sum(row["result"] == "DEFECTIVE" for row in rows)
            unresolved = position - normal - defective
            status = (
                f"### Processing {position:,}/{total:,}\n"
                f"Current: `{info.filename}` · **{rows[-1]['result']}**  \n"
                f"Normal: **{normal:,}** · Defective: **{defective:,}** · "
                f"No-object/error: **{unresolved:,}**"
            )
            table = [[row[column] for column in RESULT_COLUMNS] for row in rows]
            yield (
                status,
                annotated,
                table,
                _timing_markdown(
                    position,
                    total,
                    cumulative_yolo_ms,
                    cumulative_mobile_ms,
                    cumulative_inference_ms,
                ),
                None,
            )

    report = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    with tempfile.NamedTemporaryFile(
        prefix="inspection_results_", suffix=".csv", delete=False
    ) as handle:
        report_path = Path(handle.name)
    report.to_csv(report_path, index=False)
    normal = int(report["result"].eq("NORMAL").sum())
    defective = int(report["result"].eq("DEFECTIVE").sum())
    unresolved = len(report) - normal - defective
    final_status = (
        "### Inspection complete\n"
        f"Normal: **{normal:,}** · Defective: **{defective:,}** · "
        f"No-object/error: **{unresolved:,}** · Total: **{len(report):,}**"
    )
    yield (
        final_status,
        annotated,
        [[row[column] for column in RESULT_COLUMNS] for row in rows],
        _timing_markdown(
            len(rows),
            len(rows),
            cumulative_yolo_ms,
            cumulative_mobile_ms,
            cumulative_inference_ms,
        ),
        str(report_path),
    )


CSS = """
.gradio-container {max-width: 1500px !important;}
#status-panel {border-left: 5px solid #4f46e5; padding-left: 1rem;}
#timing-panel {border-left: 5px solid #0891b2; padding-left: 1rem;}
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="RIDAC Industrial Defect Inspection", css=CSS) as demo:
        gr.Markdown(
            "# RIDAC: Industrial Defect Inspection\n"
            "Upload a ZIP of images. Each image is processed once through "
            "**YOLO11 segmentation → class-conditioned MobileNetV3 defect classification**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                zip_file = gr.File(
                    label="ZIP file containing images",
                    file_types=[".zip"],
                    type="filepath",
                )
                confidence = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.05, label="YOLO confidence"
                )
                iou = gr.Slider(0.10, 0.90, value=0.70, step=0.05, label="YOLO IoU")
                image_size = gr.Dropdown(
                    choices=[416, 512, 640, 768, 960, 1024],
                    value=640,
                    label="YOLO inference size",
                )
                run_button = gr.Button("Inspect ZIP", variant="primary")
                gr.Markdown(
                    "The MobileNet decision threshold is read from its checkpoint. "
                    "Model loading and warm-up are excluded from all displayed timings."
                )
            with gr.Column(scale=2):
                status = gr.Markdown("Models loaded. Upload a ZIP to begin.", elem_id="status-panel")
                current_image = gr.Image(
                    label="Current processed image",
                    type="pil",
                    height=520,
                )

        results = gr.Dataframe(
            headers=RESULT_COLUMNS,
            datatype=["str", "str", "number", "number", "str", "str", "str", "str", "str", "str"],
            label="Live inspection results",
            interactive=False,
            wrap=True,
        )
        timing = gr.Markdown(
            _timing_markdown(0, 0, 0.0, 0.0, 0.0), elem_id="timing-panel"
        )
        report_file = gr.File(label="Download final CSV report", interactive=False)

        run_button.click(
            fn=inspect_zip,
            inputs=[zip_file, confidence, iou, image_size],
            outputs=[status, current_image, results, timing, report_file],
            show_progress="minimal",
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--mobilenet", type=Path, default=DEFAULT_MOBILENET)
    return parser.parse_args()


def main() -> None:
    global MODELS
    args = parse_args()
    print(f"Loading YOLO11 from {args.yolo}")
    print(f"Loading MobileNetV3 from {args.mobilenet}")
    MODELS = InspectionModels(args.yolo.resolve(), args.mobilenet.resolve())
    print(
        f"Models ready on {MODELS.device}; MobileNet threshold={MODELS.threshold:.6f}"
    )
    build_interface().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(Path(tempfile.gettempdir()).resolve())],
    )


if __name__ == "__main__":
    main()
