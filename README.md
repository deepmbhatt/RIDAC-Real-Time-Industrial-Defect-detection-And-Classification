# RIDAC: Real-Time Industrial Defect Detection and Classification

> **Inspect. Segment. Classify.**

RIDAC is a two-stage industrial visual-inspection system:

1. **YOLO11 instance segmentation** identifies and isolates manufactured objects.
2. A **class-conditioned MobileNetV3 classifier** labels every segmented object as **normal** or **defective**.

This repository is the cleaned, GitHub-ready version of the completed project. It includes the final trained checkpoints, production Gradio application, exact evaluation reports, plots, example images, and the original training/evaluation code. **No retraining is required to run the application.**

## Key result

The final pipeline processed 360 images with the following recorded model-inference time:

| Measurement | Result |
|---|---:|
| Images processed | **360 / 360** |
| YOLO11 segmentation | **7.419 s** |
| MobileNetV3 classification | **43.385 s** |
| Combined inference | **59.336 s** |
| Average latency | **164.8 ms/image** |
| Throughput | **6.07 images/s** |

The benchmark was recorded after model warm-up, excluding model loading and ZIP decoding. The project workstation contains an **NVIDIA RTX 6000 Ada Generation (48 GB)** GPU. End-to-end application time can vary with image dimensions, detected-object count, storage speed, CUDA/PyTorch versions, and hardware.

![RIDAC Gradio application](assets/screenshots/gradio_application.png)

The screenshot is a documentation render of the implemented Gradio layout populated only with the recorded 360-image benchmark values. Per-image prediction rows are intentionally omitted; the running application fills them from actual inference.

## Why the final system uses MobileNetV3

Three class-conditioned classifiers were evaluated on the same leakage-safe held-out test split of **6,815 segmented object crops**.

| Model | Parameters | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 11.64M | 99.384% | 93.564% | 95.089% | 87.295% | 91.026% | 90.796% | 98.838% | **95.897%** |
| **MobileNetV3-Large** | **4.33M** | **99.442%** | **93.792%** | **96.396%** | **87.705%** | **91.845%** | **91.667%** | **99.082%** | 94.997% |
| EfficientNet-B0 | 5.82M | 99.384% | 93.367% | 95.495% | 86.885% | 90.987% | 90.777% | 97.655% | 94.546% |

MobileNetV3 was selected because it produced the strongest overall operating point while being substantially smaller than ResNet18:

- Highest test accuracy, balanced accuracy, precision, recall, F1, MCC, and ROC-AUC.
- Only **4.33 million trainable parameters**.
- Only 8 false positives and 30 false negatives on the 6,815-crop test set.
- Best fit for repeated object-level inference inside the YOLO segmentation pipeline.

![Classifier comparison](assets/plots/classifiers/model_comparison.svg)

Exact machine-readable results are available in [`results/classification/model_comparison.csv`](results/classification/model_comparison.csv).

## Supported product classes

RIDAC was developed on 12 VisA industrial object categories:

`candle`, `capsules`, `cashew`, `chewinggum`, `fryum`, `macaroni1`, `macaroni2`, `pcb1`, `pcb2`, `pcb3`, `pcb4`, and `pipe_fryum`.

## Dataset provenance

### Original source dataset

RIDAC is based on the **Visual Anomaly (VisA) Dataset**, an industrial visual-anomaly detection and segmentation dataset introduced by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer in the ECCV 2022 paper *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*.

The original dataset was released by researchers affiliated with **AWS AI Labs** and **KAIST**. It contains:

| Property | Original VisA dataset |
|---|---:|
| Total images | 10,821 |
| Normal images | 9,621 |
| Anomalous images | 1,200 |
| Product categories | 12 |
| Application domains | 3 |
| Image type | High-resolution RGB |
| Original acquisition resolution | 4,000 × 6,000 |
| Ground truth | Image-level and pixel-level anomaly labels |

The categories cover complex printed circuit boards, scenes containing multiple object instances, and approximately aligned manufactured or food products. The reported anomalies include surface defects such as scratches, dents, cracks, and color spots, as well as structural defects such as missing or misplaced components.

Official sources:

- [Visual Anomaly (VisA) — AWS Registry of Open Data](https://registry.opendata.aws/visa/)
- [Official Amazon Science `spot-diff` repository](https://github.com/amazon-science/spot-diff)
- [ECCV 2022 paper](https://doi.org/10.1007/978-3-031-20056-4_23)
- [Open-access paper PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900389.pdf)
- [arXiv:2207.14315](https://arxiv.org/abs/2207.14315)

### Dataset used by this project

The data used in RIDAC should be understood as two related but distinct datasets:

1. **Original VisA data:** the source RGB images and anomaly masks obtained from the Visual Anomaly dataset.
2. **RIDAC-derived Roboflow export:** a project-specific object-instance segmentation dataset created by selecting 15 normal and 15 anomalous source images from each of the 12 VisA categories, manually annotating object polygons in Roboflow, and applying the documented preprocessing and augmentations.

The derived annotation seed therefore contains:

```text
12 product classes × (15 normal + 15 anomalous images) = 360 source images
```

Roboflow generated augmented train, validation, and test images from these source images. The resulting export is named **VisA Object Segmentation**, version 3:

- [VisA Object Segmentation v3 on Roboflow Universe](https://universe.roboflow.com/deeps-workspace-6tscj/visa_object_segmentation/dataset/3)

The Roboflow export is not a replacement for the complete original VisA dataset. It is a derived segmentation dataset prepared specifically for training RIDAC’s YOLO11 object-segmentation stage.

### Normal and defective examples

All examples below are real images from the project dataset.

<table>
  <thead><tr><th>Product</th><th>Normal</th><th>Defective / anomaly</th></tr></thead>
  <tbody>
    <tr><td>Candle</td><td><img src="assets/examples/candle_Normal_0085.JPG" width="300"></td><td><img src="assets/examples/candle_Anomaly_009.JPG" width="300"></td></tr>
    <tr><td>Capsules</td><td><img src="assets/examples/capsules_Normal_033.JPG" width="300"></td><td><img src="assets/examples/capsules_Anomaly_000.JPG" width="300"></td></tr>
    <tr><td>Cashew</td><td><img src="assets/examples/cashew_Normal_018.JPG" width="300"></td><td><img src="assets/examples/cashew_Anomaly_006.JPG" width="300"></td></tr>
    <tr><td>Chewing gum</td><td><img src="assets/examples/chewinggum_Normal_091.JPG" width="300"></td><td><img src="assets/examples/chewinggum_Anomaly_004.JPG" width="300"></td></tr>
    <tr><td>Fryum</td><td><img src="assets/examples/fryum_Normal_023.JPG" width="300"></td><td><img src="assets/examples/fryum_Anomaly_000.JPG" width="300"></td></tr>
    <tr><td>Macaroni 1</td><td><img src="assets/examples/macaroni1_Normal_0020.JPG" width="300"></td><td><img src="assets/examples/macaroni1_Anomaly_005.JPG" width="300"></td></tr>
    <tr><td>Macaroni 2</td><td><img src="assets/examples/macaroni2_Normal_0056.JPG" width="300"></td><td><img src="assets/examples/macaroni2_Anomaly_009.JPG" width="300"></td></tr>
    <tr><td>PCB 1</td><td><img src="assets/examples/pcb1_Normal_0254.JPG" width="300"></td><td><img src="assets/examples/pcb1_Anomaly_000.JPG" width="300"></td></tr>
    <tr><td>PCB 2</td><td><img src="assets/examples/pcb2_Normal_0041.JPG" width="300"></td><td><img src="assets/examples/pcb2_Anomaly_001.JPG" width="300"></td></tr>
    <tr><td>PCB 3</td><td><img src="assets/examples/pcb3_Normal_0035.JPG" width="300"></td><td><img src="assets/examples/pcb3_Anomaly_002.JPG" width="300"></td></tr>
    <tr><td>PCB 4</td><td><img src="assets/examples/pcb4_Normal_0177.JPG" width="300"></td><td><img src="assets/examples/pcb4_Anomaly_022.JPG" width="300"></td></tr>
    <tr><td>Pipe fryum</td><td><img src="assets/examples/pipe_fryum_Normal_013.JPG" width="300"></td><td><img src="assets/examples/pipe_fryum_Anomaly_003.JPG" width="300"></td></tr>
  </tbody>
</table>

## Complete methodology

```mermaid
flowchart LR
    source[/VisA images/]
    select[Select 15 + 15]
    annotate[/Roboflow annotation/]
    yolo[Train YOLO11]
    validate{Masks valid?}
    full[Full-dataset inference]
    crops[(Object crops)]
    labels[Normal or defective]
    balance[Balance and augment]
    compare[Train three classifiers]
    choose{Best model?}
    app[Gradio application]
    report[/Annotated images and CSV/]

    source --> select
    select --> annotate
    annotate --> yolo
    yolo --> validate
    validate -->|"Yes"| full
    validate -->|"Improve"| yolo
    full --> crops
    crops --> labels
    labels --> balance
    balance --> compare
    compare --> choose
    choose -->|"MobileNetV3"| app
    app --> report

    style annotate fill:#C2E5FF,stroke:#3DADFF
    style yolo fill:#DCCCFF,stroke:#874FFF
    style choose fill:#FFECBD,stroke:#FFC943
    style app fill:#CDF4D3,stroke:#66D575
```

### 1. Create the annotation seed

For each of the 12 product classes, 15 normal and 15 anomaly images were selected:

`12 classes × 30 images = 360 source images`

The reusable selection script is:

```bash
python scripts/select_annotation_seed.py /path/to/VisA \
  --output selected_15_normal_15_abnormal_flat \
  --samples-per-condition 15 \
  --seed 42
```

The script:

- Reads the VisA `Data/Images/Normal` and `Data/Images/Anomaly` folders.
- Uses deterministic random sampling.
- Copies all selected images into one flat folder for Roboflow upload.
- Refuses to overwrite a non-empty output directory.

### 2. Annotate and export from Roboflow

The 360 images were uploaded to Roboflow and annotated with object-level segmentation polygons for all 12 classes.

The recorded Roboflow preprocessing and augmentation were:

- EXIF auto-orientation.
- Stretch resize to 640 × 640.
- Horizontal flip with 50% probability.
- Vertical flip with 50% probability.
- Rotation between −15° and +15°.
- Brightness adjustment between −15% and +15%.

The local YOLO-format export contains:

| Split | Images | Label files |
|---|---:|---:|
| Train | 1,058 | 1,058 |
| Validation | 98 | 98 |
| Test | 50 | 50 |
| **Total** | **1,206** | **1,206** |

Dataset project: [VisA Object Segmentation on Roboflow Universe](https://universe.roboflow.com/deeps-workspace-6tscj/visa_object_segmentation/dataset/3)

### 3. Train and evaluate YOLO11 segmentation

YOLO11 was trained to segment object instances and predict the product category. The best model was first tested on a small sample and visually checked before running it across the full VisA dataset.

The production checkpoint included in this repository is:

```text
models/yolo11_seg_best.pt
```

#### Final segmentation metrics

| Metric | Teacher | Pseudo-finetuned |
|---|---:|---:|
| Box precision | 0.9703 | **0.9903** |
| Box recall | 1.0000 | **1.0000** |
| Box mAP@50 | 0.9950 | **0.9950** |
| Box mAP@50:95 | 0.9933 | **0.9949** |
| Mask precision | 0.9703 | **0.9903** |
| Mask recall | 1.0000 | **1.0000** |
| Mask mAP@50 | 0.9950 | **0.9950** |
| Mask mAP@50:95 | **0.9914** | 0.9907 |
| Pixel IoU, micro | **0.98085** | 0.98085 |
| Pixel Dice, micro | **0.99033** | 0.99033 |
| Pixel IoU, macro | 0.98096 | **0.98112** |
| Boundary F1, macro | 0.98255 | **0.98263** |

The pseudo-label stage used:

- 1,058 ground-truth training images.
- 10,182 high-confidence pseudo-labeled training images.
- 29,849 pseudo masks.
- A pseudo-label confidence threshold of 0.90.
- Leakage-safe exclusion of supervised validation and test sources.

The final model improves precision by approximately **2.0 percentage points** and box mAP@50:95 by approximately **0.15 percentage points**. The mask mAP@50:95 change is slightly negative, so the repository reports all metrics rather than presenting pseudo-labeling as a universal improvement.

![YOLO11 training results](assets/plots/segmentation/results.png)

<p>
  <img src="assets/plots/segmentation/confusion_matrix_normalized.png" width="49%" alt="Normalized YOLO confusion matrix">
  <img src="assets/plots/segmentation/MaskPR_curve.png" width="49%" alt="YOLO mask precision recall curve">
</p>

### 4. Segment the full dataset and generate object crops

The validated YOLO11 model was run over the full VisA image collection. For every detection:

1. The predicted polygon was converted to a binary mask.
2. Background pixels were removed.
3. The masked object was cropped using its bounding box.
4. Source product class, confidence, condition, object ID, and crop path were saved.
5. The source condition and defect-mask overlap were used to assign `normal` or `defective`.

This stage generated the object-level dataset consumed by the classifiers. The original run stored **34,111 valid crops** before defective-class augmentation:

- 32,890 normal crops.
- 1,221 defective crops.

Real segmentation visualizations are included in [`assets/segmentation_examples`](assets/segmentation_examples).

### 5. Balance the classifier dataset

Defects are naturally rare, so the original object-crop dataset was highly imbalanced. Only defective crops were augmented; originals were preserved.

Augmentations included:

- Horizontal, vertical, or combined flips.
- Rotation up to ±15°.
- Scale variation of ±5%.
- Brightness variation of ±15%.
- Contrast and saturation variation of ±10%.
- Low-amplitude foreground-only Gaussian noise.

The balancing process generated **5,608 additional defective crops**, producing **39,719 total object crops**:

| Crop type | Count |
|---|---:|
| Original normal | 32,890 |
| Original defective | 1,221 |
| Augmented defective | 5,608 |
| **Total** | **39,719** |

All augmented children retained their parent source-image group, preventing related samples from leaking across train, validation, and test splits.

| Split | Normal | Defective | Total |
|---|---:|---:|---:|
| Train | 21,046 | 4,393 | 25,439 |
| Validation | 5,273 | 194 | 5,467 |
| Test | 6,571 | 244 | 6,815 |

### 6. Train class-conditioned defect classifiers

Each segmented crop has two inputs:

- The RGB object crop.
- The YOLO-predicted product class ID.

```mermaid
flowchart LR
    crop[/Masked object crop/]
    classId[/YOLO class ID/]
    backbone[Visual backbone]
    embedding[Class embedding]
    attention[Spatial attention]
    film[Feature conditioning]
    merge[Feature fusion]
    probability[/Defect probability/]

    crop --> backbone
    classId --> embedding
    backbone --> attention
    embedding --> attention
    backbone --> film
    embedding --> film
    attention --> merge
    film --> merge
    merge --> probability

    style embedding fill:#C2E5FF,stroke:#3DADFF
    style attention fill:#DCCCFF,stroke:#874FFF
    style film fill:#DCCCFF,stroke:#874FFF
    style probability fill:#CDF4D3,stroke:#66D575
```

The class embedding performs two functions:

- **Class-guided spatial attention** helps the network focus on product-specific regions.
- **FiLM-style conditioning** adjusts pooled visual features using the expected product class.

The final scalar output is converted to a defect probability with a sigmoid. The selected MobileNetV3 threshold, learned from validation predictions, is **0.50927734375**.

#### MobileNetV3 confusion matrix counts

| | Predicted normal | Predicted defective |
|---|---:|---:|
| Actual normal | **6,563** | 8 |
| Actual defective | 30 | **214** |

#### MobileNetV3 per-class highlights

- Perfect test classification for `pipe_fryum`.
- F1 ≥ 0.92 for fryum, macaroni1, macaroni2, PCB1, PCB2, and PCB4.
- `pcb3` is the weakest category, with 50% defect recall, and is the clearest target for future data collection.

<p>
  <img src="assets/plots/classifiers/mobilenet_v3_training_curves.png" width="49%" alt="MobileNetV3 training curves">
  <img src="assets/plots/classifiers/mobilenet_v3_test_evaluation.png" width="49%" alt="MobileNetV3 test evaluation">
</p>

ResNet18 and EfficientNet-B0 plots are retained in [`assets/plots/classifiers`](assets/plots/classifiers) for direct comparison.

### 7. Production inference

```mermaid
flowchart LR
    zip[/ZIP upload/]
    decode[Decode image]
    segment[YOLO11 segment]
    objects{Objects found?}
    mask[Create masked crops]
    classify[Batch MobileNetV3]
    decision{Any defect?}
    normal[/Normal result/]
    defective[/Defective result/]
    unresolved[/No-object result/]
    csv[(CSV report)]

    zip --> decode
    decode --> segment
    segment --> objects
    objects -->|"No"| unresolved
    objects -->|"Yes"| mask
    mask --> classify
    classify --> decision
    decision -->|"No"| normal
    decision -->|"Yes"| defective
    normal --> csv
    defective --> csv
    unresolved --> csv

    style segment fill:#DCCCFF,stroke:#874FFF
    style classify fill:#C2E5FF,stroke:#3DADFF
    style normal fill:#CDF4D3,stroke:#66D575
    style defective fill:#FFCDC2,stroke:#FF7556
    style unresolved fill:#FFECBD,stroke:#FFC943
```

The Gradio application:

- Loads both checkpoints once.
- Warms up YOLO11 and MobileNetV3 before timing.
- Accepts a ZIP containing up to 10,000 supported images.
- Streams status, the current annotated image, a results table, and cumulative timing.
- Draws normal objects in green and defective objects in red.
- Exports a final per-image CSV report.
- Validates archive type, image size, total uncompressed size, and supported formats.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RIDAC-Industrial-Defect-Inspection.git
cd RIDAC-Industrial-Defect-Inspection
```

### 2. Create an environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU inference, install the PyTorch build matching your CUDA environment before installing the remaining requirements.

### 3. Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

Optional arguments:

```bash
python app.py \
  --host 0.0.0.0 \
  --port 7860 \
  --yolo models/yolo11_seg_best.pt \
  --mobilenet models/mobilenet_v3_best.pt
```

To create a temporary public Gradio link:

```bash
python app.py --share
```

## Input and output

### Input

Upload one ZIP file containing images in any nested folder structure.

Supported extensions:

```text
.jpg .jpeg .png .bmp .webp .tif .tiff
```

### Output

The application provides:

- A live annotated preview.
- Image-level `NORMAL`, `DEFECTIVE`, `NO OBJECTS`, or `ERROR` status.
- Detected-object and defective-object counts.
- Predicted product classes.
- Maximum defect probability.
- YOLO, MobileNetV3, and combined inference time.
- A downloadable CSV report.

## Repository structure

```text
RIDAC-Industrial-Defect-Inspection/
├── app.py
├── models/
│   ├── yolo11_seg_best.pt
│   └── mobilenet_v3_best.pt
├── src/
│   └── models.py
├── scripts/
│   ├── select_annotation_seed.py
│   └── training/
├── notebooks/
├── results/
│   ├── segmentation/
│   └── classification/
├── assets/
│   ├── examples/
│   ├── segmentation_examples/
│   ├── plots/
│   └── screenshots/
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducing the research pipeline

Retraining is optional and is **not** needed for application use. The original completed code is retained for transparency:

| Stage | File |
|---|---|
| Select annotation seed | `scripts/select_annotation_seed.py` |
| Initial YOLO11 training | `notebooks/01_train_segmentation.ipynb` |
| Segmentation evaluation and crop extraction | `notebooks/02_evaluate_segment_and_extract_crops.ipynb` |
| Balance the crop dataset | `notebooks/03_balance_classifier_dataset.ipynb` |
| ResNet baseline | `notebooks/04_train_resnet_baseline.ipynb` |
| MobileNetV3 and EfficientNet-B0 | `notebooks/05_train_mobilenet_efficientnet.ipynb` |
| Pseudo-label fine-tuning | `notebooks/06_pseudo_label_finetuning.ipynb` |
| Scripted pseudo-label workflow | `scripts/training/pseudo_label_segmentation_retrain.py` |
| Scripted classifier comparison | `scripts/training/train_lightweight_defect_models.py` |

The retained scripts reference the original local experiment directory structure. Update their dataset paths before attempting a full reproduction.

## Timing methodology

The application uses `time.perf_counter()` and synchronizes CUDA before and after timed GPU work.

The reported values mean:

- **YOLO11**: segmentation prediction time.
- **MobileNetV3**: crop preprocessing, batching, transfer, model forward pass, and sigmoid conversion.
- **Combined inference**: segmentation, crop construction, classification, and synchronization.
- **Average**: combined inference divided by processed images.
- **Throughput**: `1000 / average_ms`.

Excluded:

- Model loading.
- One-time warm-up.
- ZIP reading.
- Image decompression before the pipeline timer.
- Browser rendering and CSV download time.

The component times do not have to sum exactly to combined inference because crop extraction, Python orchestration, annotation preparation, and synchronization contribute additional overhead.

## Limitations

- RIDAC currently supports the 12 product classes used during YOLO and classifier training.
- A missed YOLO object cannot be recovered by the classifier.
- The classifier depends on the YOLO-predicted product class.
- Rare-defect recall remains the most important improvement target, particularly for PCB3.
- The displayed 6.07 images/s benchmark is a measured project result, not a guaranteed speed on other systems.
- Real production deployment should add process monitoring, confidence calibration, drift detection, and human review for uncertain cases.

## Future work

- TensorRT or ONNX export for lower classifier latency.
- Dynamic batching across multiple source images.
- More PCB3 defect samples and hard-negative mining.
- Product-specific decision thresholds.
- Confidence-calibrated reject/inspection queues.
- Camera or video-stream ingestion.
- Model and dataset drift dashboards.
- Integration with PLC, MES, or manufacturing quality-management systems.

## Dataset citation and licensing

### Required citation

If you use RIDAC, the included example images, or the VisA-derived data preparation workflow in academic work, cite the original VisA publication:

> Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, “SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation,” in *Computer Vision – ECCV 2022*, Lecture Notes in Computer Science, vol. 13690, Springer, Cham, 2022, pp. 392–408. doi: [10.1007/978-3-031-20056-4_23](https://doi.org/10.1007/978-3-031-20056-4_23).

When referring specifically to the downloaded dataset, the AWS Registry of Open Data additionally recommends identifying the access source and date:

> Visual Anomaly (VisA) Dataset, AWS Registry of Open Data, accessed July 16, 2026. [https://registry.opendata.aws/visa/](https://registry.opendata.aws/visa/)

### BibTeX

```bibtex
@inproceedings{zou2022spot,
  author    = {Zou, Yang and Jeong, Jongheon and Pemula, Latha and
               Zhang, Dongqing and Dabeer, Onkar},
  title     = {{SPot-the-Difference} Self-Supervised Pre-training for
               Anomaly Detection and Segmentation},
  booktitle = {Computer Vision -- ECCV 2022},
  series    = {Lecture Notes in Computer Science},
  volume    = {13690},
  pages     = {392--408},
  publisher = {Springer},
  address   = {Cham},
  year      = {2022},
  doi       = {10.1007/978-3-031-20056-4_23},
  url       = {https://doi.org/10.1007/978-3-031-20056-4_23}
}

@misc{visa_dataset,
  author       = {{Amazon Web Services}},
  title        = {Visual Anomaly ({VisA}) Dataset},
  howpublished = {Registry of Open Data on AWS},
  url          = {https://registry.opendata.aws/visa/},
  note         = {Accessed: 2026-07-16}
}
```

### Licensing and redistribution

- The original Visual Anomaly (VisA) dataset is distributed through the AWS Registry of Open Data under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
- The project-specific Roboflow export is also marked **CC BY 4.0** and is derived from VisA source images.
- CC BY 4.0 requires appropriate attribution, a link to the license, and an indication of whether changes were made.
- Dataset licensing is separate from this repository’s source-code license.
- The full VisA dataset is not redistributed in this repository. Only a small number of attributed examples, trained checkpoints, plots, and derived project artifacts are included.
- Anyone redistributing additional VisA images or derived annotations should preserve the original attribution, identify the derivative nature of the work, and comply with the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
- RIDAC source code is licensed under the [MIT License](LICENSE).

## Author

**Deep Manish Bhatt**

- GitHub: [deepmbhatt](https://github.com/deepmbhatt)
- Email: deepmanishbhatt@gmail.com

Contributions, reproducibility improvements, deployment optimizations, and additional industrial datasets are welcome.
