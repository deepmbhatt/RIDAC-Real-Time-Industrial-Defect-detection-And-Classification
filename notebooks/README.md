# RIDAC notebook guide

The notebooks document the complete research pipeline. They are retained for
reproducibility; the trained checkpoints in `../models/` allow the Gradio
application to run without retraining.

Run notebooks only when reproducing or extending an experiment.

| Order | Notebook | Purpose | Expensive operation |
|---:|---|---|---|
| 1 | `01_train_segmentation.ipynb` | Audit Roboflow labels and train the initial YOLO11 segmentation model | YOLO training |
| 2 | `02_evaluate_segment_and_extract_crops.ipynb` | Review masks, segment the complete VisA dataset, and save object crops | Full-dataset YOLO inference |
| 3 | `03_balance_classifier_dataset.ipynb` | Augment defective crops and establish the grouped ResNet18 baseline | Dataset generation and ResNet training |
| 4 | `04_train_resnet_baseline.ipynb` | Standalone reusable ResNet training and checkpoint-loading workflow | ResNet training |
| 5 | `05_train_mobilenet_efficientnet.ipynb` | Compare MobileNetV3-Large and EfficientNet-B0 on the same split | Two classifier training runs |
| 6 | `06_pseudo_label_finetuning.ipynb` | Generate filtered pseudo masks and fine-tune YOLO11 safely | Pseudo-label generation and YOLO training |

## Recommended reading path

For understanding the project without rerunning it:

1. Read each notebook's introduction and pipeline-position section.
2. Review configuration tables and leakage-prevention notes.
3. Inspect saved outputs and metrics without changing reuse flags.
4. Refer to the repository-level `README.md` for the final comparison and
   production benchmark.

## Before executing

- Create the environment from `../requirements.txt`.
- Restore the original VisA and Roboflow dataset directories expected by the
  historical training scripts.
- Update local path-discovery code if your directory layout differs.
- Keep reuse flags enabled unless retraining is intentional.
- Confirm available disk space before generating crops or pseudo-label data.
- Use a CUDA GPU for practical training and full-dataset inference times.

## Data-leakage rule

Never split individual crops randomly. All crops and augmented descendants from
the same source image must stay in one partition. Validation and test results
must use real, non-augmented samples only.
