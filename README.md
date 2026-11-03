# AI Image Intelligence

A local Mac-friendly computer vision + business analytics starter project.

## What is included

- Pretrained YOLO object detection for supported classes
- Multi-image Streamlit dashboard
- Bounding-box annotations, confidence scores and category counts
- Bar/pie charts and CSV exports
- Optional PostgreSQL persistence
- Custom YOLO fine-tuning script
- Held-out dataset evaluation script

## Important limitations

The default `yolo11n.pt` checkpoint is a pretrained baseline, not a model trained for every possible object. It detects classes represented in its training dataset. Specific stationery classes such as pen/pencil and wildlife species such as tiger/lion/deer may need custom labeled images and fine-tuning. A text prompt or a high confidence score does not guarantee correctness.

Do not infer whether a person is a boy or girl from appearance. The app reports detected people as `person`; any demographic analysis should use reliable, consent-based verified labels.

## 1. Create environment (macOS)

```bash
cd ai-image-intelligence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On first run, Ultralytics may download pretrained weights; internet access is needed. If installation fails, use a supported Python version for the installed PyTorch/Ultralytics packages and install again in a fresh virtual environment.

## 2. Run the dashboard

```bash
streamlit run app.py
```

Choose `auto` for device selection. On Apple Silicon, the app uses PyTorch MPS if available; otherwise it uses CPU. If MPS produces an unsupported-operation error, choose `cpu` in the sidebar.

## 3. Test the pretrained baseline

Upload several images from different settings:
- a person or group photo
- animals
- a desk with stationery
- objects under different lighting and angles

Write down missed detections, wrong classes, duplicate detections and count errors. Do not assume the model supports every object category.

## 4. Custom training dataset

Create this layout:

```text
data/stationery/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Every image must have a matching YOLO `.txt` label file. Each label row is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image width/height. Use CVAT or Roboflow to annotate bounding boxes, export in Ultralytics YOLO detection format, and check that the YAML class IDs match your annotation classes. The provided `data/stationery.yaml` is an example and must be adapted to your real dataset.

Split by source/photo session where possible, not by making near-duplicate copies across train and test. Keep the test split untouched until final evaluation.

## 5. Train on Mac

Start conservatively on an M2 Mac:

```bash
python train.py --data data/stationery.yaml --model yolo11n.pt --epochs 50 --imgsz 640 --batch 4 --device auto --name stationery_v1
```

If memory pressure is high, lower `--batch` to `2` or `--imgsz` to `512`. If MPS fails, rerun with `--device cpu` (slower). Training time depends on dataset size, memory, package versions and class complexity.

Expected checkpoint after a successful run:

```text
runs/stationery_v1/weights/best.pt
```

## 6. Evaluate on held-out test data

```bash
python evaluate.py --model runs/stationery_v1/weights/best.pt --data data/stationery.yaml --split test --device cpu
```

Report precision, recall, mAP50, mAP50-95 and count MAE on a manually verified set. The included evaluation script reports standard detection metrics; counting MAE should be computed from paired ground-truth and predicted counts for the same test images.

## 7. Use your custom checkpoint

Run the app, then set the Model checkpoint field to:

```text
runs/stationery_v1/weights/best.pt
```

Or set the environment variable before launch:

```bash
export AI_IMAGE_MODEL="runs/stationery_v1/weights/best.pt"
streamlit run app.py
```

The custom checkpoint only recognizes the classes in the dataset used to train it. For a broad all-category product, consider separate specialist models or a verified open-vocabulary detector in a later phase.

## 8. Optional PostgreSQL

Create a database in PostgreSQL, then execute `sql/schema.sql` using psql or pgAdmin. Set the connection string in your shell (replace values with your own):

```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/ai_image_intelligence"
streamlit run app.py
```

Do not commit credentials or upload private images without permission. Database persistence is disabled if `DATABASE_URL` is not set.

## 9. Suggested business analytics deliverables

- Total images processed
- Detected object count by class
- Per-image and category distributions
- Average inference time
- Precision, recall, mAP and counting MAE
- Error analysis by object size, lighting and occlusion
- Model version and reproducible experiment settings
