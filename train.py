import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained YOLO detector.")
    parser.add_argument("--data", default="data/stationery.yaml", help="Dataset YAML path")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--name", default="stationery_v1")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}. Create it and label your images first."
        )

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if args.device == "mps":
        if not mps_available:
            raise RuntimeError("PyTorch MPS is not available in this environment.")
        device = "mps"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "mps" if mps_available else "cpu"

    print(f"Training device: {device}")
    print("Important: your dataset must contain valid YOLO bounding-box labels.")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2,
        project="runs",
        name=args.name,
        patience=10,
        plots=True,
        seed=42,
    )
    print(f"Training finished. Check runs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
