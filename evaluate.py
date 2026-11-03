import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained detector on held-out data.")
    parser.add_argument("--model", default="runs/stationery_v1/weights/best.pt")
    parser.add_argument("--data", default="data/stationery.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Use cpu or mps")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project="runs",
        name=f"evaluation_{args.split}",
    )
    print("\nEvaluation summary")
    print(f"mAP50:     {float(metrics.box.map50):.4f}")
    print(f"mAP50-95:  {float(metrics.box.map):.4f}")
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall:    {float(metrics.box.mr):.4f}")
    print("See the evaluation output folder for plots and detailed results.")


if __name__ == "__main__":
    main()
