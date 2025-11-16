import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on Piciformes (CUB subset).")
    parser.add_argument("--data", type=str, default="datasets/piciformes_cub.yaml", help="Dataset YAML path")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (e.g., yolo11n.pt, yolo11s.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--name", type=str, default="piciformes_cub_yolo11n", help="Run name")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
    args = parser.parse_args()

    # Ensure dataset exists
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml.resolve()}")

    # Load base model and train
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        project=args.project,
        pretrained=True,
        verbose=True,
        exist_ok=True
    )
    print("Training finished. Best weights:", results.best)


if __name__ == "__main__":
    main()

