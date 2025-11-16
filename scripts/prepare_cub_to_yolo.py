import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set

from PIL import Image


def read_classes(cub_root: Path) -> Dict[int, str]:
    classes_path = cub_root / "classes.txt"
    class_id_to_name: Dict[int, str] = {}
    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Observed formats:
            # - "1 001.Black_footed_Albatross"
            # - "36 036.Northern_Flicker"
            # We want to map id -> "Northern_Flicker" (without the leading numeric prefix)
            parts = line.split(" ")
            first = parts[0]
            if "." in first:
                cid_str, rest = first.split(".", 1)
                class_id = int(cid_str)
                class_name = rest
            else:
                class_id = int(first)
                class_name = parts[1] if len(parts) > 1 else ""
                # Remove leading "NNN." in class_name if present
                if "." in class_name:
                    prefix, maybe_name = class_name.split(".", 1)
                    if prefix.isdigit() and len(prefix) == 3:
                        class_name = maybe_name
            class_id_to_name[class_id] = class_name
    return class_id_to_name


def read_images(cub_root: Path) -> Dict[int, str]:
    images_path = cub_root / "images.txt"
    image_id_to_relpath: Dict[int, str] = {}
    with open(images_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, relpath = line.split(" ", 1)
            image_id_to_relpath[int(image_id_str)] = relpath
    return image_id_to_relpath


def read_image_class_labels(cub_root: Path) -> Dict[int, int]:
    labels_path = cub_root / "image_class_labels.txt"
    image_id_to_class_id: Dict[int, int] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, class_id_str = line.split(" ")
            image_id_to_class_id[int(image_id_str)] = int(class_id_str)
    return image_id_to_class_id


def read_bounding_boxes(cub_root: Path) -> Dict[int, Tuple[float, float, float, float]]:
    boxes_path = cub_root / "bounding_boxes.txt"
    image_id_to_box: Dict[int, Tuple[float, float, float, float]] = {}
    with open(boxes_path, "r", encoding="utf-8") as f:
        for line in f:
            # image_id, x, y, width, height
            parts = line.strip().split(" ")
            if len(parts) != 5:
                continue
            image_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            image_id_to_box[image_id] = (x, y, w, h)
    return image_id_to_box


def read_train_test_split(cub_root: Path) -> Dict[int, int]:
    split_path = cub_root / "train_test_split.txt"
    image_id_to_is_train: Dict[int, int] = {}
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            image_id = int(parts[0])
            is_train = int(parts[1])
            image_id_to_is_train[image_id] = is_train
    return image_id_to_is_train


def load_selected_class_names(selected_classes_path: Path) -> Set[str]:
    names: Set[str] = set()
    with open(selected_classes_path, "r", encoding="utf-8") as f:
        for raw in f:
            name = raw.strip()
            if not name:
                continue
            # Normalize potential "036. Northern_Flicker" -> "036.Northern_Flicker"
            name = name.replace(". ", ".")
            names.add(name)
    return names


def build_selected_class_ids(class_id_to_name: Dict[int, str], selected_names: Set[str]) -> Set[int]:
    # The selected_names include the numeric prefix, e.g., "036.Northern_Flicker"
    # classes.txt mapping is int -> "Northern_Flicker" (if we parsed without prefix)
    # We reconstruct the "NNN.Name" to match selected_names.
    selected_ids: Set[int] = set()
    for cid, cname in class_id_to_name.items():
        key = f"{cid:03d}.{cname}"
        if key in selected_names:
            selected_ids.add(cid)
    return selected_ids


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare CUB_200_2011 (Piciformes subset) to YOLO format.")
    parser.add_argument("--cub_root", type=str, required=True, help="Path to CUB_200_2011/CUB_200_2011 directory")
    parser.add_argument("--selected_classes", type=str, required=True, help="Path to data/species/cub_selected_classes.txt")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory for YOLO dataset")
    args = parser.parse_args()

    cub_root = Path(args.cub_root)
    selected_path = Path(args.selected_classes)
    out_root = Path(args.out_root)

    class_id_to_name = read_classes(cub_root)
    image_id_to_relpath = read_images(cub_root)
    image_id_to_class_id = read_image_class_labels(cub_root)
    image_id_to_box = read_bounding_boxes(cub_root)
    image_id_to_is_train = read_train_test_split(cub_root)
    selected_names = load_selected_class_names(selected_path)
    selected_class_ids = build_selected_class_ids(class_id_to_name, selected_names)

    if not selected_class_ids:
        raise ValueError("No selected classes matched. Check selected_classes file formatting.")

    # Make local class id mapping for YOLO (0..N-1)
    selected_ids_sorted = sorted(list(selected_class_ids))
    class_id_to_local: Dict[int, int] = {cid: idx for idx, cid in enumerate(selected_ids_sorted)}

    # Prepare folders
    images_train = out_root / "images" / "train"
    images_val = out_root / "images" / "val"
    labels_train = out_root / "labels" / "train"
    labels_val = out_root / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        ensure_dir(d)

    def process_image(image_id: int, is_train: bool):
        class_id = image_id_to_class_id.get(image_id)
        if class_id not in selected_class_ids:
            return
        rel = image_id_to_relpath[image_id]
        src_img = cub_root / "images" / rel
        if not src_img.exists():
            return
        # Determine dst
        img_dst_dir = images_train if is_train else images_val
        lbl_dst_dir = labels_train if is_train else labels_val
        dst_img_path = img_dst_dir / Path(rel).name
        shutil.copy2(src_img, dst_img_path)

        # Bounding box normalization requires image size
        with Image.open(dst_img_path) as im:
            w_img, h_img = im.size

        x, y, w, h = image_id_to_box[image_id]  # top-left x,y, width, height (absolute)
        # Convert to YOLO: cx, cy, w, h normalized
        cx = (x + w / 2.0) / w_img
        cy = (y + h / 2.0) / h_img
        nw = w / w_img
        nh = h / h_img

        local_cid = class_id_to_local[class_id]
        dst_lbl_path = lbl_dst_dir / (dst_img_path.stem + ".txt")
        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            f.write(f"{local_cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Iterate all images
    for image_id, is_train in image_id_to_is_train.items():
        # CUB uses 1=train, 0=test. We'll map test->val
        process_image(image_id, is_train == 1)

    # Write classes.txt for YOLO order
    names_txt = out_root / "classes.txt"
    with open(names_txt, "w", encoding="utf-8") as f:
        for cid in selected_ids_sorted:
            f.write(f"{class_id_to_name[cid]}\n")

    print("Done. YOLO dataset prepared at:", str(out_root))


if __name__ == "__main__":
    main()

