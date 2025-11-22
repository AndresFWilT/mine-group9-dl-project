import argparse
import os
import shutil
from pathlib import Path
from typing import Set, List


def load_species(species_file: Path) -> Set[str]:
    names: Set[str] = set()
    with open(species_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names


def main():
    parser = argparse.ArgumentParser(description="Filter birds_train_small to Piciformes species and stage images.")
    parser.add_argument("--src_root", type=str, required=True, help="Path to birds_train_small directory")
    parser.add_argument("--species_file", type=str, required=True, help="Path to data/species/piciformes_birds_train_small.txt")
    parser.add_argument("--out_root", type=str, required=True, help="Output directory for filtered copy (staging)")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    species = load_species(Path(args.species_file))

    out_images = out_root / "images"
    out_manifest = out_root / "manifest.csv"
    out_images.mkdir(parents=True, exist_ok=True)

    # Copy images for selected species and write a manifest for later autolabeling
    with open(out_manifest, "w", encoding="utf-8") as mf:
        mf.write("species,src_path,dst_path\n")
        for sp in sorted(species):
            species_dir = src_root / sp
            if not species_dir.exists():
                # Try to find a directory that endswith species name (dataset may have prefixed ids)
                matched_dirs: List[Path] = [p for p in src_root.glob(f"*{sp}") if p.is_dir()]
                if matched_dirs:
                    species_dir = matched_dirs[0]
                else:
                    continue
            for img_path in species_dir.rglob("*"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                dst_path = out_images / f"{sp}__{img_path.name}"
                shutil.copy2(img_path, dst_path)
                mf.write(f"{sp},{img_path},{dst_path}\n")

    readme = out_root / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "# Staging: birds_train_small filtered (Piciformes)\n\n"
            "- Images copied here are ready for autolabeling (no boxes yet).\n"
            "- Use a base YOLO model to generate pseudo-labels and then perform a light QA.\n"
        )

    print("Done. Filtered images staged at:", str(out_images))


if __name__ == "__main__":
    main()


