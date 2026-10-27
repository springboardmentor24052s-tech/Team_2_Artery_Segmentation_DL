"""
ArteryAI baseline segmentation: binary U-Net for coronary anatomy masks.

Expected project layout:
ArteryAI/
  dataset/syntax/{train,val,test}/images/
  dataset/syntax/{train,val,test}/annotations/{train,val,test}.json
  training/unet_baseline.py

Target: union of anatomical polygon categories in dataset/syntax, excluding
the category named "stenosis". This is a binary anatomy segmentation baseline,
not a stenosis classifier/detector.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def polygon_mask(image_info: dict, annotations: list, categories: dict) -> Image.Image:
    """Rasterize all non-stenosis COCO polygon annotations into a binary mask."""
    width, height = int(image_info["width"]), int(image_info["height"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations:
        category_name = str(categories.get(ann.get("category_id"), "")).strip().lower()
        if category_name == "stenosis":
            continue
        segmentation = ann.get("segmentation", [])
        # This dataset uses polygon lists; RLE dictionaries are not silently decoded.
        if not isinstance(segmentation, list):
            continue
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1]))
                      for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=1)
    return mask


class ArcadeSyntaxDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, image_size: int = 512):
        self.split = split
        self.image_size = image_size
        split_dir = dataset_root / "syntax" / split
        annotation_path = split_dir / "annotations" / f"{split}.json"
        image_dir = split_dir / "images"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation JSON: {annotation_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {image_dir}")

        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        self.images = data.get("images", [])
        annotations_by_image: Dict[int, list] = {}
        for ann in data.get("annotations", []):
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)
        self.records = []
        for info in self.images:
            path = image_dir / info["file_name"]
            if path.exists():
                self.records.append((info, path, annotations_by_image.get(info["id"], [])))
        if not self.records:
            raise RuntimeError(f"No usable image records found for {split_dir}")
        missing = len(self.images) - len(self.records)
        if missing:
            print(f"Warning: skipped {missing} images missing from {image_dir}")
        print(f"{split}: {len(self.records)} images | "
              f"{sum(len(r[2]) for r in self.records)} annotations")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        info, path, annotations = self.records[index]
        image = Image.open(path).convert("RGB")
        mask = polygon_mask(info, annotations, self.categories)