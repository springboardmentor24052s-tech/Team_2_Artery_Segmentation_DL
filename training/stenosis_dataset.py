
import json
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "stenosis"


class StenosisDataset(Dataset):

    def __init__(self, split):

        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")

        self.image_dir = DATASET_ROOT / split / "images"
        annotation_path = (
            DATASET_ROOT / split / "annotations" / f"{split}.json"
        )

        with annotation_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        # Find the category ID for stenosis.
        categories = {
            category["id"]: category["name"]
            for category in data["categories"]
        }

        stenosis_category_ids = {
            category_id
            for category_id, name in categories.items()
            if name == "stenosis"
        }

        # Group stenosis annotations by image ID.
        self.annotations = defaultdict(list)

        for annotation in data["annotations"]:

            if annotation["category_id"] in stenosis_category_ids:
                self.annotations[annotation["image_id"]].append(
                    annotation
                )

        # Keep only images with verified stenosis annotations.
        # Images without annotations are excluded until their
        # negative-label status is confirmed.
        self.images = [
            image
            for image in data["images"]
            if image["id"] in self.annotations
        ]

        self.images.sort(key=lambda image: image["id"])

        print(
            f"{split.upper()}: "
            f"{len(self.images)} annotated images loaded"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_info = self.images[index]

        image_id = image_info["id"]
        image_path = self.image_dir / image_info["file_name"]

        # Load original angiography image.
        image = Image.open(image_path).convert("RGB")

        # Convert image to a PyTorch tensor.
        image = TF.to_tensor(image)

        boxes = []

        for annotation in self.annotations[image_id]:

            x, y, width, height = annotation["bbox"]

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(image_info["width"], x + width)
            y2 = min(image_info["height"], y + height)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        # Faster R-CNN uses label 0 for background.
        # Label 1 represents stenosis.
        labels = torch.ones(
            (len(boxes),),
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": (
                (boxes[:, 2] - boxes[:, 0]) *
                (boxes[:, 3] - boxes[:, 1])
            ),
            "iscrowd": torch.zeros(
                (len(boxes),),
                dtype=torch.int64
            )
        }

        return image, target


def detection_collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":

    for split in ["train", "val", "test"]:

        dataset = StenosisDataset(split)

        image, target = dataset[0]

        print(f"\n{split.upper()} SAMPLE")
        print("Image shape:", image.shape)
        print("Image ID:", target["image_id"].item())
        print("Bounding boxes:", target["boxes"].shape)
        print("Labels:", target["labels"].tolist())
        print("First bounding box:", target["boxes"][0].tolist())
