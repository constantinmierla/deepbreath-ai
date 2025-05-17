import os
import cv2
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[2]

SRC_DIR = BASE_DIR / "dataset" / "lung_cancer_dataset"
DST_DIR = BASE_DIR / "dataset" / "preprocessed_dataset"
CSV_PATH = BASE_DIR / "data_preprocessing" / "results" / "image_distribution.csv"
IMG_SIZE = (224, 224)


def preprocess():
    distribution = defaultdict(lambda: defaultdict(int))
    total_processed = 0

    for split in ['train', 'valid', 'test']:
        split_path = SRC_DIR / split
        if not split_path.is_dir():
            continue

        for cls in os.listdir(split_path):
            src_folder = split_path / cls
            if not src_folder.is_dir():
                continue

            dst_folder = DST_DIR / split / cls
            dst_folder.mkdir(parents=True, exist_ok=True)

            for fname in os.listdir(src_folder):
                fpath = src_folder / fname
                if not fpath.is_file():
                    continue

                img = cv2.imread(str(fpath))
                if img is None:
                    continue

                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0

                out_path = dst_folder / fname
                cv2.imwrite(str(out_path), (img * 255).astype("uint8"))

                distribution[split][cls] += 1
                total_processed += 1

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # fisier csv cu distributia datelor
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'class', 'num_images'])

        for split, classes in distribution.items():
            for cls, count in classes.items():
                writer.writerow([split, cls, count])

    # statistici finale
    print(f"Total imagini procesate: {total_processed}")
    print("Distributie pe seturi:")
    for split in distribution:
        total = sum(distribution[split].values())
        print(f" - {split}: {total} imagini")


if __name__ == '__main__':
    preprocess()
