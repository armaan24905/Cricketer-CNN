from pathlib import Path
from PIL import Image
import shutil
import random

# Configuration

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "results" / "train"
TARGET_DIR = BASE_DIR / "results" / "train_small"

IMAGES_PER_CLASS = 10
RANDOM_SEED = 42

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

random.seed(RANDOM_SEED)

# Create Small Dataset

def is_valid_image(image_path: Path) -> bool:
    """Return True if the file is a valid readable image."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except Exception:
        return False


def create_dataset():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_DIR}"
        )

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    class_directories = [
        directory
        for directory in SOURCE_DIR.iterdir()
        if directory.is_dir()
    ]

    class_directories.sort(
        key=lambda directory: directory.name.lower()
    )

    print("=" * 60)
    print("CRICKETER CNN - SMALL DATASET CREATION")
    print("=" * 60)

    print(f"\nSource      : {SOURCE_DIR}")
    print(f"Destination : {TARGET_DIR}")
    print(f"Images/Class: {IMAGES_PER_CLASS}")

    total_copied = 0

    for class_dir in class_directories:

        class_name = class_dir.name

        target_class_dir = TARGET_DIR / class_name

        target_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        image_files = [
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ]

        # Keep only readable/valid images
        valid_images = []

        for image_file in image_files:

            if is_valid_image(image_file):
                valid_images.append(image_file)

        random.shuffle(valid_images)

        selected_images = valid_images[
            :IMAGES_PER_CLASS
        ]

        print(
            f"\n{class_name}: "
            f"{len(valid_images)} valid images → "
            f"{len(selected_images)} selected"
        )

        for image_file in selected_images:

            destination = (
                target_class_dir / image_file.name
            )

            shutil.copy2(
                image_file,
                destination
            )

            total_copied += 1

    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETED")
    print("=" * 60)

    print(f"\nTotal images copied: {total_copied}")
    print(f"Dataset location   : {TARGET_DIR}")

    print("\nOriginal dataset was NOT modified.")


if __name__ == "__main__":
    create_dataset()