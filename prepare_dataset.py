import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Folder dataset asli
SOURCE_DIR = "dataset"

# Folder output
OUTPUT_DIR = "dataset_split"

# Format gambar yang didukung
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Rasio split
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Hapus folder lama jika ada
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Buat struktur folder
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

print("Memproses dataset...\n")

for class_dir in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, class_dir)

    if not os.path.isdir(class_path):
        continue

    images = [
        str(img)
        for img in Path(class_path).rglob("*")
        if img.suffix.lower() in IMG_EXTENSIONS
    ]

    if len(images) == 0:
        continue

    # train 70%
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - TRAIN_SIZE),
        random_state=42,
        shuffle=True
    )

    # val 15%, test 15%
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split_name, split_imgs in splits.items():

        split_class_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_dir
        )

        os.makedirs(split_class_dir, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy(
                img_path,
                os.path.join(
                    split_class_dir,
                    os.path.basename(img_path)
                )
            )

    print(
        f"{class_dir:<20} "
        f"Train:{len(train_imgs):>4} "
        f"Val:{len(val_imgs):>4} "
        f"Test:{len(test_imgs):>4}"
    )

print("\nDataset berhasil dibagi!")
print(f"Output: {OUTPUT_DIR}")