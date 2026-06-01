from pathlib import Path

dataset = Path("dataset")

for folder in dataset.iterdir():
    if folder.is_dir():
        total = len(list(folder.glob("*.jpg")))
        print(f"{folder.name}: {total}")