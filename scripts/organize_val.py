import csv, os
from pathlib import Path

flat_val = Path("/root/autodl-tmp/val_extract/ILSVRC/Data/CLS-LOC/val")
out_val = Path("/root/autodl-tmp/imagenet_10p/val")
csv_path = Path("/root/autodl-tmp/val_extract/LOC_val_solution.csv")

out_val.mkdir(parents=True, exist_ok=True)
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

count = 0
for row in rows:
    image_id = row["ImageId"]
    synset = row["PredictionString"].split()[0]
    src = flat_val / f"{image_id}.JPEG"
    if not src.exists():
        continue
    dst_dir = out_val / synset
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        os.rename(str(src), str(dst))
    count += 1

print(f"Val organized: {count} images")
