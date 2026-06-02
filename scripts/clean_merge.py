"""
Merge C1 + C2 exclusion lists into a single file for training.

Usage:
  python scripts/clean_merge.py \
      --c1 data/exclude_c1.txt \
      --c2 data/exclude_c2.txt \
      --output data/train_exclude.txt
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1", default="data/exclude_c1.txt")
    parser.add_argument("--c2", default="data/exclude_c2.txt")
    parser.add_argument("--output", default="data/train_exclude.txt")
    args = parser.parse_args()

    excluded = set()
    for path in [args.c1, args.c2]:
        p = Path(path)
        if p.exists():
            lines = [
                stripped for line in p.read_text(encoding="utf-8").splitlines()
                if (stripped := line.strip()) and not stripped.startswith("#")
            ]
            excluded.update(lines)
            print(f"  {path}: {len(lines)} entries")
        else:
            print(f"  [ERROR] {path} not found — did the upstream step fail?")
            raise FileNotFoundError(f"Required input missing: {path}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sorted_list = sorted(excluded)
    out.write_text("\n".join(sorted_list) + "\n", encoding="utf-8")
    print(f"\nMerged: {len(sorted_list)} unique samples -> {out}")


if __name__ == "__main__":
    main()
