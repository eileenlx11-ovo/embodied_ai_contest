import argparse
import csv
import re
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    errors = []
    row_count = 0

    with open(args.csv, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            row_count += 1
            if len(row) != 2:
                errors.append(f"Row {i+1}: expected 2 columns, got {len(row)}")
                continue

            filename = row[0].strip()
            class_name = row[1].strip()

            if not filename:
                errors.append(f"Row {i+1}: empty filename")

            if not re.match(r"^\d{4}$", class_name):
                errors.append(f"Row {i+1}: class '{class_name}' is not 4-digit number")

    if errors:
        print(f"FAILED: {len(errors)} errors found in {row_count} rows:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)
    else:
        print(f"PASSED: {row_count} rows, all valid (filename + 4-digit class)")


if __name__ == "__main__":
    main()
