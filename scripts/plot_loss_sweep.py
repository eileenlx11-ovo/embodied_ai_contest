import argparse
import csv
import json
from pathlib import Path


REQUIRED_RUNS = [
    "sce_a01_b10",
    "sce_a05_b10",
    "sce_a07_b10",
    "gce_q05",
    "gce_q07",
    "gce_q09",
]


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["epoch", "train_loss", "train_acc", "val_top1", "val_top5", "best_val_top1", "lr", "epoch_time_sec"]:
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def summarize_run(run_dir):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None, []

    rows = read_metrics(metrics_path)
    if not rows:
        return None, rows

    summary_path = run_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    first = rows[0]
    last = rows[-1]
    best_val = max((row.get("val_top1", 0.0) for row in rows if row.get("val_top1") != ""), default=0.0)
    return {
        "run_name": run_dir.name,
        "loss": last.get("loss", summary.get("loss", "")),
        "loss_params": last.get("loss_params", json.dumps(summary.get("loss_params", {}), sort_keys=True)),
        "epochs": len(rows),
        "initial_train_loss": first["train_loss"],
        "final_train_loss": last["train_loss"],
        "final_train_acc": last["train_acc"],
        "best_val_top1": best_val,
        "final_val_top1": last.get("val_top1", ""),
        "final_val_top5": last.get("val_top5", ""),
        "has_nan_or_inf": summary.get("has_nan_or_inf", ""),
        "train_loss_decreased": last["train_loss"] < first["train_loss"],
    }, rows


def write_summary_csv(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "week2_loss_sweep.csv"
    fieldnames = [
        "run_name", "loss", "loss_params", "epochs", "initial_train_loss",
        "final_train_loss", "final_train_acc", "best_val_top1", "final_val_top1",
        "final_val_top5", "has_nan_or_inf", "train_loss_decreased",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_curve_plot(all_metrics, output_dir):
    output_path = output_dir / "week2_loss_curves.png"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not all_metrics:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run_name, rows in all_metrics.items():
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label=run_name)
        axes[1].plot(epochs, [row["train_acc"] for row in rows], marker="o", label=run_name)

    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Train accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("logs"), help="包含各 run 子目录的目录")
    parser.add_argument("--output", type=Path, default=Path("docs"), help="输出 docs 目录")
    parser.add_argument("--runs", nargs="*", default=REQUIRED_RUNS, help="需要汇总的 run_name 列表")
    args = parser.parse_args()

    summaries = []
    all_metrics = {}
    missing = []
    for run_name in args.runs:
        summary, rows = summarize_run(args.input / run_name)
        if summary is None:
            missing.append(run_name)
            continue
        summaries.append(summary)
        all_metrics[run_name] = rows

    summary_path = write_summary_csv(summaries, args.output)
    plot_path = write_curve_plot(all_metrics, args.output)

    print(f"Wrote {summary_path}")
    if plot_path:
        print(f"Wrote {plot_path}")
    else:
        print("Skipped curve plot because matplotlib is unavailable or no metrics were found.")
    if missing:
        print("Missing runs: " + ", ".join(missing))


if __name__ == "__main__":
    main()
