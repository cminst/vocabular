"""Plot cached BPE merge-prefix compression measurements."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


TEXT = "#333333"
SPINE = "#333333"
CURVE = "#2F8FD0"


def register_manrope_font(repo_root: Path) -> None:
    font_dirs = [
        repo_root / "FONTS",
        Path.home() / ".local" / "share" / "fonts" / "manrope",
    ]
    for font_dir in font_dirs:
        if font_dir.exists():
            for font in font_manager.findSystemFonts([str(font_dir)]):
                font_manager.fontManager.addfont(font)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Manrope",
            "font.size": 8.0,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def vocab_tick(value: int) -> str:
    return "0" if value == 0 else ("1M" if value == 1_000_000 else f"{value // 1_000}k")


def plot_compression(input_path: Path, output_dir: Path) -> None:
    results = json.loads(input_path.read_text(encoding="utf-8"))
    points = results.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"compression points are missing from {input_path}")
    sizes = [point["vocab_size"] for point in points]
    bytes_per_token = [point["bytes_per_token"] for point in points]

    repo_root = Path(__file__).resolve().parents[1]
    register_manrope_font(repo_root)
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(2.95, 2.08))
    ax.plot(
        sizes,
        bytes_per_token,
        color=CURVE,
        linewidth=1.7,
        marker="o",
        markersize=2.1,
        markeredgewidth=0,
    )
    ticks = list(range(0, 1_000_001, 100_000))
    ax.set_xlim(0, 1_000_000)
    ax.set_xticks(ticks, [vocab_tick(tick) for tick in ticks])
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Bytes per token")
    ax.margins(y=0.08)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.78)
    ax.tick_params(axis="both", colors=SPINE, length=2.8, width=0.72)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.22, top=0.97)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "compression_vs_vocab_size"
    for extension in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=300)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plot_compression(args.input, args.output_dir)


if __name__ == "__main__":
    main()
