"""Plot Tiny Aya Water raw-vs-reranked retrieval results."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/{os.environ.get('USER', 'cusqa')}-matplotlib")

try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for this plot. On Fox, load the plotting module first: "
        "`module load nlpl-scipy-ecosystem/01-foss-2024a-Python-3.12.3`."
    ) from exc


WORK = Path(
    os.environ.get(
        "CUSQA_WORK",
        f"/cluster/work/projects/ec403/{os.environ.get('USER', 'user')}/cusqa-rag-2026",
    )
)
REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "plots" / "diagnostics"


SYSTEMS = [
    {
        "label": "Raw top-5",
        "kind": "raw",
        "mp_csv": WORK / "eval" / "mprometheus_v2_embedding_tiny_aya_water_top5_summary.csv",
        "chrf_csv": WORK / "eval" / "chrf_dev_summary.csv",
        "chrf_system_id": "embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top5",
    },
    {
        "label": "Raw top-10",
        "kind": "raw",
        "mp_csv": WORK / "eval" / "mprometheus_v2_embedding_tiny_aya_water_top10_summary.csv",
        "chrf_csv": WORK / "eval" / "chrf_dev_summary.csv",
        "chrf_system_id": "embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top10",
    },
    {
        "label": "Reranked top-5",
        "kind": "reranked",
        "mp_csv": WORK / "eval" / "mprometheus_v2_embedding_tiny_aya_water_rerank_bge_top5_summary.csv",
        "chrf_csv": WORK / "eval" / "chrf_rerank_bge_top5_summary.csv",
    },
    {
        "label": "Reranked top-10",
        "kind": "reranked",
        "mp_csv": WORK / "eval" / "mprometheus_v2_embedding_tiny_aya_water_rerank_bge_top10_summary.csv",
        "chrf_csv": WORK / "eval" / "chrf_rerank_bge_top10_summary.csv",
    },
]


def read_single_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return rows[0]


def read_matching_row(path: Path, system_id: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("system_id") == system_id:
                return row
    return None


def collect_rows() -> list[dict]:
    rows = []
    for system in SYSTEMS:
        mp_row = read_single_row(system["mp_csv"])
        if system["kind"] == "raw":
            chrf_row = read_matching_row(system["chrf_csv"], system["chrf_system_id"])
        else:
            chrf_row = read_single_row(system["chrf_csv"])
        if not mp_row and not chrf_row:
            continue
        rows.append(
            {
                "label": system["label"],
                "mp": float(mp_row["accuracy"]) if mp_row else None,
                "unknown": int(mp_row.get("unknown", 0)) if mp_row else None,
                "chrf": float(chrf_row["chrf"]) if chrf_row else None,
                "hit": float(chrf_row["target_title_hit_rate"])
                if chrf_row and chrf_row.get("target_title_hit_rate")
                else None,
            }
        )
    return rows


def annotate_bars(axis, bars, values, fmt):
    for bar, value in zip(bars, values):
        if value is None:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_matplotlib(rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    color_by_label = {
        "Raw top-5": "#bdbdbd",
        "Raw top-10": "#e0e0e0",
        "Reranked top-5": "#666666",
        "Reranked top-10": "#999999",
    }
    colors = [color_by_label.get(label, "#aaaaaa") for label in labels]
    x_positions = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.2), constrained_layout=True)

    mp_values = [row["mp"] for row in rows]
    mp_bars = axes[0].bar(
        x_positions,
        [value or 0.0 for value in mp_values],
        color=colors,
        edgecolor="#222222",
        linewidth=0.5,
    )
    axes[0].set_title("M-Prometheus accuracy")
    axes[0].set_ylim(0.74, 0.87)
    axes[0].set_ylabel("Accuracy")
    annotate_bars(axes[0], mp_bars, mp_values, "{:.3f}")

    chrf_values = [row["chrf"] for row in rows]
    chrf_bars = axes[1].bar(
        x_positions,
        [value or 0.0 for value in chrf_values],
        color=colors,
        edgecolor="#222222",
        linewidth=0.5,
    )
    axes[1].set_title("chrF")
    axes[1].set_ylim(34.5, 40.5)
    axes[1].set_ylabel("chrF")
    annotate_bars(axes[1], chrf_bars, chrf_values, "{:.1f}")

    for axis in axes:
        axis.set_xticks(x_positions, labels, rotation=25, ha="right")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle("Tiny Aya Water improves with BGE reranking", fontsize=11)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        output_path = PLOT_DIR / f"tiny_aya_water_rerank.{suffix}"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        print(output_path)
    plt.close(fig)


def main() -> int:
    rows = collect_rows()
    if not rows:
        raise ValueError("No Tiny Aya reranking result rows found.")
    save_matplotlib(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
