#!/usr/bin/env python3
"""Generate report tables and plots from completed CUS-QA experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable


LANG_ORDER = ("cs", "sk", "uk")
LANG_LABELS = {
    "cs": "Czech",
    "sk": "Slovak",
    "uk": "Ukrainian",
    "overall": "Overall",
}

GENERATOR_LABELS = {
    "qwen3_30b_a3b": "Qwen3-30B",
    "llama31_8b": "Llama-8B",
    "tiny_aya_water": "TinyAya-Water",
    "tiny_aya_global": "TinyAya-Global",
}

GENERATOR_ORDER = (
    "Qwen3-30B",
    "Llama-8B",
    "TinyAya-Water",
    "TinyAya-Global",
)

MAIN_SYSTEM_IDS = {
    "oracle_full_dev_qwen3_30b_a3b",
    "oracle_full_dev_llama31_8b",
    "oracle_full_dev_tiny_aya_water",
    "embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top10",
    "embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top5",
    "embedding_rag_llama31_8b_e5_full_dev_retrieval_top10",
    "embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top5",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}"


def decimal(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_tex_tabular(
    path: Path,
    headers: list[str],
    rows: list[list[object]],
    column_spec: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"\\begin{{tabular}}{{{column_spec}}}\n")
        handle.write("\\hline\n")
        handle.write(" & ".join(tex_escape(header) for header in headers) + r" \\" + "\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(" & ".join(tex_escape(value) for value in row) + r" \\" + "\n")
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def ensure_dirs(plots_dir: Path, tables_dir: Path) -> dict[str, Path]:
    dirs = {
        "main": plots_dir / "main_paper",
        "appendix": plots_dir / "appendix",
        "diagnostics": plots_dir / "diagnostics",
        "tables": tables_dir,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def detect_generator(system_id: str) -> str:
    if "qwen3_30b_a3b" in system_id:
        return "Qwen3-30B"
    if "llama31_8b" in system_id:
        return "Llama-8B"
    if "tiny_aya_water" in system_id:
        return "TinyAya-Water"
    if "tiny_aya_global" in system_id or system_id.endswith("tiny_aya"):
        return "TinyAya-Global"
    return system_id


def detect_condition(system_id: str) -> str:
    if system_id.startswith("oracle"):
        return "Oracle"
    if system_id.startswith("embedding"):
        return "Embedding"
    return "Unknown"


def detect_top_k(system_id: str) -> str:
    if "top10" in system_id:
        return "10"
    if "top5" in system_id:
        return "5"
    return ""


def display_system(system_id: str) -> str:
    condition = detect_condition(system_id)
    generator = detect_generator(system_id)
    top_k = detect_top_k(system_id)
    if condition == "Oracle":
        return f"Oracle {generator}"
    if top_k:
        return f"{generator} top-{top_k}"
    return f"{condition} {generator}"


def sort_key(row: dict) -> tuple[int, str]:
    generator = row["generator"]
    try:
        gen_index = GENERATOR_ORDER.index(generator)
    except ValueError:
        gen_index = len(GENERATOR_ORDER)
    condition_rank = 0 if row["condition"] == "Oracle" else 1
    top_k_rank = int(row["top_k"] or 0)
    return (gen_index, condition_rank, f"{top_k_rank:02d}")


def load_retrieval_rows(work_dir: Path) -> list[dict]:
    top5 = read_json(work_dir / "summaries" / "e5_full_dev_retrieval_top5_summary.json")
    top10 = read_json(work_dir / "summaries" / "e5_full_dev_retrieval_top10_summary.json")
    test_summary = work_dir / "summaries" / "e5_full_test_retrieval_top10_summary.json"
    test_counts = read_json(test_summary).get("language_counts", {}) if test_summary.exists() else {}

    rows = []
    for lang in LANG_ORDER:
        top5_lang = top5["language_summary"][lang]
        top10_lang = top10["language_summary"][lang]
        rows.append(
            {
                "language": LANG_LABELS[lang],
                "lang": lang,
                "dev_questions": top5["language_counts"][lang],
                "test_questions": test_counts.get(lang, ""),
                "top5_target_hit_rate": top5_lang["target_title_hit_rate"],
                "top10_target_hit_rate": top10_lang["target_title_hit_rate"],
                "top5_average_top1_score": top5_lang["average_top1_score"],
                "top10_average_top1_score": top10_lang["average_top1_score"],
            }
        )
    rows.append(
        {
            "language": "Overall",
            "lang": "overall",
            "dev_questions": top5["record_count"],
            "test_questions": sum(test_counts.values()) if test_counts else "",
            "top5_target_hit_rate": top5["target_title_hit_rate"],
            "top10_target_hit_rate": top10["target_title_hit_rate"],
            "top5_average_top1_score": "",
            "top10_average_top1_score": "",
        }
    )
    return rows


def load_mprometheus_rows(work_dir: Path) -> list[dict]:
    eval_dir = work_dir / "eval"
    rows = []
    for path in sorted(eval_dir.glob("mprometheus_v2_*_summary.json")):
        if path.name == "mprometheus_v2_dev_summary.json":
            continue
        summary = read_json(path)
        systems = summary.get("systems") or []
        if len(systems) != 1:
            continue
        system = systems[0]
        system_id = system["system_id"]
        per_lang = system.get("per_language", {})
        rows.append(
            {
                "system_id": system_id,
                "system": display_system(system_id),
                "condition": detect_condition(system_id),
                "top_k": detect_top_k(system_id),
                "generator": detect_generator(system_id),
                "accuracy": system["accuracy"],
                "cs_accuracy": per_lang.get("cs", {}).get("accuracy", math.nan),
                "sk_accuracy": per_lang.get("sk", {}).get("accuracy", math.nan),
                "uk_accuracy": per_lang.get("uk", {}).get("accuracy", math.nan),
                "correct": system["correct"],
                "incorrect": system["incorrect"],
                "unknown": system.get("unknown", 0),
                "record_count": system["record_count"],
                "correct_total": f"{system['correct']} / {system['record_count']}",
            }
        )
    return sorted(rows, key=lambda row: row["accuracy"], reverse=True)


def load_chrf_rows(work_dir: Path) -> list[dict]:
    summary = read_json(work_dir / "eval" / "chrf_dev_summary.json")
    rows = []
    for system in summary["systems"]:
        system_id = system["system_id"]
        per_lang = system.get("per_language", {})
        rows.append(
            {
                "system_id": system_id,
                "system": display_system(system_id),
                "condition": detect_condition(system_id),
                "top_k": detect_top_k(system_id),
                "generator": detect_generator(system_id),
                "chrf": system["chrf"],
                "cs_chrf": per_lang.get("cs", {}).get("chrf", math.nan),
                "sk_chrf": per_lang.get("sk", {}).get("chrf", math.nan),
                "uk_chrf": per_lang.get("uk", {}).get("chrf", math.nan),
                "record_count": system["record_count"],
                "empty_answer_count": system.get("empty_answer_count", 0),
            }
        )
    return sorted(rows, key=lambda row: row["chrf"], reverse=True)


def write_retrieval_table(rows: list[dict], tables_dir: Path) -> None:
    fieldnames = [
        "language",
        "lang",
        "dev_questions",
        "test_questions",
        "top5_target_hit_rate",
        "top10_target_hit_rate",
        "top5_average_top1_score",
        "top10_average_top1_score",
    ]
    write_csv(tables_dir / "dataset_retrieval.csv", rows, fieldnames)
    tex_rows = [
        [
            row["language"],
            row["dev_questions"],
            row["test_questions"],
            pct(row["top5_target_hit_rate"]),
            pct(row["top10_target_hit_rate"]),
        ]
        for row in rows
    ]
    write_tex_tabular(
        tables_dir / "dataset_retrieval.tex",
        ["Language", "Dev", "Test", "top-5 hit", "top-10 hit"],
        tex_rows,
        "lrrrr",
    )


def write_mprometheus_tables(rows: list[dict], tables_dir: Path) -> None:
    fieldnames = [
        "system_id",
        "system",
        "condition",
        "top_k",
        "generator",
        "accuracy",
        "cs_accuracy",
        "sk_accuracy",
        "uk_accuracy",
        "correct",
        "incorrect",
        "unknown",
        "record_count",
    ]
    write_csv(tables_dir / "full_mprometheus.csv", rows, fieldnames)
    tex_rows = [
        [
            row["system"],
            decimal(row["accuracy"]),
            decimal(row["cs_accuracy"]),
            decimal(row["sk_accuracy"]),
            decimal(row["uk_accuracy"]),
            row["correct_total"],
        ]
        for row in rows
    ]
    write_tex_tabular(
        tables_dir / "full_mprometheus.tex",
        ["System", "Overall", "cs", "sk", "uk", "Correct"],
        tex_rows,
        "lrrrrr",
    )

    main_rows = [row for row in rows if row["system_id"] in MAIN_SYSTEM_IDS]
    main_rows = sorted(main_rows, key=lambda row: row["accuracy"], reverse=True)
    write_csv(tables_dir / "main_mprometheus.csv", main_rows, fieldnames)
    main_tex_rows = [
        [
            row["system"],
            row["condition"],
            row["top_k"] or "-",
            row["generator"],
            decimal(row["accuracy"]),
            row["correct_total"],
        ]
        for row in main_rows
    ]
    write_tex_tabular(
        tables_dir / "main_mprometheus.tex",
        ["System", "Condition", "k", "Generator", "Accuracy", "Correct"],
        main_tex_rows,
        "lllrrr",
    )


def write_chrf_table(rows: list[dict], tables_dir: Path) -> None:
    fieldnames = [
        "system_id",
        "system",
        "condition",
        "top_k",
        "generator",
        "chrf",
        "cs_chrf",
        "sk_chrf",
        "uk_chrf",
        "record_count",
        "empty_answer_count",
    ]
    write_csv(tables_dir / "full_chrf.csv", rows, fieldnames)
    tex_rows = [
        [
            row["system"],
            decimal(row["chrf"]),
            decimal(row["cs_chrf"]),
            decimal(row["sk_chrf"]),
            decimal(row["uk_chrf"]),
        ]
        for row in rows
    ]
    write_tex_tabular(
        tables_dir / "full_chrf.tex",
        ["System", "Overall", "cs", "sk", "uk"],
        tex_rows,
        "lrrrr",
    )


def write_chrf_vs_mprometheus_table(
    chrf_rows: list[dict],
    mprometheus_rows: list[dict],
    tables_dir: Path,
) -> None:
    chrf_by_id = {row["system_id"]: row for row in chrf_rows}
    chrf_rank = {
        row["system_id"]: rank
        for rank, row in enumerate(
            sorted(chrf_rows, key=lambda row: row["chrf"], reverse=True),
            start=1,
        )
    }
    mp_rank = {
        row["system_id"]: rank
        for rank, row in enumerate(
            sorted(mprometheus_rows, key=lambda row: row["accuracy"], reverse=True),
            start=1,
        )
    }
    rows = []
    for mp_row in sorted(mprometheus_rows, key=lambda row: row["accuracy"], reverse=True):
        chrf = chrf_by_id.get(mp_row["system_id"])
        if not chrf:
            continue
        rows.append(
            {
                "system_id": mp_row["system_id"],
                "system": mp_row["system"],
                "condition": mp_row["condition"],
                "top_k": mp_row["top_k"],
                "generator": mp_row["generator"],
                "chrf": chrf["chrf"],
                "mprometheus_accuracy": mp_row["accuracy"],
                "rank_chrf": chrf_rank[mp_row["system_id"]],
                "rank_mprometheus": mp_rank[mp_row["system_id"]],
                "rank_difference_chrf_minus_mprometheus": (
                    chrf_rank[mp_row["system_id"]] - mp_rank[mp_row["system_id"]]
                ),
            }
        )

    fieldnames = [
        "system_id",
        "system",
        "condition",
        "top_k",
        "generator",
        "chrf",
        "mprometheus_accuracy",
        "rank_chrf",
        "rank_mprometheus",
        "rank_difference_chrf_minus_mprometheus",
    ]
    write_csv(tables_dir / "chrf_vs_mprometheus.csv", rows, fieldnames)
    tex_rows = [
        [
            row["system"],
            row["condition"],
            row["top_k"] or "-",
            row["generator"],
            decimal(row["chrf"]),
            decimal(row["mprometheus_accuracy"]),
            row["rank_chrf"],
            row["rank_mprometheus"],
        ]
        for row in rows
    ]
    write_tex_tabular(
        tables_dir / "chrf_vs_mprometheus.tex",
        ["System", "Condition", "k", "Generator", "chrF", "M-Prom.", "chrF rank", "Judge rank"],
        tex_rows,
        "lllrrrrr",
    )


def embedding_rows_by_generator(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row["condition"] != "Embedding":
            continue
        if row["top_k"] not in {"5", "10"}:
            continue
        grouped.setdefault(row["generator"], {})[row["top_k"]] = row
    return grouped


def oracle_rows_by_generator(rows: list[dict]) -> dict[str, dict]:
    return {
        row["generator"]: row
        for row in rows
        if row["condition"] == "Oracle"
    }


def write_topk_table(rows: list[dict], tables_dir: Path) -> list[dict]:
    grouped = embedding_rows_by_generator(rows)
    ablation_rows = []
    for generator in GENERATOR_ORDER:
        pair = grouped.get(generator, {})
        if "5" not in pair or "10" not in pair:
            continue
        top5 = pair["5"]["accuracy"]
        top10 = pair["10"]["accuracy"]
        ablation_rows.append(
            {
                "generator": generator,
                "top5_accuracy": top5,
                "top10_accuracy": top10,
                "delta_top10_minus_top5": top10 - top5,
            }
        )
    write_csv(
        tables_dir / "topk_ablation.csv",
        ablation_rows,
        ["generator", "top5_accuracy", "top10_accuracy", "delta_top10_minus_top5"],
    )
    tex_rows = [
        [
            row["generator"],
            decimal(row["top5_accuracy"]),
            decimal(row["top10_accuracy"]),
            f"{100 * row['delta_top10_minus_top5']:+.1f} pp",
        ]
        for row in ablation_rows
    ]
    write_tex_tabular(
        tables_dir / "topk_ablation.tex",
        ["Generator", "top-5", "top-10", "Delta"],
        tex_rows,
        "lrrr",
    )
    return ablation_rows


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots. Load the Fox plotting modules, e.g. "
            "`module load nlpl-scipy-ecosystem/01-foss-2024a-Python-3.12.3`."
        ) from exc
    return plt


def save_figure(fig, path_base: Path, formats: Iterable[str]) -> None:
    for fmt in formats:
        out_path = path_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, bbox_inches="tight", dpi=220)


def add_bar_labels(ax, bars, fmt="{:.1f}", scale=1.0, dy=0.01) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height * scale),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_retrieval(rows: list[dict], plots_dir: Path, formats: list[str]) -> None:
    plt = load_matplotlib()
    labels = [row["language"] for row in rows]
    x = list(range(len(labels)))
    width = 0.36
    top5 = [100 * row["top5_target_hit_rate"] for row in rows]
    top10 = [100 * row["top10_target_hit_rate"] for row in rows]

    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    bars1 = ax.bar([i - width / 2 for i in x], top5, width, label="top-5", color="#777777")
    bars2 = ax.bar([i + width / 2 for i in x], top10, width, label="top-10", color="#bbbbbb", edgecolor="#333333")
    add_bar_labels(ax, bars1, fmt="{:.1f}", dy=1.0)
    add_bar_labels(ax, bars2, fmt="{:.1f}", dy=1.0)
    ax.set_ylabel("Target page hit rate (%)")
    ax.set_ylim(0, max(top10) + 12)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    save_figure(fig, plots_dir / "main_paper" / "retrieval_hit_rate_by_language", formats)
    plt.close(fig)


def plot_topk_delta(ablation_rows: list[dict], plots_dir: Path, formats: list[str]) -> None:
    plt = load_matplotlib()
    labels = [row["generator"] for row in ablation_rows]
    deltas = [100 * row["delta_top10_minus_top5"] for row in ablation_rows]
    colors = ["#777777" if value >= 0 else "#cccccc" for value in deltas]

    fig, ax = plt.subplots(figsize=(6.7, 3.2))
    bars = ax.bar(labels, deltas, color=colors, edgecolor="#333333")
    ax.axhline(0, color="#222222", linewidth=0.8)
    for bar, value in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.35 if value >= 0 else -0.55),
            f"{value:+.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_ylim(min(deltas) - 2.5, max(deltas) + 2.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    save_figure(fig, plots_dir / "main_paper" / "topk_delta_by_generator", formats)
    plt.close(fig)


def plot_oracle_gap(rows: list[dict], plots_dir: Path, formats: list[str]) -> None:
    plt = load_matplotlib()
    oracle = oracle_rows_by_generator(rows)
    embedding = embedding_rows_by_generator(rows)
    labels = []
    oracle_scores = []
    best_embedding_scores = []
    for generator in GENERATOR_ORDER:
        if generator not in oracle or generator not in embedding:
            continue
        best_embedding = max(embedding[generator].values(), key=lambda row: row["accuracy"])
        labels.append(generator)
        oracle_scores.append(100 * oracle[generator]["accuracy"])
        best_embedding_scores.append(100 * best_embedding["accuracy"])

    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    bars1 = ax.bar([i - width / 2 for i in x], oracle_scores, width, label="Oracle", color="#666666")
    bars2 = ax.bar([i + width / 2 for i in x], best_embedding_scores, width, label="Embedding", color="#c8c8c8", edgecolor="#333333")
    add_bar_labels(ax, bars1, fmt="{:.1f}", dy=0.8)
    add_bar_labels(ax, bars2, fmt="{:.1f}", dy=0.8)
    ax.set_ylabel("M-Prometheus accuracy (%)")
    ax.set_ylim(65, 100)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    save_figure(fig, plots_dir / "main_paper" / "oracle_vs_embedding_gap", formats)
    plt.close(fig)


def plot_mprometheus_heatmap(rows: list[dict], plots_dir: Path, formats: list[str]) -> None:
    plt = load_matplotlib()
    labels = [row["system"] for row in rows]
    columns = ["Overall", "cs", "sk", "uk"]
    matrix = [
        [
            row["accuracy"],
            row["cs_accuracy"],
            row["sk_accuracy"],
            row["uk_accuracy"],
        ]
        for row in rows
    ]
    fig_height = max(4.0, 0.36 * len(labels))
    fig, ax = plt.subplots(figsize=(6.7, fig_height))
    vmin = 0.70
    vmax = 0.97
    image = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Accuracy")
    ax.tick_params(axis="both", length=0)
    save_figure(fig, plots_dir / "appendix" / "mprometheus_language_heatmap", formats)
    plt.close(fig)


def plot_chrf_vs_mprometheus(
    chrf_rows: list[dict],
    mprometheus_rows: list[dict],
    plots_dir: Path,
    formats: list[str],
) -> None:
    plt = load_matplotlib()
    chrf_by_id = {row["system_id"]: row for row in chrf_rows}
    points = []
    for mp_row in mprometheus_rows:
        chrf = chrf_by_id.get(mp_row["system_id"])
        if not chrf:
            continue
        points.append((chrf["chrf"], mp_row["accuracy"], mp_row["system"]))

    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    ax.scatter([p[0] for p in points], [p[1] for p in points], color="#666666")
    for x_val, y_val, label in points:
        ax.annotate(label, (x_val, y_val), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("chrF")
    ax.set_ylabel("M-Prometheus accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#dddddd", linewidth=0.6)
    save_figure(fig, plots_dir / "appendix" / "chrf_vs_mprometheus", formats)
    plt.close(fig)


def maybe_write_codabench(args: argparse.Namespace, tables_dir: Path, plots_dir: Path) -> None:
    score_path = Path("data/runs/codabench_scores.csv")
    if not args.include_codabench:
        return
    if not score_path.exists():
        print(f"Codabench scores not found at {score_path}; skipping Codabench artifacts.")
        return

    with score_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"Codabench scores file is empty: {score_path}; skipping Codabench artifacts.")
        return
    fieldnames = list(rows[0].keys())
    write_csv(tables_dir / "codabench_test_results.csv", rows, fieldnames)
    write_tex_tabular(
        tables_dir / "codabench_test_results.tex",
        fieldnames,
        [[row.get(field, "") for field in fieldnames] for row in rows],
        "l" * len(fieldnames),
    )

    if "chrf" in fieldnames:
        plt = load_matplotlib()
        labels = [row.get("system", row.get("submission", f"row {i+1}")) for i, row in enumerate(rows)]
        values = [float(row["chrf"]) for row in rows]
        fig, ax = plt.subplots(figsize=(6.7, 3.2))
        x = list(range(len(labels)))
        bars = ax.bar(x, values, color="#777777")
        add_bar_labels(ax, bars, fmt="{:.2f}", dy=max(values) * 0.01 if values else 0.1)
        ax.set_xticks(x, labels, fontsize=8, rotation=12, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Codabench chrF")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        fig.subplots_adjust(bottom=0.26)
        save_figure(fig, plots_dir / "diagnostics" / "codabench_chrf", args.formats)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "CUSQA_WORK",
                f"/cluster/work/projects/ec403/{os.environ.get('USER', 'user')}/cusqa-rag-2026",
            )
        ),
    )
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--tables-dir", type=Path, default=Path("report_acl/tables"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    parser.add_argument("--include-codabench", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", f"/tmp/{os.environ.get('USER', 'cusqa')}-matplotlib")
    dirs = ensure_dirs(args.plots_dir, args.tables_dir)

    retrieval_rows = load_retrieval_rows(args.work_dir)
    mprometheus_rows = load_mprometheus_rows(args.work_dir)
    chrf_rows = load_chrf_rows(args.work_dir)

    write_retrieval_table(retrieval_rows, dirs["tables"])
    write_mprometheus_tables(mprometheus_rows, dirs["tables"])
    write_chrf_table(chrf_rows, dirs["tables"])
    write_chrf_vs_mprometheus_table(chrf_rows, mprometheus_rows, dirs["tables"])
    topk_rows = write_topk_table(mprometheus_rows, dirs["tables"])

    plot_retrieval(retrieval_rows, args.plots_dir, args.formats)
    plot_topk_delta(topk_rows, args.plots_dir, args.formats)
    plot_oracle_gap(mprometheus_rows, args.plots_dir, args.formats)
    plot_mprometheus_heatmap(mprometheus_rows, args.plots_dir, args.formats)
    plot_chrf_vs_mprometheus(chrf_rows, mprometheus_rows, args.plots_dir, args.formats)
    maybe_write_codabench(args, dirs["tables"], args.plots_dir)

    print(f"Wrote tables to {args.tables_dir}")
    print(f"Wrote plots to {args.plots_dir}")
    print(f"Retrieval rows: {len(retrieval_rows)}")
    print(f"M-Prometheus systems: {len(mprometheus_rows)}")
    print(f"chrF systems: {len(chrf_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
