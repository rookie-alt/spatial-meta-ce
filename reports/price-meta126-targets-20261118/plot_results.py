"""Plot only experiment126 summary values; no synthetic observations."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Arial Unicode MS"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(12.6, 5.4), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    colors = {"P": "#355E5A", "E": "#BE9A61", "M": "#A74E47"}
    groups = [("imdb", "stats", "ergastf1", "genome"), ("stats", "ergastf1", "genome")]
    titles = ["四库零样本（K0）", "三库少样本（K20）"]
    for panel, (ax, databases) in enumerate(zip(axes, groups)):
        x = np.arange(len(databases))
        k = "0" if panel == 0 else "20"
        for offset, arm in zip((-0.22, 0, 0.22), ("P", "E", "M")):
            values = [summary["per_database"][d][arm]["q95"][k] for d in databases]
            ax.bar(
                x + offset,
                values,
                width=0.20,
                color=colors[arm],
                label={"P": "P 官方 PRICE", "E": "E 普通继续训练", "M": "M 元学习"}[
                    arm
                ],
                zorder=3,
            )
            for location, value in zip(x + offset, values):
                ax.annotate(
                    f"{value:.1f}",
                    (location, value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        if panel == 1:
            values = [summary["per_database"][d]["P"]["q95"]["0"] for d in databases]
            ax.scatter(
                x + 0.38,
                values,
                marker="D",
                facecolors="white",
                edgecolors="#263D3B",
                s=45,
                zorder=5,
                label="P 原始 K0",
            )
        ax.set_xticks(
            x,
            [
                d.upper()
                if d in ("imdb", "stats")
                else {"ergastf1": "ErgastF1", "genome": "Genome"}[d]
                for d in databases
            ],
        )
        ax.set_yscale("log")
        ax.set_ylim(8, 160)
        ax.set_yticks([10, 20, 40, 80, 160], [10, 20, 40, 80, 160])
        ax.set_ylabel("unique-query Q95（越低越好，对数轴）")
        ax.set_title(titles[panel], loc="left", fontsize=14, fontweight="bold", pad=15)
        ax.grid(axis="y", color="#D9DFDD", linewidth=0.7, zorder=0)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle(
        "官方预训练 PRICE 加元学习：目标库固定评估",
        x=0.07,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.01,
        "E/M 显示三条训练流均值；K20 先平均五组支持结果，再平均训练流。IMDB 缺独立支持缓存，仅有 K0。",
        fontsize=10,
        color="#4D5855",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=3)
    fig.savefig(ROOT / "target-q95.png", dpi=180, facecolor="white")
    fig.savefig(ROOT / "target-q95.svg", facecolor="white")
    svg_path = ROOT / "target-q95.svg"
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    print(ROOT / "target-q95.png")


if __name__ == "__main__":
    main()
