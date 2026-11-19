"""Layout body for candidate B.

Kept separate from b_figure.py so the composition can be reviewed and edited
without touching the build plumbing. All coordinates are in the shared
1800 x 1200 viewBox space; body text is never below 24 units and key labels
are 28-32 units.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import b_figure_lib as lib
from b_figure import REUSE

INK = lib.INK
BLACK = lib.BLACK
BLUE = lib.BLUE
BLUE_FILL = lib.BLUE_FILL
PINK = lib.PINK
PINK_FILL = lib.PINK_FILL
PEACH_FILL = lib.PEACH_FILL
MINT = lib.MINT
MINT_FILL = lib.MINT_FILL
PALE_MINT = lib.PALE_MINT
GREY_FILL = lib.GREY_FILL
FONT = "NotoSansSC"

BODY = 24
KEY = 30
SECTION = 28


def draw(canvas, root, tag_fn, bucket, place, place_pair, cell_row):
    """Compose the figure; returns the collected part buckets."""
    # ================================================================ title ==
    bucket("title-band", [
        canvas.text(40, 66, "面向空间漂移的跨数据库基数估计元学习框架",
                    size=42, weight="700"),
        canvas.text(40, 110,
                    "源库元训练 → 共享参数密封 → 目标库少样本适应（协议设计示意，非实测结果）",
                    size=BODY, fill=INK),
        canvas.line(40, 132, 1760, 132, stroke=BLACK, stroke_width=2.5),
    ])

    legend_x = 1112
    legend_keys = []
    for fill, label in ((BLUE_FILL, "源库"), (PEACH_FILL, "目标库"),
                        (GREY_FILL, "冻结参数"), (PINK_FILL, "支持集"),
                        (MINT_FILL, "查询集")):
        legend_keys.extend(canvas.card(legend_x, 28, 30, 24, fill=fill, radius=4))
        legend_keys.append(canvas.text(legend_x + 38, 49, label, size=BODY))
        legend_x += 38 + lib.measure_text_width(label, BODY) + 22
    bucket("title-band", legend_keys)

    # ============================================================== region A ==
    bucket("source-region", [
        canvas.frame(40, 156, 640, 1004, stroke=BLACK, stroke_width=3),
        canvas.text(64, 198, "A · 源库元训练", size=SECTION, weight="700"),
        canvas.text(64, 234, "26 个训练库 · 不设源验证库", size=BODY, fill=INK),
    ])

    # --- A1 source episode ---------------------------------------------------
    episode = bucket("source-episode", [
        canvas.text(64, 282, "① 源库任务 episode", size=KEY, weight="700"),
        canvas.frame(64, 296, 592, 214, stroke=INK, stroke_width=2),
        canvas.text(80, 332, "等概率采样一个源库，库内 unique hash 无放回采样",
                    size=BODY, fill=INK),
        canvas.text(80, 376, "支持集 S", size=KEY, weight="700", fill=PINK),
        canvas.text(344, 376, "查询集 Q", size=KEY, weight="700", fill=MINT),
    ])
    episode.extend(bucket("source-episode",
                          cell_row(REUSE["cell_pink"], 80, 390, 6, 40, 1.35)))
    episode.extend(bucket("source-episode",
                          cell_row(REUSE["cell_mint"], 344, 390, 8, 32, 1.35)))
    episode.extend(bucket("source-episode", [
        canvas.text(80, 452, "20 条 · 标签只进内环", size=BODY, fill=PINK),
        canvas.text(344, 452, "64 条 · 标签只进外环损失", size=BODY, fill=MINT),
        canvas.text(80, 486, "S ∩ Q = ∅，两集合不相交", size=BODY, fill=BLACK),
    ]))

    # --- A2 PRICE feature pathway -------------------------------------------
    pathway = bucket("price-pathway", [
        canvas.text(64, 540, "② 原始 PRICE 特征通路", size=KEY, weight="700"),
    ])
    # Two rows of two cards keep the four feature groups inside the source
    # panel, whose inner width is 592 units.
    features = (("连接列直方图", MINT_FILL), ("扇出", PALE_MINT),
                ("表特征", GREY_FILL), ("过滤特征", PEACH_FILL))
    for index, (label, fill) in enumerate(features):
        card_x = 64 + (index % 2) * 300
        card_y = 558 + (index // 2) * 58
        pathway.extend(canvas.card(card_x, card_y, 288, 50, fill=fill, radius=6))
        pathway.append(canvas.text(card_x + 144, card_y + 33, label, size=BODY,
                                   anchor="middle"))
    pathway.extend(bucket("price-pathway", [
        canvas.arrow(208, 678, 208, 700, fill=BLACK, head=9, shaft=2.5),
        canvas.card(64, 706, 264, 56, fill=BLUE_FILL, radius=8),
        canvas.text(196, 742, "PRICE 编码器 + 回归头", size=BODY, weight="700",
                    anchor="middle"),
        canvas.card(340, 706, 316, 56, fill=GREY_FILL, radius=8),
        canvas.text(498, 742, "辅助输入：PG 估计、token 数", size=BODY,
                    anchor="middle"),
        canvas.text(64, 792, "原生标签 z = log1p(card) + 1", size=BODY, fill=INK),
    ]))

    # --- A3 nested FOMAML ------------------------------------------------------
    bucket("nested-fomaml", [
        canvas.text(64, 832, "③ 嵌套 FOMAML（每个任务）", size=KEY, weight="700"),
        canvas.frame(64, 846, 592, 258, stroke=INK, stroke_width=2, dash="6,4"),
        canvas.frame(78, 860, 286, 230, stroke=INK, stroke_width=2),
        canvas.text(221, 892, "内环：支持集全参数 SGD", size=BODY, weight="700",
                    anchor="middle"),
        canvas.text(221, 926, "5 步 · 原生 MSE", size=BODY, anchor="middle"),
        canvas.text(221, 958, "α = 1e-6", size=BODY, anchor="middle"),
        canvas.text(221, 1004, "φ₀ = θ", size=32, weight="700", anchor="middle",
                    fill=BLUE),
        canvas.text(221, 1044, "仅用支持集标签", size=BODY, fill=INK, anchor="middle"),
        canvas.text(548, 892, "外环：适应后查询损失", size=BODY, weight="700",
                    anchor="middle"),
        canvas.text(548, 926, "一阶近似更新共享 θ", size=BODY, anchor="middle"),
        canvas.text(548, 960, "Adam 2.85e-5", size=BODY, anchor="middle"),
        canvas.text(548, 994, "4 任务 / 次 · 2000 次更新", size=BODY, anchor="middle"),
        canvas.text(548, 1028, "取 last：无早停、不选最优", size=BODY, anchor="middle"),
        canvas.text(548, 1062, "仅用查询集标签", size=BODY, fill=INK, anchor="middle"),
        canvas.arrow(368, 958, 398, 958, fill=BLACK, head=9, shaft=2.5),
    ])
    bucket("source-region", [
        canvas.text(64, 1104, "官方 checkpoint P 只读；克隆为 θ 后由外环更新",
                    size=BODY, fill=INK),
        canvas.text(64, 1138, "P 不被修改，密封产物是独立的 θ★", size=BODY, fill=INK),
    ])

    # ========================================================== boundary ======
    boundary = bucket("no-leak-boundary", [
        canvas.line(732, 156, 732, 566, stroke=BLACK, stroke_width=3,
                    dash="10,6,3,6"),
        canvas.line(732, 706, 732, 1160, stroke=BLACK, stroke_width=3,
                    dash="10,6,3,6"),
        canvas.card(684, 720, 96, 80, fill=PINK_FILL, radius=8),
        canvas.text(732, 752, "禁止", size=SECTION, weight="700", anchor="middle",
                    fill=PINK),
        canvas.text(732, 790, "泄漏", size=SECTION, weight="700", anchor="middle",
                    fill=PINK),
    ])
    rotated = ET.SubElement(root, tag_fn("text"), {
        "x": "0", "y": "0", "font-size": "28", "font-weight": "700",
        "font-family": f"{FONT}-700", "fill": PINK, "text-anchor": "middle",
        "transform": "translate(732,644) rotate(-90)",
        "data-asset-key": lib.next_key("bt"),
    })
    rotated.text = "无泄漏边界"
    boundary.append(rotated.get("data-asset-key"))

    # ========================================================== region B ======
    bucket("target-region", [
        canvas.frame(782, 156, 978, 1004, stroke=BLACK, stroke_width=3),
        canvas.text(806, 198, "B · 目标库评估（尚未执行）", size=SECTION, weight="700"),
        canvas.text(806, 234, "目标库不参与训练、超参或 checkpoint 选择", size=BODY,
                    fill=INK),
    ])

    # --- B1 sealed checkpoint and task copy -----------------------------------
    sealed = bucket("sealed-checkpoint", [
        canvas.text(806, 288, "① 密封共享参数", size=KEY, weight="700"),
    ])
    # The reused config card is 204.8 units wide; 1.70 gives 348 units, which
    # fits the "θ★ = 2000 次外环更新后的 last" label measured at 335 units.
    sealed.extend(place_pair(REUSE["card_config_grey"], 806, 300, 1.80))
    sealed.append(canvas.text(806 + 184, 332, "θ★ = 2000 次外环更新后的 last",
                              size=BODY, weight="700", anchor="middle"))
    sealed.append(canvas.text(1188, 332, "无早停、不选最优 checkpoint", size=BODY,
                              fill=INK))
    sealed.extend(bucket("sealed-checkpoint", [
        canvas.text(806, 392, "② 复制任务副本", size=KEY, weight="700"),
        canvas.arrow(806, 402, 806, 428, fill=BLACK, head=9, shaft=2.5),
    ]))
    sealed.extend(place_pair(REUSE["card_config_mint"], 806, 428, 1.80))
    sealed.append(canvas.text(806 + 184, 460, "φt ← θ★（副本可全参数适应）",
                              size=BODY, weight="700", anchor="middle"))
    sealed.append(canvas.text(1188, 460, "θ★ 本身保持冻结", size=BODY, fill=INK))

    # --- B2 target episode ------------------------------------------------------
    target_episode = bucket("target-episode", [
        canvas.text(806, 516, "③ 目标 episode", size=KEY, weight="700"),
        canvas.frame(806, 530, 930, 206, stroke=INK, stroke_width=2),
        canvas.text(822, 570, "目标支持集", size=KEY, weight="700", fill=PINK),
        canvas.text(1300, 570, "目标测试查询", size=KEY, weight="700", fill=MINT),
    ])
    target_episode.extend(bucket("target-episode",
                                 cell_row(REUSE["cell_pink"], 822, 584, 6, 40, 1.35)))
    target_episode.extend(bucket("target-episode",
                                 cell_row(REUSE["cell_mint"], 1300, 584, 8, 32, 1.35)))
    target_episode.extend(bucket("target-episode", [
        canvas.text(822, 646, "K-shot · 标签只用于适应", size=BODY, fill=PINK),
        canvas.text(1300, 646, "标签只用于最终打分", size=BODY, fill=MINT),
        canvas.text(822, 684, "S_t ∩ Test = ∅；K = 0 时无需适应", size=BODY, fill=BLACK),
        canvas.text(822, 716, "适应只改副本 φt，密封的 θ★ 不变", size=BODY, fill=INK),
    ]))

    # --- B3 adaptation and planned evaluation -------------------------------------
    evaluation = bucket("target-evaluation", [
        canvas.text(806, 772, "④ 目标适应与计划评估", size=KEY, weight="700"),
    ])
    evaluation.extend(place_pair(REUSE["card_blue_stage"], 806, 788, 1.35))
    evaluation.append(canvas.text(854, 818, "适应", size=BODY, weight="700",
                                  anchor="middle"))
    evaluation.append(canvas.arrow(890, 808, 926, 808, fill=BLACK, head=9, shaft=2.5))
    evaluation.extend(canvas.card(932, 784, 430, 54, fill=PEACH_FILL, radius=8))
    evaluation.append(canvas.text(1147, 819, "唯一查询 Q95（计划指标）", size=BODY,
                                  weight="700", anchor="middle"))
    evaluation.append(canvas.text(1376, 819, "P / E / M 同协议", size=BODY,
                                  fill=INK))

    # --- B4 protocol design vs measured result -------------------------------------
    contrast = bucket("protocol-vs-result", [
        canvas.text(806, 878, "⑤ 协议设计与实测结果的区分", size=KEY, weight="700"),
        canvas.card(806, 892, 452, 150, fill=PALE_MINT, radius=8),
        canvas.text(824, 926, "协议设计（本图）", size=KEY, weight="700", fill=MINT),
    ])
    contrast.extend(canvas.text_box(
        824, 970,
        ["库划分、任务采样、内层/外环配方",
         "支持与查询不相交、K-shot 已冻结"],
        size=BODY))
    contrast.extend(bucket("protocol-vs-result", [
        canvas.card(1274, 892, 486, 150, fill=GREY_FILL, radius=8),
        canvas.text(1292, 926, "实测结果（尚未产生）", size=KEY, weight="700", fill=INK),
    ]))
    contrast.extend(canvas.text_box(
        1292, 970,
        ["目标库成绩须在冻结后单独授权评估",
         "本图不表示任何性能提升或方法有效性"],
        size=BODY))

    # --- B5 spatial drift sketch ------------------------------------------------------
    drift = bucket("spatial-drift", [
        canvas.text(806, 1084, "空间漂移", size=KEY, weight="700"),
    ])
    bars = (REUSE["bar_blue_a"], REUSE["bar_blue_b"], REUSE["bar_blue_c"])
    source_heights = (0.30, 0.44, 0.56)
    target_heights = (0.56, 0.34, 0.20)
    # Bars are scaled to 0.40 of the source column, so the tallest bar is
    # 154.5 * 0.40 = 62 units; the sketch is anchored on a common baseline.
    baseline = 1116
    for index, (key, height) in enumerate(zip(bars, source_heights)):
        drift.append(place(key, 920 + index * 22, baseline - height * 62, 0.40))
    drift.append(canvas.text(920, 1148, "源库", size=BODY, fill=INK))
    drift.append(canvas.arrow(972, 1090, 1004, 1090, fill=BLACK, head=9, shaft=2.5))
    for index, (key, height) in enumerate(zip(bars, target_heights)):
        drift.append(place(key, 1032 + index * 22, baseline - height * 62, 0.40))
    drift.append(canvas.text(1032, 1148, "目标库", size=BODY, fill=INK))
    drift.append(canvas.text(1118, 1090, "直方图、表结构与选择率分布不同", size=BODY,
                              fill=INK))
    drift.append(canvas.text(1118, 1124, "同一模型需在目标库重新适应", size=BODY,
                              fill=INK))
