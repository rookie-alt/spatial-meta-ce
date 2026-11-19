#!/usr/bin/env python3
"""Generate the A1 architecture figure as an editable SVG."""

from __future__ import annotations

from html import escape
from pathlib import Path


W, H = 1920, 1080
OUT = Path(__file__).resolve().parent

FONT = "PingFang SC, Source Han Sans SC, Noto Sans CJK SC, sans-serif"
MONO = "SF Mono, Menlo, Consolas, monospace"

INK = "#1F2A44"
MUTED = "#6C7892"
LINE = "#D9E1EE"
ARROW = "#5E6E8A"
WHITE = "#FFFFFF"

LAV = "#7C74C9"
LAV_LIGHT = "#EEEBFF"
LAV_STROKE = "#CFC8F0"
MINT = "#3F9C86"
MINT_LIGHT = "#E8F8F1"
MINT_STROKE = "#BFE4D3"
PEACH = "#D98258"
PEACH_LIGHT = "#FFF1E6"
PEACH_STROKE = "#F1D2BC"
BLUE = "#4F78B8"
BLUE_LIGHT = "#EAF3FF"
BLUE_STROKE = "#C9DAF2"
GOLD = "#B8892F"
GOLD_LIGHT = "#FFF8E4"
GOLD_STROKE = "#EAD7A0"
GREEN = "#2F8F68"
GREEN_LIGHT = "#E9F8F0"
GREEN_STROKE = "#9FD6BC"

parts: list[str] = []


def add(value: str) -> None:
    parts.append(value)


def text(
    x: float,
    y: float,
    value: str,
    size: float = 16,
    fill: str = INK,
    weight: int = 400,
    anchor: str = "start",
    family: str = FONT,
    opacity: float = 1.0,
    letter_spacing: float = 0,
) -> str:
    spacing = f' letter-spacing="{letter_spacing}"' if letter_spacing else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'opacity="{opacity}"{spacing}>{escape(value)}</text>'
    )


def lines(
    x: float,
    y: float,
    values: list[str],
    size: float = 15,
    fill: str = INK,
    weight: int = 400,
    leading: float = 22,
    anchor: str = "start",
) -> str:
    spans = []
    for index, value in enumerate(values):
        dy = 0 if index == 0 else leading
        spans.append(
            f'<tspan x="{x}" dy="{dy}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{escape(value)}</tspan>'
        )
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" text-anchor="{anchor}">'
        + "".join(spans)
        + "</text>"
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    rx: float = 18,
    fill: str = WHITE,
    stroke: str = LINE,
    sw: float = 1.2,
    shadow: bool = True,
    dash: str | None = None,
    opacity: float = 1.0,
) -> str:
    shadow_attr = ' filter="url(#shadow)"' if shadow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"'
        f"{shadow_attr}{dash_attr}/>"
    )


def card(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = WHITE,
    stroke: str = LINE,
    rx: float = 18,
    shadow: bool = True,
    dash: str | None = None,
    sw: float = 1.2,
) -> str:
    return rect(
        x, y, w, h, rx=rx, fill=fill, stroke=stroke, sw=sw, shadow=shadow, dash=dash
    )


def pill(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    fill: str,
    fg: str,
    size: float = 12,
    weight: int = 600,
    stroke: str | None = None,
    shadow: bool = False,
) -> str:
    return (
        rect(
            x,
            y,
            w,
            h,
            rx=h / 2,
            fill=fill,
            stroke=stroke or "none",
            sw=1.0,
            shadow=shadow,
        )
        + f'<text x="{x + w / 2}" y="{y + h / 2 + size * 0.34}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fg}" text-anchor="middle">'
        f"{escape(label)}</text>"
    )


def circle(
    cx: float, cy: float, r: float, fill: str, stroke: str = "none", sw: float = 1.2
) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw}"/>'
    )


def arrow(
    d: str,
    dashed: bool = False,
    color: str = ARROW,
    width: float = 2.2,
    marker: str | None = None,
) -> str:
    marker = marker or ("arrowDashed" if dashed else "arrow")
    dash_attr = ' stroke-dasharray="7 6"' if dashed else ""
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'marker-end="url(#{marker})"{dash_attr}/>'
    )


def label_on_path(
    x: float, y: float, label: str, fill: str = WHITE, fg: str = MUTED
) -> str:
    width = max(42, len(label) * 8.2 + 18)
    return (
        rect(
            x - width / 2,
            y - 12,
            width,
            24,
            rx=12,
            fill=fill,
            stroke=LINE,
            sw=0.8,
            shadow=False,
        )
        + f'<text x="{x}" y="{y + 4.5}" font-family="{FONT}" font-size="11.5" '
        f'font-weight="500" fill="{fg}" text-anchor="middle">{escape(label)}</text>'
    )


def icon_bars(cx: float, cy: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="2.1" stroke-linecap="round" fill="none">'
        f'<path d="M{cx - 9} {cy + 9}V{cy + 2}"/>'
        f'<path d="M{cx} {cy + 9}V{cy - 4}"/>'
        f'<path d="M{cx + 9} {cy + 9}V{cy - 11}"/>'
        f"</g>"
    )


def icon_fanout(cx: float, cy: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="1.9" stroke-linecap="round" fill="none">'
        f'<circle cx="{cx - 10}" cy="{cy}" r="3.2"/>'
        f'<path d="M{cx - 6.8} {cy}H{cx + 1}V{cy - 10}H{cx + 10}"/>'
        f'<path d="M{cx + 1} {cy}H{cx + 10}"/>'
        f'<path d="M{cx + 1} {cy}V{cy + 10}H{cx + 10}"/>'
        f'<circle cx="{cx + 12}" cy="{cy - 10}" r="2.2"/>'
        f'<circle cx="{cx + 12}" cy="{cy}" r="2.2"/>'
        f'<circle cx="{cx + 12}" cy="{cy + 10}" r="2.2"/>'
        f"</g>"
    )


def icon_table(cx: float, cy: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="1.8" fill="none">'
        f'<rect x="{cx - 11}" y="{cy - 9}" width="22" height="18" rx="2.5"/>'
        f'<path d="M{cx - 11} {cy - 2.5}H{cx + 11}M{cx - 11} {cy + 4}H{cx + 11}"/>'
        f'<path d="M{cx - 3.5} {cy - 9}V{cy + 9}M{cx + 3.5} {cy - 9}V{cy + 9}"/>'
        f"</g>"
    )


def icon_filter(cx: float, cy: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="1.9" stroke-linejoin="round" fill="none">'
        f'<path d="M{cx - 11} {cy - 9}H{cx + 11}L{cx + 3} {cy + 1}V{cy + 10}'
        f'L{cx - 3} {cy + 7}V{cy + 1}Z"/>'
        f"</g>"
    )


def icon_database(cx: float, cy: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="1.8" fill="none">'
        f'<ellipse cx="{cx}" cy="{cy - 7}" rx="9" ry="3.5"/>'
        f'<path d="M{cx - 9} {cy - 7}V{cy + 7}C{cx - 9} {cy + 9} {cx - 4} {cy + 10.5} {cx} {cy + 10.5}'
        f'C{cx + 4} {cy + 10.5} {cx + 9} {cy + 9} {cx + 9} {cy + 7}V{cy - 7}"/>'
        f'<path d="M{cx - 9} {cy + 2}C{cx - 9} {cy + 4} {cx - 4} {cy + 5.5} {cx} {cy + 5.5}'
        f'C{cx + 4} {cy + 5.5} {cx + 9} {cy + 4} {cx + 9} {cy + 2}"/>'
        f"</g>"
    )


def icon_net(cx: float, cy: float, color: str) -> str:
    nodes = [
        (cx - 16, cy - 10),
        (cx - 16, cy + 10),
        (cx - 4, cy - 14),
        (cx - 4, cy),
        (cx - 4, cy + 14),
        (cx + 9, cy - 7),
        (cx + 9, cy + 7),
    ]
    edges = [
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 5),
        (4, 6),
    ]
    out = [f'<g stroke="{color}" stroke-width="1.1" opacity="0.55">']
    for a, b in edges:
        out.append(
            f'<path d="M{nodes[a][0]} {nodes[a][1]}L{nodes[b][0]} {nodes[b][1]}"/>'
        )
    out.append("</g>")
    for x, y in nodes:
        out.append(circle(x, y, 2.6, WHITE, color, 1.4))
    return "".join(out)


def icon_target(cx: float, cy: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="1.8" fill="none">'
        f'<circle cx="{cx}" cy="{cy}" r="10"/>'
        f'<circle cx="{cx}" cy="{cy}" r="5.5"/>'
        f'<path d="M{cx} {cy}L{cx + 11} {cy - 11}"/>'
        f"</g>"
    )


def icon_layers(cx: float, cy: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="1.8" stroke-linejoin="round" fill="none">'
        f'<path d="M{cx - 11} {cy - 7}L{cx} {cy - 12}L{cx + 11} {cy - 7}L{cx} {cy - 2}Z"/>'
        f'<path d="M{cx - 11} {cy}L{cx} {cy + 5}L{cx + 11} {cy}"/>'
        f'<path d="M{cx - 11} {cy + 7}L{cx} {cy + 12}L{cx + 11} {cy + 7}"/>'
        f"</g>"
    )


def icon_shield(cx: float, cy: float, color: str) -> str:
    return (
        f'<g stroke="{color}" stroke-width="1.8" stroke-linejoin="round" fill="none">'
        f'<path d="M{cx} {cy - 12}L{cx + 10} {cy - 8}V{cy + 1}'
        f"C{cx + 10} {cy + 7} {cx + 5} {cy + 11} {cx} {cy + 13}"
        f'C{cx - 5} {cy + 11} {cx - 10} {cy + 7} {cx - 10} {cy + 1}V{cy - 8}Z"/>'
        f'<path d="M{cx - 3.5} {cy}L{cx - 1} {cy + 3}L{cx + 4} {cy - 3}"/>'
        f"</g>"
    )


def icon_bolt(cx: float, cy: float, color: str) -> str:
    return (
        f'<path d="M{cx + 1} {cy - 12}L{cx - 8} {cy + 1}H{cx - 1}L{cx - 3} {cy + 12}'
        f'L{cx + 8} {cy - 2}H{cx + 1}Z" fill="{color}" opacity="0.9"/>'
    )


def header_pill(x: float, y: float, w: float, label: str, gradient: str) -> str:
    return (
        rect(
            x,
            y,
            w,
            44,
            rx=22,
            fill=f"url(#{gradient})",
            stroke="none",
            sw=0,
            shadow=True,
        )
        + f'<text x="{x + w / 2}" y="{y + 28}" font-family="{FONT}" font-size="18.5" '
        f'font-weight="600" fill="{WHITE}" text-anchor="middle">{escape(label)}</text>'
    )


add(
    f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#F8F9FD"/><stop offset="0.55" stop-color="#F5F7FC"/><stop offset="1" stop-color="#F1F4FA"/>
  </linearGradient>
  <linearGradient id="hdrLav" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#948AE0"/><stop offset="1" stop-color="#6E7CCB"/>
  </linearGradient>
  <linearGradient id="hdrMint" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#67C2A6"/><stop offset="1" stop-color="#3F9C86"/>
  </linearGradient>
  <linearGradient id="hdrPeach" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#EDA477"/><stop offset="1" stop-color="#D98258"/>
  </linearGradient>
  <linearGradient id="panelLav" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#F8F6FF"/><stop offset="1" stop-color="#F2F7FF"/>
  </linearGradient>
  <linearGradient id="panelMint" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#F3FCF8"/><stop offset="1" stop-color="#F1FAFF"/>
  </linearGradient>
  <linearGradient id="panelPeach" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#FFF9F3"/><stop offset="1" stop-color="#FFF4F0"/>
  </linearGradient>
  <linearGradient id="cardLav" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#F1EEFF"/>
  </linearGradient>
  <linearGradient id="cardMint" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#EAF9F2"/>
  </linearGradient>
  <linearGradient id="cardBlue" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#EEF5FF"/>
  </linearGradient>
  <linearGradient id="cardOutput" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#F3FCF7"/><stop offset="1" stop-color="#DFF3E9"/>
  </linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#1B2A4A" flood-opacity="0.075"/>
  </filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 Z" fill="#5E6E8A"/>
  </marker>
  <marker id="arrowDashed" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 Z" fill="#B8892F"/>
  </marker>
</defs>
<rect width="{W}" height="{H}" fill="url(#bg)"/>
'''
)

add('<ellipse cx="240" cy="90" rx="260" ry="140" fill="#EDEBFF" opacity="0.45"/>')
add('<ellipse cx="1710" cy="1000" rx="300" ry="150" fill="#EAF8F2" opacity="0.4"/>')
add('<ellipse cx="1840" cy="80" rx="220" ry="120" fill="#FFF1E6" opacity="0.5"/>')

add(
    pill(
        56,
        40,
        286,
        30,
        "122 · A1 MODEL ARCHITECTURE",
        LAV_LIGHT,
        LAV,
        11.5,
        700,
        LAV_STROKE,
    )
)
add(text(56, 108, "A1：PRICE 式两阶段编码器 + PG 残差锚", 42, INK, 600))
add(
    text(
        56,
        146,
        "四组 PRICE 特征 → 两阶段 self-attention 编码 → 零初始化残差头；PG 只作为输出锚点",
        17,
        MUTED,
        400,
    )
)

add(card(1390, 48, 220, 34, WHITE, LINE, 17, True))
add(text(1404, 70, "A1 · 293,905 params", 12.5, INK, 600))
add(card(1622, 48, 242, 34, WHITE, LINE, 17, True))
add(text(1636, 70, "d=64 · 2 blocks · 4 heads", 12.5, INK, 600))
add(card(1390, 90, 220, 34, WHITE, LINE, 17, True))
add(text(1404, 112, "A1-PG · 294,033", 12.5, INK, 600))
add(card(1622, 90, 242, 34, WHITE, LINE, 17, True))
add(text(1636, 112, "K=0 主指标", 12.5, INK, 600))

add(
    f'<path d="M1398 143H1436" stroke="{ARROW}" stroke-width="2.2" marker-end="url(#arrow)"/>'
)
add(text(1446, 147, "A1", 12.5, MUTED, 600))
add(
    f'<path d="M1510 143H1548" stroke="{GOLD}" stroke-width="2.2" stroke-dasharray="7 6" marker-end="url(#arrowDashed)"/>'
)
add(text(1558, 147, "A1-PG 探索支路", 12.5, GOLD, 600))

add(rect(56, 190, 760, 720, 28, "url(#panelLav)", LAV_STROKE, 1.4, False, "9 7"))
add(rect(840, 190, 470, 720, 28, "url(#panelMint)", MINT_STROKE, 1.4, False, "9 7"))
add(rect(1334, 190, 530, 720, 28, "url(#panelPeach)", PEACH_STROKE, 1.4, False, "9 7"))

add(header_pill(84, 214, 380, "① 查询特征编码 · Query Encoding", "hdrLav"))
add(header_pill(868, 214, 414, "② 表示融合 · Representation Fusion", "hdrMint"))
add(header_pill(1362, 214, 414, "③ 残差预测 · Residual Prediction", "hdrPeach"))

add(text(86, 282, "四组 PRICE 特征（容量 [12, 12, 7, 10]）", 13, MUTED, 500))
add(text(870, 282, "把 query 表示压成定长向量", 13, MUTED, 500))
add(text(1364, 282, "PG 锚定 + 零初始化残差修正", 13, MUTED, 500))

inputs = [
    ("Join Histogram", "连接直方图", "12 × 40", LAV, LAV_LIGHT, icon_bars, 300),
    ("Fanout", "扇出分布", "12 × 40", BLUE, BLUE_LIGHT, icon_fanout, 382),
    ("Table", "表规模 / log size", "7 × 4", MINT, MINT_LIGHT, icon_table, 480),
    ("Filter", "过滤条件", "10 × 43", PEACH, PEACH_LIGHT, icon_filter, 562),
]
for title, subtitle, cap, accent, tint, icon, y in inputs:
    add(card(86, y, 300, 70))
    add(rect(86, y + 13, 4, 44, 2, accent, "none", 0, False))
    add(circle(119, y + 35, 18, tint, "none"))
    add(icon(119, y + 35, accent))
    add(text(148, y + 28, title, 17, INK, 600))
    add(text(148, y + 49, subtitle, 11.5, MUTED, 400))
    add(pill(308, y + 21, 60, 26, cap, tint, accent, 10.5, 700))

add(card(430, 300, 330, 64, "url(#cardLav)", LAV_STROKE))
add(text(452, 326, "ScaleEmbedding", 17, INK, 600))
add(text(452, 347, "join_hist / fanout → token embedding", 11.5, MUTED, 400))
add(pill(676, 316, 64, 26, "Linear", LAV_LIGHT, LAV, 10.5, 700))

add(card(430, 382, 330, 96, "url(#cardLav)", LAV_STROKE))
add(text(452, 410, "Scale Encoder", 17, INK, 600))
add(text(452, 432, "2 × Transformer block", 11.8, MUTED, 400))
add(pill(452, 444, 134, 22, "d=64 · 4 heads · dropout 0", LAV_LIGHT, LAV, 9.8, 600))
add(icon_net(708, 430, LAV))

add(card(430, 500, 330, 64, "url(#cardMint)", MINT_STROKE))
add(text(452, 526, "FilterEmbedding", 17, INK, 600))
add(text(452, 547, "追加 table / filter token", 11.5, MUTED, 400))
add(pill(676, 516, 64, 26, "append", MINT_LIGHT, MINT, 10.5, 700))

add(card(430, 582, 330, 96, "url(#cardMint)", MINT_STROKE))
add(text(452, 610, "Filter Encoder", 17, INK, 600))
add(text(452, 632, "2 × Transformer block → CLS", 11.8, MUTED, 400))
add(pill(452, 644, 134, 22, "d=64 · 4 heads · dropout 0", MINT_LIGHT, MINT, 9.8, 600))
add(icon_net(708, 630, MINT))

add(arrow("M386 335H410V322H430"))
add(
    f'<path d="M386 417H410V322" fill="none" stroke="{ARROW}" stroke-width="2.2" stroke-linecap="round"/>'
)
add(arrow("M386 515H410V526H430"))
add(
    f'<path d="M386 597H410V526" fill="none" stroke="{ARROW}" stroke-width="2.2" stroke-linecap="round"/>'
)
add(arrow("M595 364V382"))
add(arrow("M595 478V500"))
add(label_on_path(625, 489, "scale tokens"))
add(arrow("M595 564V582"))
add(arrow("M760 630H800V328H870"))

add(card(86, 720, 674, 142, WHITE, LINE, 18, True))
add(text(108, 750, "实现约束", 15, INK, 600))
constraints = [
    ("padding", "PRICE padding = -1000；table 组首维固定为 0"),
    ("fanout", "缓存中为 log1p，进入 embedding 前先 expm1 还原"),
    ("容量", "只由 23 个训练库计算；超过容量显式 unsupported，不截断"),
]
for index, (tag, value) in enumerate(constraints):
    yy = 778 + index * 27
    tag_fill = LAV_LIGHT if index == 0 else (BLUE_LIGHT if index == 1 else MINT_LIGHT)
    tag_fg = LAV if index == 0 else (BLUE if index == 1 else MINT)
    add(pill(108, yy - 13, 62, 20, tag, tag_fill, tag_fg, 9.5, 700))
    add(text(182, yy + 1, value, 11.8, MUTED, 400))

add(card(870, 300, 410, 76, WHITE, LINE))
add(circle(900, 338, 18, LAV_LIGHT))
add(text(900, 343, "CLS", 10, LAV, 700, "middle"))
add(text(938, 328, "CLS 64-d", 18, INK, 600))
add(text(938, 350, "filter encoder 的全局 query 读出", 11.8, MUTED, 400))
add(pill(1164, 316, 92, 24, "64-d", LAV_LIGHT, LAV, 10.5, 700))

add(card(870, 410, 170, 68, WHITE, LINE))
add(text(890, 436, "counts (4)", 15, INK, 600))
add(text(890, 456, "join / fanout / table / filter", 10.5, MUTED, 400))
add(pill(1058, 428, 64, 30, "log1p", BLUE_LIGHT, BLUE, 10.5, 700))
add(card(1140, 410, 140, 68, "url(#cardBlue)", BLUE_STROKE))
add(text(1158, 436, "len_net", 15, INK, 600))
add(text(1158, 456, "4 → 16", 11.5, MUTED, 600))
add(pill(1206, 416, 58, 22, "16-d", BLUE_LIGHT, BLUE, 9.8, 700))

add(card(870, 510, 170, 68, WHITE, LINE))
add(text(890, 536, "table log sizes", 15, INK, 600))
add(text(890, 556, "各表 log(size) 求和", 10.5, MUTED, 400))
add(pill(1058, 528, 64, 30, "sum", MINT_LIGHT, MINT, 10.5, 700))
add(card(1140, 510, 140, 68, "url(#cardMint)", MINT_STROKE))
add(text(1158, 536, "log_cartesian", 13.5, INK, 600))
add(text(1158, 556, "1-d · 笛卡尔积规模", 10.8, MUTED, 600))

add(arrow("M1280 338H1298V670H1280"))
add(arrow("M1210 478H1120V590H1075V620"))
add(arrow("M1250 578V600H1160V620"))

add(card(870, 620, 410, 100, "url(#cardBlue)", BLUE_STROKE))
add(text(892, 652, "Concat 81-d", 18, INK, 600))
add(text(892, 674, "CLS 64 + length 16 + log_cartesian 1", 11.8, MUTED, 400))
add(pill(892, 688, 94, 22, "CLS 64", LAV_LIGHT, LAV, 9.8, 700))
add(text(994, 704, "+", 15, MUTED, 600))
add(pill(1010, 688, 92, 22, "len 16", BLUE_LIGHT, BLUE, 9.8, 700))
add(text(1110, 704, "+", 15, MUTED, 600))
add(pill(1126, 688, 118, 22, "log_cart 1", MINT_LIGHT, MINT, 9.8, 700))
add(pill(1162, 632, 96, 30, "81-d", BLUE, WHITE, 12.5, 700))

add(card(870, 752, 410, 70, GOLD_LIGHT, GOLD_STROKE, 18, True, "8 6"))
add(text(892, 780, "＋ pg_log（A1-PG 可选）", 15.5, GOLD, 600))
add(text(892, 801, "82-d head input · 仅多 128 params · 默认关闭", 10.8, MUTED, 400))
add(arrow("M1280 787H1320V520H1388V500", dashed=True, color=GOLD))

add(
    lines(
        870,
        856,
        [
            "A1：PG 只作为输出锚点，不进入 encoder trunk 或 head。",
            "A1-PG：pg_log 额外拼进残差头，编码器结构不变。",
        ],
        11.5,
        MUTED,
        400,
        19,
    )
)

add(card(1364, 300, 470, 200, "url(#cardBlue)", BLUE_STROKE))
add(text(1388, 336, "Residual MLP", 20, INK, 600))
add(text(1388, 358, "残差修正头", 11.5, MUTED, 400))
add(icon_net(1778, 366, BLUE))

layer_labels = [("81", 1388, 70), ("128", 1474, 64), ("64", 1554, 58), ("1", 1628, 44)]
for label, x, width in layer_labels:
    add(rect(x, 382, width, 38, 10, BLUE_LIGHT, BLUE_STROKE, 1.0, False))
    add(text(x + width / 2, 407, label, 14.5, BLUE, 700, "middle", MONO))
    if x != 1628:
        add(
            f'<path d="M{x + width + 2} 401H{x + width + 14}" stroke="{BLUE}" stroke-width="1.8" marker-end="url(#arrow)"/>'
        )

add(text(1388, 452, "ReLU · 最后一层零初始化 · dropout 0", 12, MUTED, 400))
add(pill(1388, 466, 150, 24, "81 → 128 → 64 → 1", BLUE_LIGHT, BLUE, 10.5, 700))

add(card(1364, 530, 210, 72, WHITE, LINE))
add(text(1386, 558, "residual r", 17, INK, 600))
add(text(1386, 580, "query-specific correction", 10.8, MUTED, 400))
add(pill(1512, 544, 44, 22, "1-d", LAV_LIGHT, LAV, 9.8, 700))

add(card(1604, 530, 230, 72, GOLD_LIGHT, GOLD_STROKE))
add(icon_database(1632, 566, GOLD))
add(text(1652, 558, "PG anchor", 16.5, INK, 600))
add(text(1652, 580, "log(max(PG, 1))", 11.5, GOLD, 600, family=MONO))

add(
    f'<path d="M1469 602V615H1594" fill="none" stroke="{ARROW}" stroke-width="2.2" stroke-linecap="round"/>'
)
add(
    f'<path d="M1719 602V615H1594" fill="none" stroke="{ARROW}" stroke-width="2.2" stroke-linecap="round"/>'
)
add(circle(1594, 615, 12, WHITE, LINE, 1.2))
add(text(1594, 620, "+", 17, MUTED, 600, "middle"))
add(arrow("M1594 627V650"))

add(card(1364, 650, 470, 105, "url(#cardOutput)", GREEN_STROKE, 20, True))
add(text(1388, 684, "输出", 12, GREEN, 700, letter_spacing=1.5))
add(text(1388, 718, "ŷ = log(max(PG, 1)) + r", 25, INK, 600, family=MONO))
add(text(1388, 742, "zero-init ⇒ 训练前 / 0-step 输出逐位等于 PG", 11.5, MUTED, 400))
add(pill(1720, 664, 88, 26, "anchor + residual", GREEN_LIGHT, GREEN, 9.2, 700))

add(
    lines(
        1364,
        800,
        [
            "A1 正式臂：PG 不进 trunk/head，只做输出锚。",
            "A1-PG 探索臂：pg_log 额外进入 head（82-d）。",
        ],
        12,
        MUTED,
        400,
        20,
    )
)
add(pill(1364, 842, 210, 28, "A1 293,905 · A1-PG 294,033", LAV_LIGHT, LAV, 10.2, 700))

add(card(56, 930, 1808, 100, WHITE, LINE, 22, True))
add(
    f'<path d="M520 950V1010M980 950V1010M1440 950V1010" stroke="{LINE}" stroke-width="1.2"/>'
)

bottom = [
    ("训练 · ERM", "4000 updates · 3 seeds · last.pt", LAV, LAV_LIGHT, icon_bolt, 80),
    (
        "评估适配",
        "K=5/20 · full scope · LR 1e-3 · 5 steps",
        MINT,
        MINT_LIGHT,
        icon_target,
        544,
    ),
    (
        "LoRA（B 阶段）",
        "Q/V · r=4 · α=4 · 10,240 params",
        BLUE,
        BLUE_LIGHT,
        icon_layers,
        1004,
    ),
    (
        "边界",
        "目标库不参与训练/选择 · 容量仅由 23 个训练库计算",
        PEACH,
        PEACH_LIGHT,
        icon_shield,
        1464,
    ),
]
for title, subtitle, accent, tint, icon, x in bottom:
    add(circle(x + 18, 980, 17, tint))
    add(icon(x + 18, 980, accent))
    add(text(x + 44, 974, title, 14.5, INK, 600))
    add(text(x + 44, 996, subtitle, 10.8, MUTED, 400))

add("</svg>")

(OUT / "a1-architecture.svg").write_text("".join(parts), encoding="utf-8")
print(OUT / "a1-architecture.svg")
