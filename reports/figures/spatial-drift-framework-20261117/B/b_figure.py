"""Candidate B: spatial-drift framework figure for cross-database CE.

The composition is reorganised around the source-to-target spatial-drift story:

    source databases -> task episode -> PRICE feature path -> nested FOMAML
    -> sealed shared parameters -> target episode -> planned evaluation

Real VA-196 objects are reused rather than redrawn: card twin pairs (fill plus
stroke path), panel frames, solid arrow heads, the dash-dot panel divider, the
query/weight mini glyph cells, the encoder column bars and the cost-chart bars
reused as histogram sketches. Only objects whose meaning changed are replaced
by newly drawn, editable SVG objects.
"""
from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

try:
    from source_asset import run, local
except ModuleNotFoundError:  # pragma: no cover - packaged copies keep lib/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
    from source_asset import run, local

import b_figure_lib as lib

INK = lib.INK
BLACK = lib.BLACK
BLUE = lib.BLUE
BLUE_FILL = lib.BLUE_FILL
PINK = lib.PINK
PINK_FILL = lib.PINK_FILL
PEACH = lib.PEACH
PEACH_FILL = lib.PEACH_FILL
MINT = lib.MINT
MINT_FILL = lib.MINT_FILL
PALE_MINT = lib.PALE_MINT
GREEN_FILL = lib.GREEN_FILL
GREY_FILL = lib.GREY_FILL
CYAN = lib.CYAN

FONT = "NotoSansSC"

# --------------------------------------------------------------- reuse map --
# VA-196 source objects reused verbatim (geometry cloned into the new layout).
REUSE = {
    # Model-stack cards (fill + stroke twin pairs).
    "card_grey": ("e0007", "e0008"),
    "card_grey2": ("e0009", "e0010"),
    "card_blue": ("e0011", "e0012"),
    "card_blue2": ("e0013", "e0014"),
    # Expanded-block cards (Q / K / V generation).
    "card_green": ("e0030", "e0031"),
    "card_pink": ("e0032", "e0033"),
    "card_peach": ("e0034", "e0035"),
    # Vertical encoder columns.
    "col_mint_a": ("e0038", "e0039"),
    "col_mint_b": ("e0051", "e0052"),
    # Training-steps vocabulary.
    "bar_mint": ("e0187", "e0188"),
    "col_grey": ("e0184", "e0185"),
    "card_peach_s": ("e0192", "e0193"),
    "card_mint_s": ("e0195", "e0196"),
    "col_mint": ("e0241", "e0242"),
    # Inference-steps vocabulary.
    "card_blue_stage": ("e0245", "e0246"),
    "card_blue_stage2": ("e0257", "e0258"),
    "col_peach": ("e0261", "e0262"),
    # Profiling-loop vocabulary.
    "card_config_grey": ("e0470", "e0471"),
    "card_config_mint": ("e0474", "e0475"),
    # Mini glyph cells.
    "cell_pink": ("e0116", "e0117"),
    "cell_mint": ("e0294", "e0295"),
    "cell_blue": ("e0286", "e0287"),
    # Histogram bars (cost-chart series reused as histogram sketches).
    "bar_blue_a": "e0540",
    "bar_blue_b": "e0541",
    "bar_blue_c": "e0542",
    # Solid arrow heads.
    "arrow_right": "e0233",
    "arrow_right_b": "e0249",
    "arrow_right_c": "e0263",
    "arrow_down": "e0026",
    # Panel frames.
    "frame": "e0004",
    # Dash-dot panel divider.
    "divider": "e0181",
}


def _subset_fonts(font_dir: Path, characters: str) -> None:
    """Subset the two used faces to the characters present in the figure."""
    from fontTools import subset
    from fontTools.ttLib import TTFont

    for face, family, subfamily in (("Regular", "NotoSansSC", "Regular"),
                                    ("Bold", "NotoSansSC-700", "Bold")):
        source = font_dir / f"NotoSansSC-{face}.otf"
        font = TTFont(source)
        options = subset.Options()
        options.layout_features = ["*"]
        options.name_IDs = ["*"]
        options.notdef_outline = True
        options.recalc_bounds = True
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=characters)
        subsetter.subset(font)
        # The @font-face family must match the CSS family used in the SVG,
        # otherwise the renderer silently substitutes another face.
        for record in font["name"].names:
            if record.nameID == 1:
                record.string = family
            elif record.nameID == 2:
                record.string = subfamily
            elif record.nameID == 4:
                record.string = family if subfamily == "Regular" else f"{family} {subfamily}"
            elif record.nameID == 6:
                record.string = "NotoSansSC" if subfamily == "Regular" else "NotoSansSC-Bold"
            elif record.nameID == 16:
                record.string = family
            elif record.nameID == 17:
                record.string = subfamily
        font.save(font_dir / f"NotoSansSC-{face}.subset.otf")


def customize(root: ET.Element, context):
    """Rebuild the composition around the spatial-drift narrative."""
    geometry = context["geometry"]
    elements = context["elements"]
    tag_fn = context["tag"]
    metas = {e["key"]: e for e in geometry["elements"]}

    for child in list(root):
        if local(child) in {"g", "defs"}:
            root.remove(child)
        elif local(child) == "style":
            root.remove(child)

    lib.reset_keys()
    canvas = lib.Canvas(root, tag_fn, FONT)
    canvas.bold_family = f"{FONT}-700"
    root.set("viewBox", f"0 0 {lib.CANVAS_W:g} {lib.CANVAS_H:g}")

    collected: dict[str, list[str]] = {}

    def bucket(name: str, keys):
        """Record the keys of one semantic part; card helpers return tuples."""
        flat = []
        for item in keys:
            if isinstance(item, str):
                flat.append(item)
            else:
                flat.extend(item)
        collected.setdefault(name, []).extend(flat)
        return flat

    def place(source_key: str, x: float, y: float, scale: float = 1.0) -> str:
        """Clone a source object to a new position, keeping its real geometry."""
        element = elements.get(source_key)
        if element is None:
            raise KeyError(f"Unknown source object {source_key!r}")
        clone = ET.fromstring(ET.tostring(element))
        key = lib.next_key("br")
        clone.set("data-asset-key", key)
        bx, by = metas[source_key]["bounds"][0], metas[source_key]["bounds"][1]
        source_transform = element.get("transform") or ""
        tx = x - bx * scale
        ty = y - by * scale
        clone.set("transform",
                  f"translate({tx:.4f},{ty:.4f}) scale({scale:.4f}) {source_transform}")
        root.append(clone)
        return key

    def place_pair(pair, x: float, y: float, scale: float = 1.0) -> list[str]:
        return [place(pair[0], x, y, scale), place(pair[1], x, y, scale)]

    def cell_row(pair, x: float, y: float, count: int, step: float,
                 scale: float = 1.0) -> list[str]:
        keys = []
        for index in range(count):
            keys.extend(place_pair(pair, x + index * step, y, scale))
        return keys

    import b_layout
    b_layout.draw(canvas, root, tag_fn, bucket, place, place_pair, cell_row)

    # ------------------------------------------------------------- fonts ------
    characters = "".join(
        element.text or "" for element in root.iter() if local(element) == "text")
    font_dir = Path(__file__).resolve().parent / "fonts"
    _subset_fonts(font_dir, characters)
    lib.embed_fonts(root, font_dir, [
        (FONT, "NotoSansSC-Regular.subset.otf", "400"),
        (f"{FONT}-700", "NotoSansSC-Bold.subset.otf", "700"),
    ])

    # -------------------------------------------------------------- parts -----
    parts = {}
    definitions = {
        "title-band": ("标题与图例", "component",
                        "图题、副题与颜色角色图例。", "沿用源资产标题排版层级。",
                        [40, 24, 1720, 104]),
        "source-region": ("源库元训练面板", "component",
                          "26 个源训练库上的元训练区域。", "沿用源资产面板外框语言。",
                          [40, 150, 630, 1000]),
        "source-episode": ("源库任务 episode", "component",
                           "同一源库内不相交的支持集与查询集采样。",
                           "复用源资产查询单元格图元。", [62, 272, 590, 210]),
        "price-pathway": ("原始 PRICE 特征通路", "component",
                          "四组特征与回归头辅助输入。",
                          "复用源资产阶段卡片与箭头图元。", [62, 510, 590, 216]),
        "nested-fomaml": ("嵌套 FOMAML", "component",
                          "内环支持集全参数 SGD 与外环查询损失更新。",
                          "复用源资产虚线分栏与实线子框。", [62, 750, 590, 266]),
        "no-leak-boundary": ("无泄漏边界", "component",
                              "源库与目标库的数据库级隔离边界。",
                              "复用源资产点划线分栏线。", [660, 150, 120, 1000]),
        "target-region": ("目标库评估面板", "component",
                          "密封参数、目标适应与计划评估区域。",
                          "沿用源资产面板外框语言。", [770, 150, 990, 1000]),
        "sealed-checkpoint": ("密封参数与任务副本", "component",
                              "θ★ 密封与 φt 任务副本。",
                              "复用源资产配置卡片。", [792, 258, 946, 196]),
        "target-episode": ("目标支持集与测试查询", "component",
                           "不相交的目标支持集与测试查询。",
                           "复用源资产查询单元格图元。", [792, 472, 946, 214]),
        "target-evaluation": ("目标适应与计划评估", "component",
                              "目标适应后以唯一查询 Q95 作计划评估。",
                              "复用源资产阶段卡片与箭头图元。", [792, 712, 946, 80]),
        "protocol-vs-result": ("协议设计与实测结果", "component",
                               "区分已冻结的协议设计与尚未产生的实测结果。",
                               "沿用源资产浅色卡片语言。", [792, 822, 946, 204]),
        "spatial-drift": ("空间漂移示意", "component",
                          "源库与目标库直方图分布差异示意。",
                          "复用源资产柱形图元作直方图草图。", [792, 1044, 946, 84]),
    }
    for part_id, (name, role, meaning, reuse, box) in definitions.items():
        keys = collected.get(part_id, [])
        parts[part_id] = {
            "id": part_id,
            "name": name,
            "role": role,
            "box": box,
            "bounds": [box[0], box[1], box[2], box[3]],
            "meaning": meaning,
            "reuse": reuse,
            "element_keys": keys,
        }
    context["parts"] = parts
    return None


if __name__ == "__main__":
    run(__file__, customize)
