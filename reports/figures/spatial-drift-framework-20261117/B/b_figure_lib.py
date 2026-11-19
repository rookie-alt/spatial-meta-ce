"""Shared drawing helpers for candidate B of the spatial-drift framework figure.

The helpers work in the VA-196 viewBox coordinate space (1800 x 1200 units) and
reuse the source asset's real graphic language:

* cards are a twin pair (one filled path plus one stroked path with the same
  ``d``), rounded corners drawn with cubic segments;
* connectors are solid arrow-head paths (no stroke) or stroked polylines;
* panel frames are stroked rectangles with square corners;
* text is black or a single semantic accent colour, bold titles plus regular
  body labels.

Nothing here rasterises or rewrites the source asset; every drawn object keeps a
unique ``data-asset-key`` so it can be exported as a genuine vector part.
"""
from __future__ import annotations

import base64
from pathlib import Path
import xml.etree.ElementTree as ET

NS = "http://www.w3.org/2000/svg"

# ---------------------------------------------------------------- palette ---
# Source palette roles (VA-196): blue = forward / model stack, pink = backward
# or key, light orange = value / output, mint = norm / expectation, teal =
# update, cyan = data series. Reused here with the same semantics.
INK = "#042433"
BLACK = "#000000"
BLUE = "#315ab4"
BLUE_FILL = "#cddfff"
PINK = "#c25160"
PINK_FILL = "#ffd9d6"
PEACH = "#ffac49"
PEACH_FILL = "#ffecd6"
MINT = "#068c79"
MINT_FILL = "#d0e4e2"
PALE_MINT = "#dcf5f7"
GREEN_FILL = "#e6f8e3"
GREY_FILL = "#d9d9d9"
CYAN = "#00c8d3"

CANVAS_W = 1800.0
CANVAS_H = 1200.0

# Source stroke widths (viewBox units): panel frame 3, card outline 2,
# connector 1.5, divider 2.
FRAME_W = 3.0
CARD_W = 2.0
LINE_W = 1.5

_KEY_SEQ = [0]


def next_key(prefix: str = "b") -> str:
    """Allocate a unique, stable object key for a newly drawn element."""
    _KEY_SEQ[0] += 1
    return f"{prefix}{_KEY_SEQ[0]:04d}"


def reset_keys(prefix: str = "b") -> None:
    _KEY_SEQ[0] = 0


def _f(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def rounded_rect_path(x: float, y: float, w: float, h: float, r: float = 8.0) -> str:
    """Rounded rectangle path in the source's cubic-corner style."""
    r = max(0.0, min(r, w / 2, h / 2))
    x2, y2 = x + w, y + h
    return (
        f"M{_f(x)} {_f(y + r)}"
        f"C{_f(x)} {_f(y + r * 0.28)} {_f(x + r * 0.28)} {_f(y)} {_f(x + r)} {_f(y)}"
        f"H{_f(x2 - r)}"
        f"C{_f(x2 - r * 0.28)} {_f(y)} {_f(x2)} {_f(y + r * 0.28)} {_f(x2)} {_f(y + r)}"
        f"V{_f(y2 - r)}"
        f"C{_f(x2)} {_f(y2 - r * 0.28)} {_f(x2 - r * 0.28)} {_f(y2)} {_f(x2 - r)} {_f(y2)}"
        f"H{_f(x + r)}"
        f"C{_f(x + r * 0.28)} {_f(y2)} {_f(x)} {_f(y2 - r * 0.28)} {_f(x)} {_f(y2 - r)}"
        "Z"
    )


def arrow_path(x1: float, y1: float, x2: float, y2: float, head: float = 11.0,
              shaft: float = 3.0) -> str:
    """Solid right-pointing arrow, matching the source asset's arrow shapes."""
    return (
        f"M{_f(x1)} {_f(y1 - shaft)}H{_f(x2 - head)}V{_f(y1 - shaft * 1.9)}"
        f"L{_f(x2)} {_f(y1)}L{_f(x2 - head)} {_f(y1 + shaft * 1.9)}"
        f"V{_f(y1 + shaft)}H{_f(x1)}Z"
    )


def arrow_down_path(x: float, y1: float, y2: float, head: float = 11.0,
                    shaft: float = 3.0) -> str:
    """Solid down-pointing arrow (the source's vertical arrow idiom)."""
    return (
        f"M{_f(x - shaft)} {_f(y1)}H{_f(x + shaft)}V{_f(y2 - head)}"
        f"H{_f(x + shaft * 1.9)}L{_f(x)} {_f(y2)}L{_f(x - shaft * 1.9)} {_f(y2 - head)}"
        f"H{_f(x - shaft)}Z"
    )


def arrow_left_path(x1: float, y1: float, x2: float, y2: float, head: float = 11.0,
                    shaft: float = 3.0) -> str:
    """Solid left-pointing arrow (used for the outer-loop feedback)."""
    return (
        f"M{_f(x1)} {_f(y1 - shaft)}H{_f(x2 + head)}V{_f(y1 - shaft * 1.9)}"
        f"L{_f(x2)} {_f(y1)}L{_f(x2 + head)} {_f(y1 + shaft * 1.9)}"
        f"V{_f(y1 + shaft)}H{_f(x1)}Z"
    )


def polyline_path(points) -> str:
    parts = []
    for index, (x, y) in enumerate(points):
        parts.append(f"{'M' if index == 0 else 'L'} {_f(x)} {_f(y)}")
    return " ".join(parts)


class Canvas:
    """Builds SVG objects with the source asset's twin-pair / arrow language."""

    def __init__(self, root: ET.Element, tag, font_family: str = "NotoSansSC"):
        self.root = root
        self.tag = tag
        self.font_family = font_family
        self.bold_family = font_family
        self.objects: list[tuple[str, ET.Element]] = []

    def _append(self, element: ET.Element, key: str) -> ET.Element:
        self.objects.append((key, element))
        return element

    # --------------------------------------------------------------- cards --
    def card(self, x: float, y: float, w: float, h: float, fill: str = BLUE_FILL,
             stroke: str = INK, stroke_width: float = CARD_W, radius: float = 8.0,
             key: str | None = None) -> tuple[str, str]:
        """Draw a card as the source's fill+stroke twin pair."""
        path = rounded_rect_path(x, y, w, h, radius)
        fill_key = key or next_key("bf")
        stroke_key = next_key("bs")
        fill_el = ET.SubElement(self.root, self.tag("path"), {
            "d": path, "fill": fill, "fill-rule": "evenodd",
            "data-asset-key": fill_key,
        })
        self._append(fill_el, fill_key)
        stroke_el = ET.SubElement(self.root, self.tag("path"), {
            "d": path, "fill": "none", "stroke": stroke,
            "stroke-width": f"{stroke_width:g}", "stroke-linejoin": "miter",
            "stroke-linecap": "butt", "data-asset-key": stroke_key,
        })
        self._append(stroke_el, stroke_key)
        return fill_key, stroke_key

    def frame(self, x: float, y: float, w: float, h: float, stroke: str = BLACK,
              stroke_width: float = FRAME_W, dash: str | None = None) -> str:
        key = next_key("bf")
        attrs = {
            "d": f"M{_f(x)} {_f(y)}H{_f(x + w)}V{_f(y + h)}H{_f(x)}Z",
            "fill": "none", "stroke": stroke,
            "stroke-width": f"{stroke_width:g}", "stroke-linejoin": "miter",
            "data-asset-key": key,
        }
        if dash:
            attrs["stroke-dasharray"] = dash
        element = ET.SubElement(self.root, self.tag("path"), attrs)
        self._append(element, key)
        return key

    def rect(self, x: float, y: float, w: float, h: float, fill: str,
             stroke: str | None = None, stroke_width: float = CARD_W,
             key: str | None = None) -> str:
        key = key or next_key("bf")
        attrs = {
            "d": f"M{_f(x)} {_f(y)}H{_f(x + w)}V{_f(y + h)}H{_f(x)}Z",
            "fill": fill, "data-asset-key": key,
        }
        if stroke:
            attrs["stroke"] = stroke
            attrs["stroke-width"] = f"{stroke_width:g}"
            attrs["fill"] = "none"
        element = ET.SubElement(self.root, self.tag("path"), attrs)
        self._append(element, key)
        return key

    # ---------------------------------------------------------- connectors --
    def arrow(self, x1: float, y1: float, x2: float, y2: float, fill: str = BLACK,
              head: float = 11.0, shaft: float = 3.0,
              key: str | None = None) -> str:
        key = key or next_key("bl")
        element = ET.SubElement(self.root, self.tag("path"), {
            "d": arrow_path(x1, y1, x2, y2, head, shaft),
            "fill": fill, "data-asset-key": key,
        })
        self._append(element, key)
        return key

    def arrow_down(self, x: float, y1: float, y2: float, fill: str = BLACK,
                   head: float = 11.0, shaft: float = 3.0,
                   key: str | None = None) -> str:
        key = key or next_key("bl")
        element = ET.SubElement(self.root, self.tag("path"), {
            "d": arrow_down_path(x, y1, y2, head, shaft),
            "fill": fill, "data-asset-key": key,
        })
        self._append(element, key)
        return key

    def arrow_left(self, x1: float, y1: float, x2: float, y2: float, fill: str = BLACK,
                   head: float = 11.0, shaft: float = 3.0,
                   key: str | None = None) -> str:
        key = key or next_key("bl")
        element = ET.SubElement(self.root, self.tag("path"), {
            "d": arrow_left_path(x1, y1, x2, y2, head, shaft),
            "fill": fill, "data-asset-key": key,
        })
        self._append(element, key)
        return key

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = INK,
             stroke_width: float = LINE_W, dash: str | None = None,
             key: str | None = None) -> str:
        key = key or next_key("bl")
        attrs = {
            "d": f"M{_f(x1)} {_f(y1)}L{_f(x2)} {_f(y2)}",
            "fill": "none", "stroke": stroke,
            "stroke-width": f"{stroke_width:g}", "stroke-linecap": "butt",
            "data-asset-key": key,
        }
        if dash:
            attrs["stroke-dasharray"] = dash
        element = ET.SubElement(self.root, self.tag("path"), attrs)
        self._append(element, key)
        return key

    def polyline(self, points, stroke: str = INK, stroke_width: float = LINE_W,
                 dash: str | None = None, key: str | None = None) -> str:
        key = key or next_key("bl")
        attrs = {
            "d": polyline_path(points), "fill": "none", "stroke": stroke,
            "stroke-width": f"{stroke_width:g}", "stroke-linejoin": "miter",
            "data-asset-key": key,
        }
        if dash:
            attrs["stroke-dasharray"] = dash
        element = ET.SubElement(self.root, self.tag("path"), attrs)
        self._append(element, key)
        return key

    # --------------------------------------------------------------- text ---
    def text(self, x: float, y: float, value: str, size: float = 24.0,
             weight: str = "400", fill: str = BLACK, anchor: str = "start",
             family: str | None = None, key: str | None = None,
             letter_spacing: float | None = None) -> str:
        """Add a text run. ``y`` is the baseline, matching SVG text semantics."""
        key = key or next_key("bt")
        family = family or (self.bold_family if weight in ("700", "bold") else
                            self.font_family)
        element = ET.SubElement(self.root, self.tag("text"), {
            "x": _f(x), "y": _f(y), "font-size": f"{size:g}",
            "font-family": family, "font-weight": weight, "fill": fill,
            "text-anchor": anchor, "data-asset-key": key,
        })
        if letter_spacing is not None:
            element.set("letter-spacing", f"{letter_spacing:g}")
        element.text = value
        self._append(element, key)
        return key

    def text_box(self, x: float, y: float, lines, size: float = 24.0,
                 weight: str = "400", fill: str = BLACK, anchor: str = "start",
                 line_height: float = 1.45, family: str | None = None) -> list[str]:
        """Draw a multi-line label; returns the created keys."""
        keys = []
        for index, line in enumerate(lines):
            keys.append(self.text(
                x, y + index * size * line_height, line, size=size, weight=weight,
                fill=fill, anchor=anchor, family=family))
        return keys

    def label_in(self, x: float, y: float, w: float, h: float, value: str,
                 size: float = 24.0, weight: str = "400", fill: str = BLACK) -> str:
        """Centre a single line of text inside a box."""
        return self.text(x + w / 2, y + h / 2 + size * 0.36, value, size=size,
                         weight=weight, fill=fill, anchor="middle")


def embed_fonts(root: ET.Element, font_dir: Path,
                families: list[tuple[str, str, str]]) -> None:
    """Embed font files so the SVG and its PDF stay self-contained.

    Each entry is (family, filename, weight). Several files may share one
    family as long as they declare different weights, so bold text selects the
    real bold face instead of a synthesised one.
    """
    rules = []
    for family, filename, weight in families:
        path = font_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Font file missing: {path}")
        uri = base64.b64encode(path.read_bytes()).decode()
        rules.append(
            f"@font-face{{font-family:\"{family}\";"
            f"font-weight:{weight};font-style:normal;"
            f"src:url(data:font/otf;base64,{uri}) format(\"opentype\");}}"
        )
    style = ET.SubElement(root, "{http://www.w3.org/2000/svg}style",
                          {"id": "vivid-replacement-fonts"})
    style.text = "\n".join(rules)


def measure_text_width(text: str, size: float) -> float:
    """Approximate text width for CJK/latin mixes, used for layout centering."""
    width = 0.0
    for char in text:
        if ord(char) > 0x2E80:
            width += 1.0
        elif char in "iljI.,:;'|!()[]":
            width += 0.32
        elif char in "mwMW@":
            width += 0.85
        elif char == " ":
            width += 0.28
        elif char.isupper():
            width += 0.62
        else:
            width += 0.52
    return width * size

