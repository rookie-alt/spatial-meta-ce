"""VA-196 derived framework figure for the Spatial Meta-CE protocol 125 route.

Source asset: VA-196 (architecture.source.mlsys26-charon_fig2). The panel logic of
the source vector objects is preserved -- three stacked black frames, each split
into a left sub-panel and a right sub-panel by a dashed divider, rounded layer
chips, mint parameter bars, cyan auxiliary chips, the coloured step accents and
the arrowhead/dashed-connector vocabulary. Every object is re-derived for the
current semantics: GPU parallelism, MoE, KV-cache and the cost chart are removed
because they have no basis in this task.

Entry point for the delivered figure:
    python3 make_figure.py --config figure.json --out . --parts
"""
from pathlib import Path
import base64
import copy
import io
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET

from fontTools.pens.basePen import BasePen
from fontTools.cu2qu import curves_to_quadratic

try:
    from source_asset import run
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from source_asset import run


SVG_NS = 'http://www.w3.org/2000/svg'
CANVAS_W, CANVAS_H = 1800.0, 1200.0
FONT = 'Source Han Sans CN'

INK = '#000000'
DARK = '#042433'
BLUE = '#315ab4'
RED = '#c25160'
GREEN = '#068c79'

C_SOURCE = '#cddfff'
C_TARGET = '#ffd9d6'
C_SUPPORT = '#e6f8e3'
C_QUERY = '#ffecd6'
C_PARAM = '#d0e4e2'
C_AUX = '#dcf5f7'
C_NEUTRAL = '#d9d9d9'

FONT_SOURCES = {
    'Regular': 'https://mirrors.tuna.tsinghua.edu.cn/adobe-fonts/source-han-sans/SubsetOTF/CN/SourceHanSansCN-Regular.otf',
    'Bold': 'https://mirrors.tuna.tsinghua.edu.cn/adobe-fonts/source-han-sans/SubsetOTF/CN/SourceHanSansCN-Bold.otf',
}


class _QuadPen(BasePen):
    """Rewrites cubic CFF segments into quadratic ones for a TrueType glyf table."""

    def __init__(self, out, max_err):
        super().__init__(None)
        self.out = out
        self.max_err = max_err

    def _moveTo(self, point):
        self.out.moveTo(point)

    def _lineTo(self, point):
        self.out.lineTo(point)

    def _curveToOne(self, p1, p2, p3):
        for quad in curves_to_quadratic(
                [(self._getCurrentPoint(), p1, p2, p3)], [self.max_err]):
            self.out.qCurveTo(quad[1], quad[2])

    def _closePath(self):
        self.out.closePath()

    def _endPath(self):
        self.out.endPath()


def _cff_to_glyf(font) -> None:
    """Convert CFF outlines to TrueType so Chromium embeds a real Type0 PDF font."""
    from fontTools.ttLib import newTable
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.pens.boundsPen import BoundsPen

    glyph_set = font.getGlyphSet()
    order = font.getGlyphOrder()
    glyf = newTable('glyf')
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(_QuadPen(pen, 1.0))
        glyph = pen.glyph()
        bounds = BoundsPen(glyph_set)
        glyph_set[name].draw(bounds)
        glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax = [
            int(round(value)) for value in (bounds.bounds or (0, 0, 0, 0))]
        glyf.glyphs[name] = glyph
    font.tables['glyf'] = glyf
    glyf.glyphOrder = order
    font['maxp'].tableVersion = 0x00010000
    font['maxp'].numGlyphs = len(order)
    font.tables['loca'] = newTable('loca')
    try:
        font['maxp'].recalc(font)
    except Exception:
        pass
    for attr in ('maxPoints', 'maxContours', 'maxCompositePoints',
                 'maxCompositeContours', 'maxZones', 'maxTwilightPoints',
                 'maxStorage', 'maxFunctionDefs', 'maxInstructionDefs',
                 'maxStackElements', 'maxSizeOfInstructions',
                 'maxComponentElements', 'maxComponentDepth'):
        if not hasattr(font['maxp'], attr):
            setattr(font['maxp'], attr, 0)
    del font['CFF ']
    for tag in ('VORG', 'vhea', 'vmtx'):
        if tag in font:
            del font[tag]
    metrics = font['hmtx']
    for name in order:
        metrics.metrics[name] = (metrics.metrics[name][0], 0)


def ensure_fonts(font_dir: Path, chars: str) -> dict:
    """Subset the two CJK weights to the characters actually used and return bytes."""
    from fontTools import subset
    cache = Path.home() / '.cache' / 'vivid-spatial-meta-ce-fonts'
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    keep = sorted(set(chars) | set(
        '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        '=+*().,/:-;\'\"!?%&#@[]{}<>|~^_`$\\'))
    for weight, url in FONT_SOURCES.items():
        target = font_dir / f'SourceHanSansCN-{weight}.otf'
        if target.exists():
            out[weight] = target.read_bytes()
            continue
        raw_path = cache / f'SourceHanSansCN-{weight}.otf'
        if not raw_path.exists():
            with urllib.request.urlopen(url, timeout=180) as response:
                raw_path.write_bytes(response.read())
        options = subset.Options()
        options.layout_features = ['*']
        options.name_IDs = ['*']
        options.notdef_outline = True
        options.recode_borders = True
        options.drop_tables = ['DSIG']
        font = subset.load_font(io.BytesIO(raw_path.read_bytes()), options)
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=''.join(keep))
        subsetter.subset(font)
        _cff_to_glyf(font)
        buf = io.BytesIO()
        font.save(buf)
        data = buf.getvalue()
        target.write_bytes(data)
        out[weight] = data
    return out


class Builder:
    """Places reused VA-196 vector objects and new objects on the 1800x1200 canvas."""

    def __init__(self, root, context):
        self.root = root
        self.context = context
        self.tag = context['tag']
        self.geom = context['geometry']
        self.meta = {e['key']: e for e in self.geom['elements']}
        self.src = {}
        for element in root.iter():
            key = element.get('data-asset-key')
            if key:
                self.src[key] = element
        self.parts = {}
        self.counter = 0
        self.texts = []
        self.layer = ET.Element(self.tag('g'), {'id': 'vivid-figure'})

    # -- registry helpers -------------------------------------------------
    def key(self):
        self.counter += 1
        return f'a{self.counter:04d}'

    def part(self, pid, name, description):
        self.parts[pid] = {
            'id': pid, 'name': name, 'description': description, 'element_keys': [],
        }
        return pid

    def own(self, key, pid=None):
        if pid and pid in self.parts:
            self.parts[pid]['element_keys'].append(key)

    def add(self, element, pid=None, cid=None):
        key = self.key()
        element.set('data-asset-key', key)
        if cid:
            element.set('id', cid)
        self.layer.append(element)
        self.own(key, pid)
        return key

    # -- reused source objects -------------------------------------------
    def place_src(self, key, x, y, w, h, pid=None, cid=None,
                  fill=None, stroke=None, sw=None, dash=None, opacity=None):
        source = self.src[key]
        transform = source.get('transform', '')
        if not transform.startswith('matrix(1,0,0,-1'):
            raise ValueError(f'{key} is not a y-flipped path object: {transform!r}')
        box = self.meta[key]['bounds']
        # Degenerate source extents (a purely horizontal or vertical divider) cannot be
        # scaled from the measured box; fall back to a neutral unit scale.
        sx = w / box[2] if box[2] > 1e-9 else 1.0
        sy = h / box[3] if box[3] > 1e-9 else 1.0
        a = x - sx * (454.257 + box[0])
        c = y + sy * (1081.31 - box[1])
        clone = copy.deepcopy(source)
        clone.set('transform', f'translate({a:.8g},{c:.8g}) scale({sx:.8g},{-sy:.8g})')
        clone.attrib.pop('clip-path', None)
        for attr, value in (('fill', fill), ('stroke', stroke),
                            ('stroke-width', sw), ('stroke-dasharray', dash),
                            ('opacity', opacity)):
            if value is not None:
                clone.set(attr, str(value))
                if attr in ('fill', 'stroke', 'opacity'):
                    style = ';'.join(p for p in clone.get('style', '').split(';')
                                     if p.strip() and not p.strip().startswith(attr + ':'))
                    clone.set('style', f'{style};{attr}:{value}' if style else f'{attr}:{value}')
        return self.add(clone, pid, cid)

    def chip_pair(self, pair, x, y, w, h, fill, pid=None, cid=None, sw=2.5):
        """Place the source's own fill+stroke construction for a rounded chip."""
        fill_key, stroke_key = pair
        keys = [self.place_src(fill_key, x, y, w, h, pid=pid, cid=cid, fill=fill),
                self.place_src(stroke_key, x, y, w, h, pid=pid, sw=sw)]
        return keys

    # -- new objects ------------------------------------------------------
    def rrect(self, x, y, w, h, rx=12, fill='none', stroke=INK, sw=2, dash=None,
              pid=None, cid=None, opacity=None):
        element = ET.Element(self.tag('path'), {
            'd': (f'M{x + rx} {y}H{x + w - rx}a{rx} {rx} 0 0 1 {rx} {rx}'
                  f'V{y + h - rx}a{rx} {rx} 0 0 1 {-rx} {rx}H{x + rx}'
                  f'a{rx} {rx} 0 0 1 {-rx} {-rx}V{y + rx}a{rx} {rx} 0 0 1 {rx} {-rx}Z'),
            'fill': fill, 'stroke': stroke, 'stroke-width': str(sw),
            'stroke-linejoin': 'miter',
        })
        if dash:
            element.set('stroke-dasharray', dash)
        if opacity is not None:
            element.set('opacity', str(opacity))
        return self.add(element, pid, cid)

    def rect(self, x, y, w, h, fill=INK, pid=None, cid=None):
        element = ET.Element(self.tag('path'), {
            'd': f'M{x} {y}H{x + w}V{y + h}H{x}Z', 'fill': fill})
        return self.add(element, pid, cid)

    def line(self, x1, y1, x2, y2, pid=None, cid=None, sw=2.5, stroke=INK,
             dash=None, frm=None, to=None):
        element = ET.Element(self.tag('path'), {
            'd': f'M{x1} {y1}L{x2} {y2}', 'fill': 'none', 'stroke': stroke,
            'stroke-width': str(sw), 'stroke-linecap': 'butt'})
        if dash:
            element.set('stroke-dasharray', dash)
        if frm:
            element.set('data-from', frm)
        if to:
            element.set('data-to', to)
        return self.add(element, pid, cid)

    def text(self, x, y, size, content, pid=None, weight=700, fill=INK,
             anchor='start', box=None, cid=None):
        element = ET.Element(self.tag('text'), {
            'x': f'{x:.4g}', 'y': f'{y:.4g}', 'font-size': str(size),
            'font-family': FONT, 'font-weight': str(weight), 'fill': fill,
            'text-anchor': anchor, '{http://www.w3.org/XML/1998/namespace}space': 'preserve',
        })
        element.text = content
        if box:
            element.set('data-text-box', ' '.join(f'{v:.4g}' for v in box))
        self.texts.append((content, size, box))
        return self.add(element, pid, cid)

    def text_rich(self, x, y, size, runs, pid=None, weight=700, fill=INK,
                  anchor='start', box=None):
        """Runs of (text, subscript) so phi_0 / phi_5 / phi_t can be shown without
        Unicode subscript glyphs, which Source Han Sans CN does not provide."""
        element = ET.Element(self.tag('text'), {
            'x': f'{x:.4g}', 'y': f'{y:.4g}', 'font-size': str(size),
            'font-family': FONT, 'font-weight': str(weight), 'fill': fill,
            'text-anchor': anchor,
            '{http://www.w3.org/XML/1998/namespace}space': 'preserve'})
        for content, subscript in runs:
            span = ET.SubElement(element, self.tag('tspan'), {
                'font-size': str(round(size * 0.62, 4)) if subscript else str(size),
                'font-family': FONT, 'font-weight': str(weight), 'fill': fill,
                'baseline-shift': 'sub' if subscript else 'baseline'})
            span.text = content
        if box:
            element.set('data-text-box', ' '.join(f'{v:.4g}' for v in box))
        self.texts.append((''.join(c for c, _ in runs), size, box))
        return self.add(element, pid)

    # -- arrows -----------------------------------------------------------
    def arrow_down(self, x_center, y_top, height, pid=None, frm=None, to=None,
                   color=DARK):
        width = 9.0
        key = self.place_src('e0026', x_center - width / 2, y_top, width, height,
                             pid=pid, fill=color)
        element = self.layer[-1]
        if frm:
            element.set('data-from', frm)
        if to:
            element.set('data-to', to)
        return key

    def arrow_right(self, x_left, y_center, length, pid=None, frm=None, to=None,
                    color=DARK):
        height = length * 6.0 / 16.6
        key = self.place_src('e0235', x_left, y_center - height / 2, length, height,
                             pid=pid, fill=color)
        element = self.layer[-1]
        if frm:
            element.set('data-from', frm)
        if to:
            element.set('data-to', to)
        return key

    def arrow_up_head(self, x_center, y_tip, size=11.0, pid=None, color=DARK):
        height = size * 13.8 / 9.0
        return self.place_src('e0028', x_center - size / 2, y_tip, size, height,
                              pid=pid, fill=color)

    def arrow_up_head_semantic(self, x_center, y_tip, pid=None, frm=None, to=None):
        placed = self.place_src('e0028', x_center - 5.5, y_tip, 11.0, 16.9, pid=pid,
                                fill=DARK)
        element = self.layer[-1]
        if frm:
            element.set('data-from', frm)
        if to:
            element.set('data-to', to)
        return placed

    # -- finalisation -----------------------------------------------------
    def finish(self, root):
        root.append(self.layer)
        for element in list(root):
            if element is self.layer:
                continue
            if element.tag.split('}')[-1] == 'defs':
                for child in list(element):
                    element.remove(child)
            elif element.tag.split('}')[-1] in ('g', 'style'):
                root.remove(element)
        return root


def customize(root, context):
    builder = Builder(root, context)
    tag = builder.tag

    FRAME_A = (24.0, 24.0, 1752.0, 344.0)
    FRAME_B = (24.0, 390.0, 1752.0, 396.0)
    FRAME_C = (24.0, 846.0, 1752.0, 330.0)
    DIVIDER_X = 884.0

    builder.part('panel-frames', '面板外框', '三块黑色外框与源域/目标域区域标签')
    builder.part('source-task', '源域任务划分', '源域数据库容器、支持集/查询集不相交划分')
    builder.part('price-pathway', 'PRICE 特征与模型通路', '四组特征、编码器、回归头与辅助输入')
    builder.part('fomaml-inner', 'FOMAML 内循环', 'phi_0=theta、5 步全参数 SGD、phi_5')
    builder.part('fomaml-outer', 'FOMAML 外循环', '查询损失、一阶梯度、Adam 更新共享 theta')
    builder.part('loop-legend', '标签流向与外循环配置', '标签流向图例与外循环批量/步数说明')
    builder.part('leakage-boundary', '无泄漏边界', '源域与目标域之间的分隔线与边界标签')
    builder.part('target-flow', '目标域适应与评估', '目标支持集、phi_t 副本、K-shot 路径与评分')
    builder.part('pem-comparison', 'P/E/M 计划比较', 'P、E、M 三条并列与计划状态说明')

    # ---------------------------------------------------------------- frames
    for index, frame in enumerate((FRAME_A, FRAME_B, FRAME_C)):
        x, y, w, h = frame
        builder.place_src(['e0004', 'e0005', 'e0006'][index], x, y, w, h,
                          pid='panel-frames', cid=f'frame-{"abc"[index]}')

    for letter, frame, label, tint, cid in (
        ('a', FRAME_A, '源域 SOURCE · 元训练', C_SOURCE, 'tag-source-a'),
        ('b', FRAME_B, '源域 SOURCE · 元训练', C_SOURCE, 'tag-source-b'),
        ('c', FRAME_C, '目标域 TARGET · 评估', C_TARGET, 'tag-target-c'),
    ):
        fx, fy, fw, fh = frame
        tx, tw = fx + fw - 380.0, 356.0
        builder.chip_pair(('e0011', 'e0012'), tx, fy + 12.0, tw, 46.0, tint,
                          pid='panel-frames', cid=cid)
        builder.text(tx + tw / 2, fy + 43.0, 24, label, pid='panel-frames',
                     anchor='middle', box=(tx, fy + 12.0, tw, 46.0))

    for frame in (FRAME_A, FRAME_B, FRAME_C):
        fx, fy, fw, fh = frame
        builder.place_src('e0181', DIVIDER_X, fy + 54.0, 2.0, fh - 68.0,
                          pid='panel-frames', cid=None, dash='8,6,2,6', sw=2)

    for letter, frame, title in (
        ('a', FRAME_A, '源域任务构建与原始 PRICE 特征/模型通路'),
        ('b', FRAME_B, 'FOMAML 嵌套循环：内循环更新 φ，外循环更新共享 θ'),
        ('c', FRAME_C, '目标域支持集适应与不相交唯一查询评估'),
    ):
        fx, fy, fw, fh = frame
        builder.text(fx + 20.0, fy + 38.0, 30, f'({letter})', pid='panel-frames')
        builder.text(fx + 76.0, fy + 38.0, 32, title, pid='panel-frames')

    # ------------------------------------------------------------ panel (a)
    left_x, left_y, left_w, left_h = 48.0, 78.0, 820.0, 278.0
    builder.place_src('e0002', left_x, left_y, left_w, left_h, pid='source-task',
                      cid='source-region', fill=C_SOURCE, opacity=0.22)
    builder.place_src('e0003', left_x, left_y, left_w, left_h, pid='source-task', sw=2.5)
    builder.text(left_x + 24.0, left_y + 36.0, 28, '源域数据库（26 个训练库）',
                 pid='source-task')

    sup_x, sup_y, sup_w, sup_h = 76.0, 132.0, 316.0, 84.0
    qry_x = 472.0
    builder.chip_pair(('e0030', 'e0031'), sup_x, sup_y, sup_w, sup_h, C_SUPPORT,
                      pid='source-task', cid='chip-support')
    builder.text(sup_x + 20.0, sup_y + 40.0, 28, '支持集 support', pid='source-task',
                 box=(sup_x, sup_y, sup_w, sup_h))
    builder.text(sup_x + 20.0, sup_y + 70.0, 24, '20 条查询 · 进入内循环',
                 pid='source-task', box=(sup_x, sup_y, sup_w, sup_h))

    builder.chip_pair(('e0034', 'e0035'), qry_x, sup_y, sup_w, sup_h, C_QUERY,
                      pid='source-task', cid='chip-query')
    builder.text(qry_x + 20.0, sup_y + 40.0, 28, '查询集 query', pid='source-task',
                 box=(qry_x, sup_y, sup_w, sup_h))
    builder.text(qry_x + 20.0, sup_y + 70.0, 24, '64 条查询 · 进入外层损失',
                 pid='source-task', box=(qry_x, sup_y, sup_w, sup_h))

    builder.place_src('e0181', 432.0, 140.0, 2.0, 68.0, pid='source-task', sw=2)
    builder.text(432.0, 180.0, 24, '不相交', pid='source-task', anchor='middle')

    for offset, note in enumerate((
        '支持集与查询集在任务内按 query_hash 不相交',
        '每任务等概率取一个源库；无独立源验证库',
        '不据此早停或选择 checkpoint',
    )):
        builder.text(left_x + 28.0, 252.0 + offset * 34.0, 24, note, pid='source-task')

    builder.text(904.0, 114.0, 28, '原始 PRICE 架构（结构与预训练权重不变）',
                 pid='price-pathway')

    features = (('连接列直方图', C_SUPPORT, 'feat-join'),
                ('扇出 fanout', C_TARGET, 'feat-fanout'),
                ('表特征', C_QUERY, 'feat-table'),
                ('过滤特征', C_AUX, 'feat-filter'))
    chip_w, chip_h, gap = 200.0, 60.0, 17.0
    for index, (label, tint, cid) in enumerate(features):
        cx = 904.0 + index * (chip_w + gap)
        builder.chip_pair(('e0030', 'e0031'), cx, 130.0, chip_w, chip_h, tint,
                          pid='price-pathway', cid=cid)
        builder.text(cx + chip_w / 2, 130.0 + 38.0, 26, label, pid='price-pathway',
                     anchor='middle', box=(cx, 130.0, chip_w, chip_h))
        builder.arrow_down(cx + chip_w / 2, 190.0, 14.0, pid='price-pathway',
                           frm=cid, to='encoder')

    builder.chip_pair(('e0011', 'e0012'), 904.0, 204.0, 851.0, 62.0, C_SOURCE,
                      pid='price-pathway', cid='encoder', sw=2.5)
    builder.text(904.0 + 851.0 / 2, 244.0, 28, 'PRICE 编码器（结构不变）',
                 pid='price-pathway', anchor='middle', box=(904.0, 204.0, 851.0, 62.0))

    builder.arrow_down(1124.0, 266.0, 22.0, pid='price-pathway',
                       frm='encoder', to='head')
    builder.chip_pair(('e0038', 'e0039'), 904.0, 288.0, 440.0, 64.0, C_PARAM,
                      pid='price-pathway', cid='head', sw=2.5)
    builder.text(1124.0, 320.0, 28, '回归头', pid='price-pathway', anchor='middle',
                 box=(904.0, 288.0, 440.0, 64.0))
    builder.text(1124.0, 344.0, 22, 'z = log1p(cardinality) + 1，MSE',
                 pid='price-pathway', anchor='middle', box=(904.0, 288.0, 440.0, 64.0))

    builder.line(1344.0, 320.0, 1444.0, 320.0, pid='price-pathway', sw=1.5,
                 dash='4.5,1.5', frm='aux-input', to='head')
    builder.chip_pair(('e0068', 'e0069'), 1444.0, 288.0, 311.0, 64.0, C_AUX,
                      pid='price-pathway', cid='aux-input', sw=2.5)
    builder.text(1599.5, 326.0, 24, 'PG 估计 + token 计数', pid='price-pathway',
                 anchor='middle', box=(1444.0, 288.0, 311.0, 64.0))

    # ------------------------------------------------------------ panel (b)
    builder.text(56.0, 476.0, 24, '①', pid='fomaml-inner', fill=BLUE)
    builder.text(100.0, 480.0, 28, '内循环 · 只用支持集标签', pid='fomaml-inner')
    builder.text(916.0, 476.0, 24, '②', pid='fomaml-outer', fill=RED)
    builder.text(960.0, 480.0, 28, '外循环 · 只用查询集标签', pid='fomaml-outer')
    builder.text(1522.0, 476.0, 24, '③', pid='fomaml-outer', fill=GREEN)

    cy, ch = 498.0, 84.0
    builder.chip_pair(('e0038', 'e0039'), 56.0, cy, 236.0, ch, C_PARAM,
                      pid='fomaml-inner', cid='phi0', sw=2.5)
    builder.text_rich(76.0, cy + 48.0, 30, [('φ', False), ('0', True),
                                            (' = θ', False)],
                      pid='fomaml-inner', box=(56.0, cy, 236.0, ch))
    builder.text(76.0, cy + 76.0, 24, '共享参数副本', pid='fomaml-inner',
                 box=(56.0, cy, 236.0, ch))
    builder.arrow_right(300.0, cy + ch / 2, 30.0, pid='fomaml-inner',
                        frm='phi0', to='inner-step')
    builder.chip_pair(('e0030', 'e0031'), 340.0, cy, 256.0, ch, C_SUPPORT,
                      pid='fomaml-inner', cid='inner-step', sw=2.5)
    builder.text(360.0, cy + 48.0, 26, '5 步全参数 SGD', pid='fomaml-inner',
                 box=(340.0, cy, 256.0, ch))
    builder.text(360.0, cy + 76.0, 24, 'α = 1e-6', pid='fomaml-inner',
                 box=(340.0, cy, 256.0, ch))
    builder.arrow_right(604.0, cy + ch / 2, 30.0, pid='fomaml-inner',
                        frm='inner-step', to='phi5')
    builder.chip_pair(('e0068', 'e0069'), 644.0, cy, 224.0, ch, C_AUX,
                      pid='fomaml-inner', cid='phi5', sw=2.5)
    builder.text_rich(664.0, cy + 48.0, 26, [('φ', False), ('5', True),
                                              (' 适应后参数', False)],
                      pid='fomaml-inner', box=(644.0, cy, 224.0, ch))
    builder.text(664.0, cy + 76.0, 24, '全参数', pid='fomaml-inner',
                 box=(644.0, cy, 224.0, ch))

    builder.chip_pair(('e0034', 'e0035'), 916.0, cy, 300.0, ch, C_QUERY,
                      pid='fomaml-outer', cid='query-loss', sw=2.5)
    builder.text(936.0, cy + 48.0, 26, '查询集损失', pid='fomaml-outer',
                 box=(916.0, cy, 300.0, ch))
    builder.text(936.0, cy + 76.0, 24, '适应后计算 MSE', pid='fomaml-outer',
                 box=(916.0, cy, 300.0, ch))
    builder.arrow_right(1224.0, cy + ch / 2, 30.0, pid='fomaml-outer',
                        frm='query-loss', to='outer-grad')
    builder.chip_pair(('e0032', 'e0033'), 1264.0, cy, 210.0, ch, C_TARGET,
                      pid='fomaml-outer', cid='outer-grad', sw=2.5)
    builder.text(1284.0, cy + 48.0, 26, '一阶梯度', pid='fomaml-outer',
                 box=(1264.0, cy, 210.0, ch))
    builder.text(1284.0, cy + 76.0, 24, '回传经过 φ', pid='fomaml-outer',
                 box=(1264.0, cy, 210.0, ch))
    builder.arrow_right(1482.0, cy + ch / 2, 30.0, pid='fomaml-outer',
                        frm='outer-grad', to='adam-update')
    builder.chip_pair(('e0011', 'e0012'), 1522.0, cy, 234.0, ch, C_SOURCE,
                      pid='fomaml-outer', cid='adam-update', sw=2.5)
    builder.text(1542.0, cy + 48.0, 26, 'Adam 更新 θ', pid='fomaml-outer',
                 box=(1522.0, cy, 234.0, ch))
    builder.text(1542.0, cy + 76.0, 24, 'lr = 2.85e-5', pid='fomaml-outer',
                 box=(1522.0, cy, 234.0, ch))

    builder.line(1639.0, 582.0, 1639.0, 614.0, pid='fomaml-outer', sw=2.5,
                 frm='adam-update', to='phi0')
    builder.line(1639.0, 614.0, 174.0, 614.0, pid='fomaml-outer', sw=2.5,
                 frm='adam-update', to='phi0')
    builder.line(174.0, 614.0, 174.0, 596.0, pid='fomaml-outer', sw=2.5,
                 frm='adam-update', to='phi0')
    builder.arrow_up_head_semantic(174.0, 584.0, pid='fomaml-outer',
                                   frm='adam-update', to='phi0')

    builder.text(56.0, 648.0, 24, '内循环在支持集上算 MSE；一阶近似下内层梯度被 detach',
                 pid='fomaml-inner')
    builder.text(56.0, 682.0, 24, '内循环不改变 θ，也不改变已封存的 θ*', pid='fomaml-inner')

    builder.rrect(56.0, 700.0, 1700.0, 74.0, rx=10, fill='#f4f4f4', stroke=DARK,
                  sw=1.5, pid='loop-legend', cid='loop-legend-box')
    for x, tint, note in ((76.0, C_SUPPORT, '支持集标签只进入内循环'),
                          (560.0, C_QUERY, '查询集标签只进入外层损失'),
                          (1050.0, C_PARAM, 'θ 由外循环更新，P 与 θ* 不被更新')):
        builder.rect(x, 712.0, 22.0, 22.0, fill=tint, pid='loop-legend')
        builder.text(x + 32.0, 730.0, 24, note, pid='loop-legend')
    builder.text(76.0, 760.0, 24,
                 '每次外更新串行累积 4 个任务；共 2000 次外更新；固定取最后一次 checkpoint，'
                 '不早停、不选最优', pid='loop-legend')

    # ------------------------------------------------------------- boundary
    builder.line(40.0, 816.0, 1760.0, 816.0, pid='leakage-boundary', sw=2,
                 dash='8,6,2,6')
    builder.rrect(650.0, 794.0, 500.0, 44.0, rx=10, fill='#ffffff', stroke=INK,
                  sw=2, pid='leakage-boundary', cid='boundary-chip')
    builder.text(900.0, 824.0, 26, '无泄漏边界：目标库不进入训练与选择',
                 pid='leakage-boundary', anchor='middle', box=(650.0, 794.0, 500.0, 44.0))

    # ------------------------------------------------------------ panel (c)
    builder.place_src('e0002', 32.0, 854.0, 1736.0, 314.0, pid='target-flow',
                      cid='target-region', fill=C_TARGET, opacity=0.16)
    builder.text(56.0, 926.0, 28, '目标库（不参与训练与选择）', pid='target-flow')

    tcy, tch = 946.0, 84.0
    builder.chip_pair(('e0030', 'e0031'), 56.0, tcy, 210.0, tch, C_SUPPORT,
                      pid='target-flow', cid='target-support', sw=2.5)
    builder.text(76.0, tcy + 48.0, 26, '目标支持集', pid='target-flow',
                 box=(56.0, tcy, 210.0, tch))
    builder.text(76.0, tcy + 76.0, 24, '仅用于适应', pid='target-flow',
                 box=(56.0, tcy, 210.0, tch))
    builder.arrow_right(274.0, tcy + tch / 2, 30.0, pid='target-flow',
                        frm='target-support', to='phi-t')
    builder.chip_pair(('e0038', 'e0039'), 312.0, tcy, 252.0, tch, C_PARAM,
                      pid='target-flow', cid='phi-t', sw=2.5)
    builder.text_rich(332.0, tcy + 48.0, 26, [('φ', False), ('t', True),
                                              (' = θ* 副本', False)],
                      pid='target-flow', box=(312.0, tcy, 252.0, tch))
    builder.text(332.0, tcy + 76.0, 24, '全参数适应', pid='target-flow',
                 box=(312.0, tcy, 252.0, tch))
    builder.arrow_right(572.0, tcy + tch / 2, 30.0, pid='target-flow',
                        frm='phi-t', to='target-estimate')
    builder.chip_pair(('e0068', 'e0069'), 610.0, tcy, 258.0, tch, C_AUX,
                      pid='target-flow', cid='target-estimate', sw=2.5)
    builder.text(630.0, tcy + 48.0, 26, '估计测试查询', pid='target-flow',
                 box=(610.0, tcy, 258.0, tch))
    builder.text(630.0, tcy + 76.0, 24, '不相交唯一查询', pid='target-flow',
                 box=(610.0, tcy, 258.0, tch))

    builder.text(56.0, 1084.0, 26, '适应强度 K（含 K = 0）', pid='target-flow')
    builder.chip_pair(('e0030', 'e0031'), 300.0, 1046.0, 140.0, 56.0, C_SUPPORT,
                      pid='target-flow', cid='kshot-positive', sw=2)
    builder.text(370.0, 1084.0, 24, 'K > 0', pid='target-flow', anchor='middle',
                 box=(300.0, 1046.0, 140.0, 56.0))
    builder.chip_pair(('e0032', 'e0033'), 460.0, 1046.0, 180.0, 56.0, C_TARGET,
                      pid='target-flow', cid='kshot-zero', sw=2)
    builder.text(550.0, 1084.0, 24, 'K = 0 零样本', pid='target-flow', anchor='middle',
                 box=(460.0, 1046.0, 180.0, 56.0))
    builder.text(56.0, 1132.0, 24, 'K 的具体取值按最终目标协议确定，本图不预设',
                 pid='target-flow')
    builder.text(56.0, 1160.0, 24, '目标支持集与测试查询不相交；测试标签只用于最终评分',
                 pid='target-flow')

    builder.text(916.0, 926.0, 28, 'P / E / M 并列比较（计划）', pid='pem-comparison')
    for index, (letter, lines, tint, cid) in enumerate((
        ('P', ('官方 PRICE 权重', '不继续训练'), C_NEUTRAL, 'pem-p'),
        ('E', ('普通继续训练', '监督 support+query'), C_SOURCE, 'pem-e'),
        ('M', ('全参数 FOMAML', '即本图 (b)'), C_SUPPORT, 'pem-m'),
    )):
        cx = 916.0 + index * 284.0
        builder.chip_pair(('e0030', 'e0031'), cx, 946.0, 264.0, 120.0, tint,
                          pid='pem-comparison', cid=cid, sw=2.5)
        builder.text(cx + 132.0, 996.0, 34, letter, pid='pem-comparison',
                     anchor='middle', box=(cx, 946.0, 264.0, 120.0))
        builder.text(cx + 132.0, 1030.0, 24, lines[0], pid='pem-comparison',
                     anchor='middle', box=(cx, 946.0, 264.0, 120.0))
        builder.text(cx + 132.0, 1058.0, 22, lines[1], pid='pem-comparison',
                     anchor='middle', box=(cx, 946.0, 264.0, 120.0))

    builder.rrect(916.0, 1086.0, 832.0, 78.0, rx=10, fill='#ffffff', stroke=DARK,
                  sw=1.5, dash='6,4', pid='pem-comparison', cid='pem-status')
    builder.text(936.0, 1116.0, 24,
                 '同一协议：相同 support / query 身份与 K 设置，主口径 Q95',
                 pid='pem-comparison')
    builder.text(936.0, 1148.0, 24,
                 '目标评估尚未运行；本图仅表示协议设计，不含实测结果',
                 pid='pem-comparison', weight=700)

    # ------------------------------------------------------------- finalise
    root.set('viewBox', f'0 0 {CANVAS_W:g} {CANVAS_H:g}')
    builder.finish(root)

    chars = ''.join(content for content, _, _ in builder.texts)
    font_dir = Path(__file__).resolve().parent / 'fonts'
    font_dir.mkdir(parents=True, exist_ok=True)
    fonts = ensure_fonts(font_dir, chars)
    rules = []
    for weight, face, style in (('Regular', '400', 'normal'), ('Bold', '700', 'normal')):
        uri = base64.b64encode(fonts[weight]).decode()
        rules.append(f'@font-face{{font-family:"{FONT}";font-weight:{face};'
                     f'font-style:{style};src:url(data:font/otf;base64,{uri}) format("opentype");}}')
    style = ET.Element(tag('style'), {'id': 'vivid-figure-fonts'})
    style.text = '\n'.join(rules)
    root.append(style)

    profile = ET.Element(tag('metadata'), {'id': 'vivid-font-profile'})
    profile.text = json.dumps({
        'profile': 'project-cjk',
        'svg_embeds_font_files': True,
        'roles': [
            {'role': 'cjk-sans-regular', 'family': FONT, 'postscript_name': 'SourceHanSansCN-Regular'},
            {'role': 'cjk-sans-bold', 'family': FONT, 'postscript_name': 'SourceHanSansCN-Bold'},
        ],
    }, ensure_ascii=False)
    root.append(profile)

    context['parts'] = builder.parts
    return root


if __name__ == '__main__':
    run(__file__, customize)
