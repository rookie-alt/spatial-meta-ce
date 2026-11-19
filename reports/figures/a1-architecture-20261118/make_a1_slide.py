#!/usr/bin/env python3
"""Build a 16:9 presentation slide for the A1 architecture."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches


OUT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H = 1920, 1080


def icon_bars() -> str:
    return (
        '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" '
        'stroke="currentColor" stroke-width="2.6" stroke-linecap="round">'
        '<path d="M7 25V16"/><path d="M15 25V10"/><path d="M23 25V6"/></svg>'
    )


def icon_fanout() -> str:
    return (
        '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" '
        'stroke="currentColor" stroke-width="2.2" stroke-linecap="round">'
        '<circle cx="5" cy="15" r="3"/><path d="M8 15h7v-8h6M15 15h6M15 15v8h6"/>'
        '<circle cx="24" cy="7" r="2.2"/><circle cx="24" cy="15" r="2.2"/>'
        '<circle cx="24" cy="23" r="2.2"/></svg>'
    )


def icon_table() -> str:
    return (
        '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" '
        'stroke="currentColor" stroke-width="2.2">'
        '<rect x="4" y="5" width="22" height="20" rx="3"/>'
        '<path d="M4 12h22M4 19h22M11 5v20M19 5v20"/></svg>'
    )


def icon_filter() -> str:
    return (
        '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" '
        'stroke="currentColor" stroke-width="2.2" stroke-linejoin="round">'
        '<path d="M4 5h22l-8 11v9l-6-3v-6z"/></svg>'
    )


def icon_database() -> str:
    return (
        '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" '
        'stroke="currentColor" stroke-width="2.1">'
        '<ellipse cx="15" cy="7" rx="9" ry="4"/>'
        '<path d="M6 7v14c0 2.2 4 4 9 4s9-1.8 9-4V7"/>'
        '<path d="M6 14c0 2.2 4 4 9 4s9-1.8 9-4"/></svg>'
    )


CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  width: 3840px; height: 2160px; overflow: hidden; background: #f6f8fc;
}
.stage {
  position: absolute; left: 0; top: 0; width: 1920px; height: 1080px;
  transform: scale(2); transform-origin: top left; overflow: hidden;
  background: linear-gradient(135deg, #f8f9fd 0%, #f5f7fc 55%, #f1f4fa 100%);
  color: #1f2a44;
  font-family: "PingFang SC", "Source Han Sans SC", "Noto Sans CJK SC", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.blob { position: absolute; border-radius: 50%; }
.blob.a { left: -90px; top: -90px; width: 600px; height: 320px; background: radial-gradient(circle, #ece9ff 0%, rgba(236,233,255,0) 70%); }
.blob.b { right: -140px; bottom: -120px; width: 700px; height: 360px; background: radial-gradient(circle, #e5f8ef 0%, rgba(229,248,239,0) 70%); }
.blob.c { right: -80px; top: -80px; width: 480px; height: 260px; background: radial-gradient(circle, #fff0e5 0%, rgba(255,240,229,0) 70%); }

.badge {
  position: absolute; left: 64px; top: 44px; height: 32px; padding: 0 18px;
  border-radius: 16px; background: #eeebff; border: 1px solid #cfc8f0;
  color: #736bc4; font-size: 14px; font-weight: 700; letter-spacing: .5px;
  display: flex; align-items: center;
}
h1 { position: absolute; left: 64px; top: 64px; margin: 0; font-size: 64px; line-height: 1.12; font-weight: 700; letter-spacing: -1px; }
.subtitle { position: absolute; left: 64px; top: 148px; margin: 0; font-size: 28px; color: #66738a; }
.stat {
  position: absolute; left: 1530px; top: 58px; width: 320px; height: 56px;
  border-radius: 28px; background: #fff; border: 1px solid #d8e0ec;
  box-shadow: 0 6px 18px rgba(27,42,74,.08);
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; font-weight: 600;
}
.rule { position: absolute; left: 64px; top: 198px; width: 1792px; height: 1.5px; background: #d8e0ec; }

.panel { position: absolute; top: 220px; height: 680px; border-radius: 32px; border: 1.6px dashed; z-index: 0; }
.panel.lav { left: 64px; width: 720px; background: linear-gradient(180deg,#f8f6ff 0%,#f2f7ff 100%); border-color: #cfc8f0; }
.panel.mint { left: 820px; width: 500px; background: linear-gradient(180deg,#f3fcf8 0%,#f1faff 100%); border-color: #bfe4d3; }
.panel.peach { left: 1356px; width: 500px; background: linear-gradient(180deg,#fff9f3 0%,#fff4f0 100%); border-color: #f1d2bc; }

.panel-title {
  position: absolute; top: 244px; width: 300px; height: 52px; border-radius: 26px;
  color: #fff; font-size: 26px; font-weight: 600;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 5px 14px rgba(27,42,74,.12); z-index: 3;
}
.panel-title.lav { left: 92px; background: linear-gradient(90deg,#948ae0,#6e7ccb); }
.panel-title.mint { left: 848px; background: linear-gradient(90deg,#67c2a6,#3f9c86); }
.panel-title.peach { left: 1384px; background: linear-gradient(90deg,#eda477,#d98258); }

.card {
  position: absolute; background: #fff; border: 1.2px solid #d8e0ec;
  border-radius: 20px; box-shadow: 0 5px 14px rgba(27,42,74,.08); z-index: 2;
}
.card .accent { position: absolute; left: 0; width: 5px; border-radius: 3px; background: var(--accent); }
.card .icon {
  position: absolute; left: 26px; top: 22px; width: 42px; height: 42px;
  border-radius: 50%; background: var(--tint); color: var(--accent);
  display: flex; align-items: center; justify-content: center;
}
.card .title { position: absolute; font-weight: 600; white-space: nowrap; }
.card .sub { position: absolute; color: #66738a; white-space: nowrap; }
.card .cap {
  position: absolute; right: 14px; top: 28px; height: 30px; padding: 0 12px;
  border-radius: 15px; background: var(--tint); color: var(--accent);
  font-size: 16px; font-weight: 700; display: flex; align-items: center;
}

.card.input { width: 270px; height: 86px; }
.card.input .accent { top: 16px; height: 54px; }
.card.input .title { left: 82px; top: 14px; font-size: 26px; }
.card.input .sub { left: 82px; top: 48px; font-size: 20px; }

.card.encoder { width: 320px; height: 140px; }
.card.encoder .title { left: 26px; top: 28px; font-size: 32px; }
.card.encoder .sub { left: 26px; top: 70px; font-size: 22px; }
.card.encoder .tag {
  position: absolute; left: 26px; top: 102px; height: 28px; padding: 0 12px;
  border-radius: 14px; font-size: 18px; font-weight: 700;
  display: flex; align-items: center;
}

.card.feature { width: 190px; height: 80px; }
.card.feature .accent { top: 14px; height: 52px; width: 4px; }
.card.feature .title { left: 22px; top: 14px; font-size: 24px; }
.card.feature .sub { left: 22px; top: 48px; font-size: 18px; }

.card.concat { width: 200px; height: 120px; background: linear-gradient(135deg,#fff,#eef5ff); border-color: #c9daf2; }
.card.concat .title { left: 22px; top: 18px; font-size: 26px; }
.card.concat .sub { left: 22px; top: 50px; font-size: 20px; }
.card.concat .pill {
  position: absolute; left: 22px; bottom: 12px; height: 28px; padding: 0 12px;
  border-radius: 14px; background: #4f78b8; color: #fff; font-size: 16px;
  font-weight: 700; display: flex; align-items: center;
}

.card.note { width: 440px; height: 80px; background: #fff8e4; border: 1.5px dashed #ead7a0; }
.card.note .title { left: 24px; top: 12px; font-size: 24px; color: #b8892f; }
.card.note .sub { left: 24px; top: 44px; font-size: 18px; }

.card.callout { width: 450px; height: 100px; background: #fff1e6; border-color: #f1d2bc; box-shadow: none; }
.card.callout .title { left: 30px; top: 20px; font-size: 28px; color: #d98258; }
.card.callout .sub { left: 30px; top: 56px; font-size: 20px; }

.card.mlp { width: 450px; height: 180px; background: linear-gradient(135deg,#fff,#eef5ff); border-color: #c9daf2; }
.card.mlp .title { left: 30px; top: 22px; font-size: 34px; }
.layers { position: absolute; left: 30px; top: 78px; display: flex; align-items: center; gap: 12px; }
.layer {
  height: 48px; border-radius: 12px; background: #eaf3ff; border: 1px solid #c9daf2;
  color: #4f78b8; font-family: "SF Mono", Menlo, monospace; font-size: 24px;
  font-weight: 700; display: flex; align-items: center; justify-content: center;
}
.layer-arrow { color: #4f78b8; font-size: 24px; }
.mlp-note { position: absolute; left: 30px; top: 142px; font-size: 20px; color: #66738a; }

.card.small { width: 210px; height: 80px; }
.card.small .title { left: 26px; top: 14px; font-size: 28px; }
.card.small .sub { left: 26px; top: 48px; font-size: 18px; }
.card.pg { background: #fff8e4; border-color: #ead7a0; }
.card.pg .icon { left: 20px; top: 26px; width: 30px; height: 30px; background: transparent; color: #b8892f; }
.card.pg .title { left: 56px; top: 14px; }
.card.pg .sub { left: 56px; top: 48px; color: #b8892f; font-family: "SF Mono", Menlo, monospace; font-size: 16px; }

.card.output { width: 450px; height: 100px; background: linear-gradient(135deg,#f3fcf7,#dff3e9); border-color: #9fd6bc; }
.card.output .formula {
  position: absolute; left: 30px; top: 22px; font-family: "SF Mono", Menlo, monospace;
  font-size: 34px; font-weight: 600;
}
.card.output .sub { position: absolute; left: 30px; top: 64px; font-size: 20px; color: #66738a; }

.label { position: absolute; color: #66738a; font-size: 18px; font-weight: 500; z-index: 4; white-space: nowrap; }
.label.center { text-align: center; }

.meta-band {
  position: absolute; left: 64px; top: 920px; width: 1792px; height: 120px;
  background: #fff; border: 1.2px solid #d8e0ec; border-radius: 24px;
  box-shadow: 0 6px 18px rgba(27,42,74,.08); z-index: 3;
}
.meta-title {
  position: absolute; left: 20px; top: 24px; width: 180px; height: 48px;
  border-radius: 24px; background: linear-gradient(90deg,#5d8fd6,#4f78b8);
  color: #fff; font-size: 24px; font-weight: 600;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 5px 14px rgba(27,42,74,.12);
}
.meta {
  position: absolute; top: 16px; width: 300px; height: 88px;
  background: #fff; border: 1.2px solid #d8e0ec; border-radius: 18px;
  box-shadow: 0 4px 12px rgba(27,42,74,.06); padding: 10px 18px;
  display: flex; flex-direction: column; justify-content: center;
}
.meta .step { font-size: 12px; color: #4f78b8; font-weight: 700; letter-spacing: 1px; margin-bottom: 2px; }
.meta .title { font-size: 22px; font-weight: 600; white-space: nowrap; }
.meta .sub { font-size: 16px; color: #66738a; margin-top: 4px; white-space: nowrap; }
.meta-arrow { position: absolute; top: 54px; width: 30px; text-align: center; color: #6b7890; font-size: 24px; }
.meta-loop {
  position: absolute; left: 1700px; top: 28px; width: 140px; height: 32px;
  border-radius: 16px; background: #f3f0ff; border: 1px solid #cfc8f0;
  color: #736bc4; font-size: 15px; font-weight: 600;
  display: flex; align-items: center; justify-content: center;
}

.connectors { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; z-index: 1; pointer-events: none; }
.connectors > path { fill: none; stroke: #6b7890; stroke-width: 2.6; stroke-linecap: round; stroke-linejoin: round; }
.connectors marker path { fill: #6b7890; stroke: none; }
.connectors .arrow { marker-end: url(#arrow); }
"""


def input_card(
    title: str, sub: str, cap: str, accent: str, tint: str, icon: str, top: int
) -> str:
    return f"""
    <div class="card input" style="left:92px;top:{top}px;--accent:{accent};--tint:{tint};">
      <div class="accent"></div><div class="icon">{icon}</div>
      <div class="title">{title}</div><div class="sub">{sub}</div>
      <div class="cap">{cap}</div>
    </div>"""


def feature_card(
    title: str, sub: str, accent: str, tint: str, top: int, size: int = 24
) -> str:
    return f"""
    <div class="card feature" style="left:850px;top:{top}px;--accent:{accent};--tint:{tint};">
      <div class="accent"></div>
      <div class="title" style="font-size:{size}px;">{title}</div>
      <div class="sub">{sub}</div>
    </div>"""


html = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div class="stage">
  <div class="blob a"></div><div class="blob b"></div><div class="blob c"></div>

  <div class="badge">A1 · 122</div>
  <h1>A1 模型架构</h1>
  <p class="subtitle">两阶段编码 + PG 残差锚 + 元学习适应</p>
  <div class="stat">293,905 参数</div>
  <div class="rule"></div>

  <div class="panel lav"></div>
  <div class="panel mint"></div>
  <div class="panel peach"></div>

  <div class="panel-title lav">① 查询编码</div>
  <div class="panel-title mint">② 表示融合</div>
  <div class="panel-title peach">③ 残差预测</div>

  {input_card("Join", "连接直方图", "12 × 40", "#736bc4", "#eeebff", icon_bars(), 334)}
  {input_card("Fanout", "扇出", "12 × 40", "#4f78b8", "#eaf3ff", icon_fanout(), 444)}
  {input_card("Table", "表规模", "7 × 4", "#3f9c86", "#e8f8f1", icon_table(), 554)}
  {input_card("Filter", "过滤条件", "10 × 43", "#d98258", "#fff1e6", icon_filter(), 664)}

  <div class="card encoder" style="left:430px;top:362px;background:linear-gradient(135deg,#fff,#f1eeff);border-color:#cfc8f0;">
    <div class="title">Scale Encoder</div>
    <div class="sub">2 blocks · 4 heads</div>
    <div class="tag" style="background:#eeebff;color:#736bc4;">d=64</div>
  </div>
  <div class="card encoder" style="left:430px;top:582px;background:linear-gradient(135deg,#fff,#eaf9f2);border-color:#bfe4d3;">
    <div class="title">Filter Encoder</div>
    <div class="sub">2 blocks · 4 heads</div>
    <div class="tag" style="background:#e8f8f1;color:#3f9c86;">d=64</div>
  </div>

  <div class="label" style="left:612px;top:532px;">scale tokens</div>
  <div class="label" style="left:92px;top:856px;font-size:18px;">容量 [12, 12, 7, 10] · 超限 unsupported</div>

  {feature_card("log_cartesian", "1 维 · 表规模", "#3f9c86", "#e8f8f1", 350, 22)}
  {feature_card("length 16", "counts → len_net", "#4f78b8", "#eaf3ff", 470, 24)}
  {feature_card("CLS 64", "编码器输出", "#736bc4", "#eeebff", 590, 24)}

  <div class="card concat" style="left:1090px;top:490px;">
    <div class="title">Concat 81</div>
    <div class="sub">64 + 16 + 1</div>
    <div class="pill">81 维</div>
  </div>

  <div class="card note" style="left:850px;top:730px;">
    <div class="title">A1-PG：额外输入 pg_log</div>
    <div class="sub">82 维 · 默认关闭</div>
  </div>
  <div class="label" style="left:850px;top:842px;font-size:20px;">A1：PG 只做输出锚</div>
  <div class="label" style="left:850px;top:868px;font-size:18px;">目标库不参与训练/选择</div>

  <div class="card callout" style="left:1380px;top:316px;">
    <div class="title">PG 只做锚点</div>
    <div class="sub">网络只学残差</div>
  </div>

  <div class="card mlp" style="left:1380px;top:460px;">
    <div class="title">MLP</div>
    <div class="layers">
      <div class="layer" style="width:70px;">81</div><div class="layer-arrow">→</div>
      <div class="layer" style="width:80px;">128</div><div class="layer-arrow">→</div>
      <div class="layer" style="width:70px;">64</div><div class="layer-arrow">→</div>
      <div class="layer" style="width:50px;">1</div>
    </div>
    <div class="mlp-note">最后一层零初始化</div>
  </div>

  <div class="card small" style="left:1380px;top:690px;">
    <div class="title">残差 r</div><div class="sub">query 修正</div>
  </div>
  <div class="card small pg" style="left:1620px;top:690px;">
    <div class="icon">{icon_database()}</div>
    <div class="title">PG</div><div class="sub">log(max(PG,1))</div>
  </div>

  <div class="card output" style="left:1380px;top:790px;">
    <div class="formula">ŷ = log(PG) + r</div>
    <div class="sub">初始输出 = PG</div>
  </div>

  <div class="meta-band">
    <div class="meta-title">④ 元学习</div>
    <div class="meta" style="left:300px;">
      <div class="step">STEP 1</div>
      <div class="title">采样跨库 episode</div>
      <div class="sub">源库 support/query 不重叠</div>
    </div>
    <div class="meta-arrow" style="left:610px;">→</div>
    <div class="meta" style="left:660px;">
      <div class="step">STEP 2</div>
      <div class="title">Support 内循环</div>
      <div class="sub">5 步 · 只更新 LoRA Q/V</div>
    </div>
    <div class="meta-arrow" style="left:970px;">→</div>
    <div class="meta" style="left:1020px;">
      <div class="step">STEP 3 · ERM 对照</div>
      <div class="title">Query 外循环</div>
      <div class="sub">FOMAML 更新 backbone/head</div>
    </div>
    <div class="meta-arrow" style="left:1330px;">→</div>
    <div class="meta" style="left:1380px;">
      <div class="step">STEP 4</div>
      <div class="title">评估</div>
      <div class="sub">K=0 vs K=5/20</div>
    </div>
    <div class="meta-loop">↻ 下一 episode</div>
  </div>
  <svg class="connectors" viewBox="0 0 1920 1080">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M0 0L10 5L0 10Z" fill="#6b7890" stroke="none"/>
    </marker>
  </defs>
  <path class="arrow" d="M362 377L430 410"/>
  <path class="arrow" d="M362 487L430 455"/>
  <path class="arrow" d="M362 597L430 630"/>
  <path class="arrow" d="M362 707L430 675"/>
  <path class="arrow" d="M590 502V582"/>
  <path class="arrow" d="M750 652H850"/>
  <path class="arrow" d="M1040 390L1090 510"/>
  <path class="arrow" d="M1040 510L1090 550"/>
  <path class="arrow" d="M1040 630L1090 590"/>
  <path class="arrow" d="M1290 550H1380"/>
  <path class="arrow" d="M1485 640V690"/>
  <path class="arrow" d="M1485 770V790"/>
  <path class="arrow" d="M1725 770V790"/>
  </svg>
</div>
</body></html>
"""

html_path = OUT / "a1-slide.html"
png_path = OUT / "a1-slide.png"
preview_path = OUT / "a1-slide-preview.png"
pptx_path = OUT / "A1-architecture-slide.pptx"

html_path.write_text(html, encoding="utf-8")

subprocess.run(
    [
        CHROME,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=3840,2160",
        f"--screenshot={png_path}",
        f"file://{html_path}",
    ],
    check=True,
    capture_output=True,
)

image = Image.open(png_path).convert("RGB")
image.resize((W, H), Image.LANCZOS).save(preview_path)

presentation = Presentation()
presentation.slide_width = Inches(13.333)
presentation.slide_height = Inches(7.5)
slide = presentation.slides.add_slide(presentation.slide_layouts[6])
picture = slide.shapes.add_picture(
    str(png_path),
    0,
    0,
    width=presentation.slide_width,
    height=presentation.slide_height,
)
picture._element.nvPicPr.cNvPr.set("descr", "A1 模型架构：两阶段编码 + PG 残差锚")
slide.notes_slide.notes_text_frame.text = (
    "左：四组查询特征经 Scale/Filter 两阶段编码器得到 CLS。"
    "中：CLS 64 + length 16 + log_cartesian 1 拼接为 81 维。"
    "右：MLP 学习残差 r，PG 只作为输出锚点，y_hat = log(max(PG,1)) + r。"
    "A1-PG 是探索支路：额外把 pg_log 送入 head，默认关闭。"
    "元学习：采样 support/query episode；support 内循环 5 步只更新 LoRA Q/V；"
    "query 外循环用 FOMAML 更新 backbone/head；评估比较 K=0 与 K=5/20。"
    "训练为 4000 steps、3 seeds；目标库不参与训练或选择。"
)
presentation.core_properties.title = "A1 模型架构"
presentation.core_properties.author = "williamlin"
presentation.save(pptx_path)

print(html_path)
print(png_path)
print(preview_path)
print(pptx_path)
