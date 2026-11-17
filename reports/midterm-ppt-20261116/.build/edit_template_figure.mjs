import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { FileBlob, PresentationFile } from "/Users/williamlin/.cache/codex-runtimes/codex-runtime-install-3jSuSH/payload/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "/Users/williamlin/Downloads/13/spatial-meta-ce";
const INPUT = "/Users/williamlin/.t3/userdata/attachments/e08fd24e-3c69-4792-8d13-2fe0ca72b809-9168a3a6-be08-45fc-bccd-eaff5326c9bd-pptx.pptx";
const SKILL_DIR = "/Users/williamlin/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations";
const BUILD = path.join(ROOT, "reports/midterm-ppt-20261116/.build/template-figure-edit");
const OUTPUT = path.join(ROOT, "reports/midterm-ppt-20261116/output/面向数据漂移的自适应基数估计-中期汇报-模板版修订.pptx");
const RUNTIME_PYTHON = "/Users/williamlin/.cache/codex-runtimes/codex-runtime-install-3jSuSH/payload/codex-primary-runtime/dependencies/python/bin/python3";

const colors = {
  navy: "#253558",
  ink: "#374151",
  blue: "#3870B1",
  teal: "#257B67",
  amber: "#B45309",
  pale: "#F6F9FC",
  border: "#CBD8E6",
  white: "#FFFFFF",
};

const sourceBytes = await fs.readFile(INPUT);
const sourceSha256 = createHash("sha256").update(sourceBytes).digest("hex");
const presentation = await PresentationFile.importPptx(await FileBlob.load(INPUT));
if (presentation.slides.items.length !== 12) throw new Error("Expected the supplied 12-slide template deck");

const imageInspect = await presentation.inspect({ kind: "image", search: "Picture 31", maxChars: 3000 });
const picture = imageInspect.ndjson
  .split("\n")
  .filter(Boolean)
  .map(line => JSON.parse(line))
  .find(record => record.kind === "image" && record.name === "Picture 31" && record.slideIndex === 2);
if (!picture) throw new Error("Could not identify the slide 3 drift illustration by name and page");

const frame = picture.position;
const slide = presentation.slides.items[2];
const beforeDir = path.join(BUILD, "before");
const afterDir = path.join(BUILD, "after");
const finalDir = path.join(BUILD, "final-import");
await Promise.all([beforeDir, afterDir, finalDir, path.dirname(OUTPUT)].map(dir => fs.mkdir(dir, { recursive: true })));

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function renderDeck(deck, dir) {
  for (const [index, deckSlide] of deck.slides.items.entries()) {
    const file = path.join(dir, `slide-${String(index + 1).padStart(2, "0")}.png`);
    await writeBlob(file, await deck.export({ slide: deckSlide, format: "png", scale: 1 }));
  }
}

await renderDeck(presentation, beforeDir);
presentation.delete(picture.id);

function rect(name, x, y, width, height, fill, lineFill = "none", lineWidth = 0, geometry = "rect") {
  return slide.shapes.add({
    geometry,
    name,
    position: { left: x, top: y, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function text(name, value, x, y, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    typeface: "Microsoft YaHei",
    fontSize: options.size ?? 12.5,
    bold: options.bold ?? false,
    color: options.color ?? colors.ink,
    alignment: options.align ?? "left",
    verticalAlignment: options.valign ?? "middle",
    autoFit: "shrinkText",
    wrap: "square",
    insets: 0,
  };
  return shape;
}

function tableIcon(prefix, x, y, width, height, columns, accent) {
  rect(`${prefix}-table`, x, y, width, height, colors.white, colors.border, 1, "roundRect");
  rect(`${prefix}-table-header`, x + 1, y + 1, width - 2, 7, accent, "none", 0);
  rect(`${prefix}-table-row-rule`, x + 1, y + height / 2 + 3, width - 2, 1, colors.border);
  for (let index = 1; index < columns; index += 1) {
    const lineX = x + (width * index) / columns;
    rect(`${prefix}-table-column-${index}`, lineX, y + 8, 1, height - 9, colors.border);
  }
}

function driftPanel(prefix, x, accent) {
  rect(`${prefix}-panel`, x, frame.top, 414.72, frame.height, colors.pale, colors.border, 1, "roundRect");
  rect(`${prefix}-panel-accent`, x, frame.top, 414.72, 7, accent);
}

const leftX = 52.8;
const rightX = 492.48;
const top = frame.top;

driftPanel("spatial", leftX, colors.amber);
text("spatial-title", "空间漂移：跨数据库", leftX + 15, top + 9, 384, 20, { size: 14.5, bold: true, color: colors.navy });

tableIcon("source-db", leftX + 22, top + 38, 42, 28, 2, colors.amber);
text("source-db-label", "源库 A", leftX + 10, top + 68, 66, 17, { size: 12.5, bold: true, align: "center", color: colors.navy });
text("direct-transfer-arrow", "→", leftX + 75, top + 38, 48, 22, { size: 23, bold: true, align: "center", color: colors.amber });
text("direct-transfer-label", "模型迁移", leftX + 74, top + 59, 50, 15, { size: 11.5, align: "center", color: colors.amber });
tableIcon("target-db", leftX + 135, top + 38, 42, 28, 3, colors.teal);
text("target-db-label", "目标库 B", leftX + 122, top + 68, 68, 17, { size: 12.5, bold: true, align: "center", color: colors.navy });
text("spatial-differences", "表结构、属性分布与\n查询模式存在差异", leftX + 198, top + 35, 199, 35, { size: 12.5, color: colors.ink });
text("spatial-transfer-risk", "直接套用可能失准", leftX + 198, top + 69, 199, 17, { size: 12.5, bold: true, color: colors.amber });
text("spatial-current-focus", "当前重点：跨库少样本适应", leftX + 17, top + 98, 382, 16, { size: 12.5, bold: true, color: colors.teal });

driftPanel("temporal", rightX, colors.blue);
text("temporal-title", "时间漂移：同一数据库跨时间", rightX + 15, top + 9, 384, 20, { size: 14.5, bold: true, color: colors.navy });

tableIcon("time-t0", rightX + 20, top + 39, 38, 27, 2, colors.blue);
text("time-t0-label", "T0", rightX + 17, top + 68, 44, 16, { size: 12.5, bold: true, align: "center", color: colors.navy });
text("time-arrow-01", "→", rightX + 60, top + 40, 29, 18, { size: 18, bold: true, align: "center", color: colors.blue });
tableIcon("time-t1", rightX + 91, top + 39, 38, 27, 2, colors.blue);
text("time-t1-label", "T1", rightX + 88, top + 68, 44, 16, { size: 12.5, bold: true, align: "center", color: colors.navy });
text("time-arrow-12", "→", rightX + 131, top + 40, 29, 18, { size: 18, bold: true, align: "center", color: colors.blue });
tableIcon("time-t2", rightX + 162, top + 39, 38, 27, 2, colors.blue);
text("time-t2-label", "T2", rightX + 159, top + 68, 44, 16, { size: 12.5, bold: true, align: "center", color: colors.navy });
text("temporal-effect", "数据与负载变化\n静态模型逐渐过期", rightX + 216, top + 37, 183, 34, { size: 12.5, color: colors.ink });
text("temporal-status", "后续方案：反馈校准，尚无实现与精度结果", rightX + 17, top + 92, 382, 20, { size: 12, bold: true, color: colors.blue });

await renderDeck(presentation, afterDir);
const runTag = new Date().toISOString().replace(/[:.]/g, "-");
const candidatePath = path.join(BUILD, `candidate-${runTag}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const validation = await finalizePresentation({
  explicitTotalSlideCount: 12,
  requiredNativeTableOwnerSlides: [5, 8, 11],
  workspaceDir: ROOT,
  candidatePath,
  finalPath: OUTPUT,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "9144000,6858000",
    "--require-native-table-slide", "5",
    "--require-native-table-slide", "8",
    "--require-native-table-slide", "11",
  ],
  verifyArtifactToolImport: true,
  receiptPath: path.join(BUILD, "finalization.json"),
});

const finalPresentation = await PresentationFile.importPptx(await FileBlob.load(OUTPUT));
if (finalPresentation.slides.items.length !== 12) throw new Error("Final deck slide count changed");
const finalSlide = finalPresentation.slides.items[2];
const finalImages = await finalPresentation.inspect({ kind: "image", maxChars: 12000 });
const finalImageRecords = finalImages.ndjson.split("\n").filter(Boolean).map(line => JSON.parse(line));
if (finalImageRecords.some(record => record.slideIndex === 2)) throw new Error("Slide 3 still contains a raster image");

const tableInspect = await finalPresentation.inspect({ kind: "table", maxChars: 6000 });
const tableOwnerSlides = [...new Set(tableInspect.ndjson.split("\n").filter(Boolean)
  .map(line => JSON.parse(line).slideIndex + 1))].sort((a, b) => a - b);
if (JSON.stringify(tableOwnerSlides) !== JSON.stringify([5, 8, 11])) {
  throw new Error(`Native table ownership changed: ${JSON.stringify(tableOwnerSlides)}`);
}

const requiredLabels = ["空间漂移：跨数据库", "当前重点：跨库少样本适应", "时间漂移：同一数据库跨时间", "后续方案：反馈校准，尚无实现与精度结果"];
for (const label of requiredLabels) {
  const found = await finalPresentation.inspect({ kind: "textbox,shape", search: label, maxChars: 3000 });
  if (!found.ndjson.includes(label)) throw new Error(`Missing editable label in final deck: ${label}`);
}

if (finalSlide.images.items.length !== 0 || finalSlide.shapes.items.length < 60) {
  throw new Error(`Unexpected slide 3 object counts: ${finalSlide.images.items.length} images, ${finalSlide.shapes.items.length} shapes`);
}
await renderDeck(finalPresentation, finalDir);

const verification = {
  source: INPUT,
  sourceSha256,
  output: OUTPUT,
  outputSha256: createHash("sha256").update(await fs.readFile(OUTPUT)).digest("hex"),
  slideCount: finalPresentation.slides.items.length,
  slide3RasterImages: finalSlide.images.items.length,
  slide3NativeShapes: finalSlide.shapes.items.length,
  nativeTableOwnerSlides: tableOwnerSlides,
  requiredLabels,
  renderedSlideCount: finalPresentation.slides.items.length,
  finalization: validation,
};
await fs.writeFile(path.join(BUILD, "verification.json"), `${JSON.stringify(verification, null, 2)}\n`);
console.log(JSON.stringify(verification, null, 2));
