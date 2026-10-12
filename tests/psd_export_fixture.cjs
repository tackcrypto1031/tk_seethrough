// Run the real browser export functions and bundled PSD writer with tiny canvases.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { pathToFileURL } = require("url");
const outDir = process.argv[2];
class Canvas {
  constructor() { this.width = 2; this.height = 2; this.data = new Uint8ClampedArray(16); }
  getContext() {
    const canvas = this;
    return {
      globalCompositeOperation: "source-over",
      drawImage(image) { canvas.data = new Uint8ClampedArray(image.data); },
      clearRect() { canvas.data.fill(0); },
      getImageData() { return { width: canvas.width, height: canvas.height, data: canvas.data }; },
    };
  }
}
global.window = { comfyAPI: {
  app: { app: { registerExtension() {} } },
  api: { api: { apiURL: (url) => url } },
}};
global.document = {
  createElement(kind) { return kind === "canvas" ? new Canvas() : { click() {} }; },
  head: { appendChild(script) { script.onload(); } },
  body: { appendChild() {}, removeChild() {} },
};
global.Image = class {
  set src(url) {
    const filename = new URL(url, "http://test.invalid").searchParams.get("filename");
    const rgba = filename.includes("original") ? [0, 255, 0, 255] :
      filename.includes("back") ? [255, 0, 0, 255] : [0, 0, 255, 255];
    this.data = new Uint8ClampedArray([...rgba, ...rgba, ...rgba, ...rgba]);
    this.onload();
  }
};
global.alert = (message) => { throw new Error(message); };
URL.createObjectURL = () => "blob:test";
URL.revokeObjectURL = () => {};
const web = path.join(__dirname, "../web");
vm.runInThisContext(fs.readFileSync(path.join(web, "ag-psd.bundle.js"), "utf8"));
const realWriter = window.AgPsd.writePsd;
let outputName = "normal";
function convert(layer) {
  const result = { ...layer };
  if (layer.canvas) {
    result.imageData = layer.canvas.getContext("2d").getImageData();
    delete result.canvas;
  }
  if (layer.children) result.children = layer.children.map(convert);
  return result;
}
window.AgPsd.writePsd = (psd) => {
  const bytes = realWriter(convert(psd));
  fs.writeFileSync(path.join(outDir, outputName + ".psd"), Buffer.from(bytes));
  return bytes;
};
vm.runInThisContext(fs.readFileSync(path.join(web, "seethrough_psd.js"), "utf8")
  .replaceAll("import.meta.url", JSON.stringify(pathToFileURL(path.join(web, "seethrough_psd.js")).href)));
const layers = ["back", "front"].map(name => ({
  name, filename: name + ".png", depth_filename: name + "_depth.png",
  left: 0, top: 0, right: 2, bottom: 2,
}));
const info = { layers, width: 2, height: 2, prefix: "test", timestamp: "test", source_filename: "original.png",
  all_runs: [{ run: 1, seed: 42, layer_count: 2, layers }] };
(async () => {
  await createPSD(info, "normal");
  outputName = "depth";
  await createPSD(info, "depth");
  outputName = "runs";
  await createAllRunsPSD(info);
})().catch(error => { console.error(error); process.exitCode = 1; });

