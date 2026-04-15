/**
 * SeeThrough PSD Generator
 * Uses ag-psd to generate PSD files in the browser from decomposed layers.
 */

// import { app } from "../../scripts/app.js";

// const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

let agPsdLoaded = false;
let agPsdLoadPromise = null;

const _ownScriptDir = new URL('./', import.meta.url).href;

async function ensureAgPsdLoaded() {
    if (agPsdLoaded) return;
    if (agPsdLoadPromise) return agPsdLoadPromise;

    agPsdLoadPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = _ownScriptDir + "ag-psd.bundle.js";
        script.onload = () => {
            agPsdLoaded = true;
            console.log("[SeeThrough] ag-psd bundle loaded");
            resolve();
        };
        script.onerror = (e) => {
            console.error("[SeeThrough] Failed to load ag-psd bundle:", e);
            reject(new Error("Failed to load ag-psd bundle"));
        };
        document.head.appendChild(script);
    });

    return agPsdLoadPromise;
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load: ${url}`));
        img.src = url;
    });
}

async function createPSD(layerInfo, psdType) {
    await ensureAgPsdLoaded();

    const {
        layers, width, height, prefix, timestamp,
        base, source_name, source_filename,
    } = layerInfo;
    const isDepth = psdType === "depth";
    const suffix = isDepth ? "_depth" : "";

    console.log(`[SeeThrough] Creating ${isDepth ? "depth " : ""}PSD: ${width}x${height}, ${layers.length} layers`);

    const compositeCanvas = document.createElement("canvas");
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext("2d");

    const partLayers = [];
    for (const layer of layers) {
        const filenameKey = isDepth ? "depth_filename" : "filename";
        const filename = layer[filenameKey];
        if (!filename) continue;

        const url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=output&t=${Date.now()}`);
        const img = await loadImage(url);

        const lw = layer.right - layer.left;
        const lh = layer.bottom - layer.top;

        const layerCanvas = document.createElement("canvas");
        layerCanvas.width = lw;
        layerCanvas.height = lh;
        const layerCtx = layerCanvas.getContext("2d");
        layerCtx.drawImage(img, 0, 0, lw, lh);

        compositeCtx.drawImage(img, layer.left, layer.top, lw, lh);

        partLayers.push({
            name: layer.name,
            canvas: layerCanvas,
            left: layer.left,
            top: layer.top,
            right: layer.right,
            bottom: layer.bottom,
            blendMode: "normal",
            opacity: 1,
        });
    }

    const children = [];
    children.push({
        name: "Parts",
        hidden: true,
        opened: false,
        children: partLayers,
    });

    if (source_filename) {
        const origUrl = api.apiURL(`/view?filename=${encodeURIComponent(source_filename)}&type=output&t=${Date.now()}`);
        try {
            const origImg = await loadImage(origUrl);
            const origCanvas = document.createElement("canvas");
            origCanvas.width = width;
            origCanvas.height = height;
            origCanvas.getContext("2d").drawImage(origImg, 0, 0, width, height);
            const prevComposite = compositeCtx.globalCompositeOperation;
            compositeCtx.globalCompositeOperation = "destination-over";
            compositeCtx.drawImage(origImg, 0, 0, width, height);
            compositeCtx.globalCompositeOperation = prevComposite;
            children.push({
                name: "Original",
                hidden: false,
                canvas: origCanvas,
                left: 0,
                top: 0,
                right: width,
                bottom: height,
                blendMode: "normal",
                opacity: 1,
            });
        } catch (e) {
            console.warn("[SeeThrough] Failed to load original base layer:", e);
        }
    }

    const psd = { width, height, canvas: compositeCanvas, children };
    const psdBuffer = window.AgPsd.writePsd(psd);
    const blob = new Blob([psdBuffer], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const downloadBase = base || `${prefix}_${timestamp}`;
    a.href = url;
    a.download = `${downloadBase}${suffix}.psd`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[SeeThrough] PSD downloaded: ${a.download}`);
}

async function createAllRunsPSD(layerInfo) {
    await ensureAgPsdLoaded();

    const { all_runs, width, height, prefix, timestamp } = layerInfo;
    if (!all_runs || all_runs.length === 0) {
        alert("No multi-run data available. Enable auto_fill and run with multiple inference passes.");
        return;
    }

    console.log(`[SeeThrough] Creating All Runs PSD: ${width}x${height}, ${all_runs.length} runs`);

    // Create composite canvas
    const compositeCanvas = document.createElement("canvas");
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext("2d");

    const runGroups = [];

    for (const runData of all_runs) {
        const groupLayers = [];

        for (const layer of runData.layers) {
            const filename = layer.filename;
            if (!filename) continue;

            const url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=output&t=${Date.now()}`);
            const img = await loadImage(url);

            const lw = layer.right - layer.left;
            const lh = layer.bottom - layer.top;

            const layerCanvas = document.createElement("canvas");
            layerCanvas.width = lw;
            layerCanvas.height = lh;
            const layerCtx = layerCanvas.getContext("2d");
            layerCtx.drawImage(img, 0, 0, lw, lh);

            groupLayers.push({
                name: layer.name,
                canvas: layerCanvas,
                left: layer.left,
                top: layer.top,
                right: layer.right,
                bottom: layer.bottom,
                blendMode: "normal",
                opacity: 1,
            });
        }

        runGroups.push({
            name: `Run ${runData.run} (seed=${runData.seed}, ${runData.layer_count} layers)`,
            children: groupLayers,
            opened: false,
            hidden: true,
        });
    }

    const children = [{
        name: "Runs",
        hidden: true,
        opened: false,
        children: runGroups,
    }];

    if (layerInfo.source_filename) {
        const origUrl = api.apiURL(`/view?filename=${encodeURIComponent(layerInfo.source_filename)}&type=output&t=${Date.now()}`);
        try {
            const origImg = await loadImage(origUrl);
            const origCanvas = document.createElement("canvas");
            origCanvas.width = width;
            origCanvas.height = height;
            origCanvas.getContext("2d").drawImage(origImg, 0, 0, width, height);
            const prev = compositeCtx.globalCompositeOperation;
            compositeCtx.globalCompositeOperation = "destination-over";
            compositeCtx.drawImage(origImg, 0, 0, width, height);
            compositeCtx.globalCompositeOperation = prev;
            children.push({
                name: "Original",
                hidden: false,
                canvas: origCanvas,
                left: 0, top: 0, right: width, bottom: height,
                blendMode: "normal",
                opacity: 1,
            });
        } catch (e) {
            console.warn("[SeeThrough] Failed to load original base layer:", e);
        }
    }

    const psd = { width, height, canvas: compositeCanvas, children };
    const psdBuffer = window.AgPsd.writePsd(psd);
    const blob = new Blob([psdBuffer], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const downloadBase = layerInfo.base || `${prefix}_${timestamp}`;
    a.href = url;
    a.download = `${downloadBase}_all_runs.psd`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[SeeThrough] All Runs PSD downloaded: ${a.download}`);
}

app.registerExtension({
    name: "ComfyUI-See-through.SavePSD",

    async nodeCreated(node) {
        if (node.comfyClass !== "SeeThrough_SavePSD") return;

        console.log("[SeeThrough] Setting up frontend PSD generator");

        // Download RGBA PSD button
        const dlBtn = node.addWidget(
            "button",
            "Download PSD",
            "Download PSD",
            async () => {
                try {
                    dlBtn.name = "Generating PSD...";
                    const logResp = await fetch(
                        api.apiURL("/view?filename=seethrough_psd_info.log&type=output&t=" + Date.now())
                    );
                    if (!logResp.ok) {
                        alert("Please run the workflow first to generate layers.");
                        return;
                    }
                    const infoFilename = (await logResp.text()).trim();
                    const infoResp = await fetch(
                        api.apiURL(`/view?filename=${encodeURIComponent(infoFilename)}&type=output&t=${Date.now()}`)
                    );
                    if (!infoResp.ok) {
                        alert("Failed to load layer information.");
                        return;
                    }
                    const layerInfo = await infoResp.json();
                    await createPSD(layerInfo, "rgba");
                } catch (error) {
                    console.error("[SeeThrough] Error:", error);
                    alert(`Failed to generate PSD: ${error.message}`);
                } finally {
                    dlBtn.name = "Download PSD";
                }
            }
        );
        dlBtn.color = "#10B981";
        dlBtn.bgcolor = "#059669";

        // Download Depth PSD button
        const dlDepthBtn = node.addWidget(
            "button",
            "Download Depth PSD",
            "Download Depth PSD",
            async () => {
                try {
                    dlDepthBtn.name = "Generating Depth PSD...";
                    const logResp = await fetch(
                        api.apiURL("/view?filename=seethrough_psd_info.log&type=output&t=" + Date.now())
                    );
                    if (!logResp.ok) {
                        alert("Please run the workflow first to generate layers.");
                        return;
                    }
                    const infoFilename = (await logResp.text()).trim();
                    const infoResp = await fetch(
                        api.apiURL(`/view?filename=${encodeURIComponent(infoFilename)}&type=output&t=${Date.now()}`)
                    );
                    if (!infoResp.ok) {
                        alert("Failed to load layer information.");
                        return;
                    }
                    const layerInfo = await infoResp.json();
                    await createPSD(layerInfo, "depth");
                } catch (error) {
                    console.error("[SeeThrough] Error:", error);
                    alert(`Failed to generate depth PSD: ${error.message}`);
                } finally {
                    dlDepthBtn.name = "Download Depth PSD";
                }
            }
        );
        dlDepthBtn.color = "#6366F1";
        dlDepthBtn.bgcolor = "#4F46E5";

        // Download All Runs PSD button (only useful with auto_fill)
        const dlAllRunsBtn = node.addWidget(
            "button",
            "Download All Runs PSD",
            "Download All Runs PSD",
            async () => {
                try {
                    dlAllRunsBtn.name = "Generating All Runs PSD...";
                    const logResp = await fetch(
                        api.apiURL("/view?filename=seethrough_psd_info.log&type=output&t=" + Date.now())
                    );
                    if (!logResp.ok) {
                        alert("Please run the workflow first to generate layers.");
                        return;
                    }
                    const infoFilename = (await logResp.text()).trim();
                    const infoResp = await fetch(
                        api.apiURL(`/view?filename=${encodeURIComponent(infoFilename)}&type=output&t=${Date.now()}`)
                    );
                    if (!infoResp.ok) {
                        alert("Failed to load layer information.");
                        return;
                    }
                    const layerInfo = await infoResp.json();
                    if (!layerInfo.all_runs || layerInfo.all_runs.length === 0) {
                        alert("No multi-run data. Enable auto_fill in GenerateLayers_Custom to use this feature.");
                        return;
                    }
                    await createAllRunsPSD(layerInfo);
                } catch (error) {
                    console.error("[SeeThrough] Error:", error);
                    alert(`Failed to generate All Runs PSD: ${error.message}`);
                } finally {
                    dlAllRunsBtn.name = "Download All Runs PSD";
                }
            }
        );
        dlAllRunsBtn.color = "#F59E0B";
        dlAllRunsBtn.bgcolor = "#D97706";
    },
});

console.log("[SeeThrough] PSD extension loaded");
