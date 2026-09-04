import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    restoreExecutedWidgetValues,
    restoreGenerationPrompt,
} from "./prompt_optimizer_core.mjs";


const COMBINED_TARGET_NODE = "H3OptimizerRef2VAPromptPackageGeneratorCS";
const SPLIT_TARGET_NODE = "H3OptimizerRef2VAPromptPackageCS";
const TARGET_NODES = [COMBINED_TARGET_NODE, SPLIT_TARGET_NODE];
const RECORDER_NODE = "H3OptimizerVideoOutputCS";
const SEGMENT_SETTINGS_NODE = "H3OptimizerSegmentSettingsCS";
const SIDEBAR_ID = "h3-prompt-optimizer";
const UNTRACKED_STATE_HASH = "0".repeat(64);

let panel = null;
let currentGeneration = null;
let currentSnapshot = null;
let currentAnalysis = null;
let activeAnalysisRequestId = null;
let cancellingAnalysisRequestId = null;
let generationsById = new Map();
let generationLoadToken = 0;


function graphNodes(type) {
    return (app.graph?._nodes || []).filter((node) => node.type === type || node.comfyClass === type);
}


function optimizerNodes() {
    return TARGET_NODES.flatMap((type) => graphNodes(type));
}


function widget(node, name) {
    return node?.widgets?.find((item) => item.name === name);
}


function widgetValue(node, name) {
    return widget(node, name)?.value;
}


function inputSource(node, name) {
    const input = node?.inputs?.find((item) => item.name === name);
    const link = linkInfo(input?.link);
    if (!link) {
        return null;
    }
    const source = app.graph.getNodeById(link.origin_id);
    return {
        node: source,
        output: source?.outputs?.[link.origin_slot]?.name,
    };
}


function settingsOptimizerId(node) {
    const projectName = String(widgetValue(node, "project_name") || "");
    const segmentId = Number(widgetValue(node, "segment_id"));
    if (!projectName || !Number.isInteger(segmentId) || segmentId < 1) {
        return "";
    }
    return `${projectName}_segment_${String(segmentId).padStart(3, "0")}`;
}


function nodeOptimizerId(node) {
    const source = inputSource(node, "optimizer_id");
    if (source?.output === "optimizer_id" &&
        (source.node?.type === SEGMENT_SETTINGS_NODE ||
            source.node?.comfyClass === SEGMENT_SETTINGS_NODE)) {
        return settingsOptimizerId(source.node);
    }
    return String(widgetValue(node, "optimizer_id") || "").trim();
}


function optimizerIds() {
    return [...new Set(optimizerNodes()
        .map((node) => nodeOptimizerId(node))
        .filter(Boolean))].sort((left, right) => {
            const leftNumber = Number(left.match(/\d+(?!.*\d)/)?.[0] ?? -1);
            const rightNumber = Number(right.match(/\d+(?!.*\d)/)?.[0] ?? -1);
            return rightNumber - leftNumber || right.localeCompare(
                left, undefined, {numeric: true, sensitivity: "base"},
            );
        });
}


function targetNode(optimizerId) {
    const matches = optimizerNodes().filter(
        (node) => nodeOptimizerId(node) === optimizerId,
    );
    if (matches.length !== 1) {
        throw new Error(`optimizer_id '${optimizerId}' の Prompt Package ノードは1個にしてください。`);
    }
    return matches[0];
}


function linkInfo(linkId) {
    if (linkId == null) {
        return null;
    }
    const links = app.graph?.links;
    return links instanceof Map ? links.get(linkId) : links?.[linkId];
}


async function requestJson(path, options = {}) {
    const response = await api.fetchApi(path, options);
    let body = null;
    try {
        body = await response.json();
    } catch (_error) {
        body = {};
    }
    if (!response.ok) {
        throw new Error(body.error || `${response.status} ${response.statusText}`);
    }
    return body;
}


function setStatus(message, kind = "") {
    if (!panel) {
        return;
    }
    panel.status.textContent = message;
    panel.status.dataset.kind = kind;
}


async function refreshBackendStatus() {
    if (!panel) {
        return;
    }
    try {
        const status = await requestJson("/h3_optimizer/status");
        const backend = status.critic_backend === "llama_cli" ? "local llama-cli" : "OpenAI-compatible";
        panel.backend.textContent = `VLM: ${backend} / ${status.critic_model}`;
    } catch (error) {
        panel.backend.textContent = `VLM設定エラー: ${error.message}`;
    }
}


function setWidgetValue(node, name, value) {
    const item = widget(node, name);
    if (!item) {
        throw new Error(`${node.title || node.type} に ${name} widget がありません。`);
    }
    const previous = item.value;
    item.value = value;
    item.callback?.(value);
    node.onWidgetChanged?.(name, value, previous, item);
}


function snapshotSettings(snapshot) {
    const lines = [];
    const keys = new Set([
        "seed", "noise_seed", "steps", "cfg", "denoise", "sampler_name", "scheduler",
        "width", "height", "fps", "frame_rate", "num_frames", "duration_seconds",
    ]);
    for (const [nodeId, node] of Object.entries(snapshot?.api_prompt || {})) {
        const values = Object.entries(node.inputs || {}).filter(
            ([name, value]) => keys.has(name) && ["string", "number", "boolean"].includes(typeof value),
        );
        if (!values.length) {
            continue;
        }
        const title = node._meta?.title || node.class_type || `Node ${nodeId}`;
        lines.push(`${title} (#${nodeId}): ${values.map(([name, value]) => `${name}=${value}`).join(", ")}`);
    }
    return lines.length ? lines.join("\n") : "保存済みAPI prompt内に表示対象の生成条件がありません。";
}


async function restoreGeneration(generate) {
    try {
        if (!currentGeneration || !currentSnapshot?.workflow) {
            throw new Error("この世代には復元可能なWorkflowスナップショットがありません。");
        }
        const action = generate ? "復元して同じ条件で再生成" : "復元";
        if (!window.confirm(`${currentGeneration.created_at} の生成条件を${action}します。現在のWorkflowは置き換わります。`)) {
            return;
        }

        setStatus("生成条件を復元中…");
        await app.loadGraphData(structuredClone(currentSnapshot.workflow));
        app.graph.beforeChange?.();
        try {
            restoreExecutedWidgetValues(app.rootGraph, currentSnapshot.api_prompt, setWidgetValue);
            const target = targetNode(currentGeneration.optimizer_id);
            requireApprovedReferences(target);
            restoreGenerationPrompt(target, currentGeneration.prompt, setWidgetValue);
        } finally {
            app.graph.afterChange?.();
            app.graph.change?.();
            app.canvas?.setDirty?.(true, true);
        }
        if (generate) {
            setStatus("保存したPrompt・seed・生成条件でキューに追加します。", "success");
            await app.queuePrompt(0);
        } else {
            setStatus("生成条件を復元しました。解像度やstepsを変更してからQueueできます。", "success");
        }
    } catch (error) {
        setStatus(error.message, "error");
    }
}


function requireApprovedReferences(node) {
    const combined = node.type === COMBINED_TARGET_NODE || node.comfyClass === COMBINED_TARGET_NODE;
    const signatureFields = ["ref_images", "ref_videos", "ref_video_audios", "ref_audios"];
    for (const signatureField of signatureFields) {
        const inputGroup = combined ? signatureField : `approved_${signatureField}`;
        const expected = currentGeneration?.reference_signature?.[signatureField]?.length || 0;
        const connected = (node.inputs || []).filter(
            (input) => input.name.startsWith(`${inputGroup}.`) && input.link != null,
        ).length;
        if (connected < expected) {
            throw new Error(
                `${inputGroup}へGeneratorと同じ参照を接続してください（必要${expected}、接続${connected}）。`,
            );
        }
    }
}


function setResumeInput(node, segmentId) {
    const input = node.inputs?.find((item) => item.name === "resume_from");
    const link = linkInfo(input?.link);
    if (!link) {
        setWidgetValue(node, "resume_from", segmentId);
        return;
    }
    const origin = app.graph.getNodeById(link.origin_id);
    const isSettings = origin?.type === SEGMENT_SETTINGS_NODE ||
        origin?.comfyClass === SEGMENT_SETTINGS_NODE;
    const valueWidget = isSettings
        ? widget(origin, "resume_from")
        : widget(origin, "value") || origin.widgets?.find(
            (item) => typeof item.value === "number",
        );
    if (!valueWidget) {
        throw new Error("接続された resume_from の数値widgetを特定できません。手動で更新してください。");
    }
    setWidgetValue(origin, valueWidget.name, segmentId);
}


function updateResumeFrom(optimizerId, segmentId) {
    const recorders = graphNodes(RECORDER_NODE).filter(
        (node) => nodeOptimizerId(node) === optimizerId,
    );
    if (recorders.length !== 1) {
        throw new Error(`optimizer_id '${optimizerId}' の Video Output ノードは1個にしてください。`);
    }
    const recorder = recorders[0];
    setResumeInput(recorder, segmentId);

    const videoInput = recorder.inputs?.find((item) => item.name === "video");
    const videoLink = linkInfo(videoInput?.link);
    const checkpoint = videoLink ? app.graph.getNodeById(videoLink.origin_id) : null;
    if (checkpoint && (checkpoint.type === "VideoSegmentCheckpointCS" ||
        checkpoint.comfyClass === "VideoSegmentCheckpointCS")) {
        setResumeInput(checkpoint, segmentId);
    }
}


function applyGraphChange(analysis, generate) {
    const optimizerId = analysis.optimizer_id;
    const target = targetNode(optimizerId);
    const action = analysis.proposal.action;
    if (!["PATCH_PROMPT", "REGENERATE_SAME_PROMPT"].includes(action)) {
        throw new Error(`${action} は自動Apply対象ではありません。提案内容を確認して手動対応してください。`);
    }
    requireApprovedReferences(target);

    app.graph.beforeChange?.();
    try {
        if (action === "PATCH_PROMPT") {
            setWidgetValue(target, "approved_prompt", analysis.optimized_prompt);
        } else {
            setWidgetValue(target, "approved_prompt", currentGeneration.prompt);
        }
        setWidgetValue(target, "use_approved_prompt", true);
        if (generate) {
            updateResumeFrom(optimizerId, Number(currentGeneration.segment_id));
        }
    } finally {
        app.graph.afterChange?.();
        app.graph.change?.();
        app.canvas?.setDirty?.(true, true);
    }
}


function requireAnalysis() {
    if (!currentAnalysis || !currentGeneration) {
        throw new Error("Applyする解析結果がありません。");
    }
}


async function applyAnalysis(generate) {
    try {
        setStatus("承認内容を反映中…");
        requireAnalysis();
        applyGraphChange(currentAnalysis, generate);
        panel.apply.disabled = true;
        panel.applyGenerate.disabled = true;
        if (generate) {
            setStatus("承認内容を反映し、再生成をキューに追加します。", "success");
            await app.queuePrompt(0);
        } else {
            await requestJson(`/h3_optimizer/analyses/${currentAnalysis.analysis_id}/applied`, {
                method: "POST",
            });
            setStatus("承認内容をワークフローへ反映しました。", "success");
        }
    } catch (error) {
        setStatus(error.message, "error");
    }
}


function renderAnalysis(analysis) {
    panel.analysis.replaceChildren();
    const critic = analysis.critic;
    const proposal = analysis.proposal;
    const heading = document.createElement("h3");
    heading.textContent = `${proposal.action} — ${critic.issue_type}`;
    const reason = document.createElement("p");
    reason.textContent = proposal.reason;
    const evidenceTitle = document.createElement("strong");
    evidenceTitle.textContent = "観察結果";
    const observations = document.createElement("ul");
    for (const observation of critic.observations || []) {
        const item = document.createElement("li");
        const range = observation.start_sec == null || observation.end_sec == null
            ? "時刻不明"
            : `${Number(observation.start_sec).toFixed(2)}–${Number(observation.end_sec).toFixed(2)}s`;
        item.textContent = `${range}: ${observation.text}`;
        observations.append(item);
    }
    const diffTitle = document.createElement("strong");
    diffTitle.textContent = "Prompt差分";
    const diff = document.createElement("pre");
    diff.textContent = analysis.diff || "（Prompt変更なし）";
    panel.analysis.append(heading, reason, evidenceTitle, observations, diffTitle, diff);

    const canApply = ["PATCH_PROMPT", "REGENERATE_SAME_PROMPT"].includes(proposal.action);
    panel.apply.disabled = !canApply;
    panel.applyGenerate.disabled = !canApply;
}


async function cancelAnalysis() {
    const requestId = activeAnalysisRequestId;
    if (!requestId || cancellingAnalysisRequestId === requestId) {
        return;
    }
    cancellingAnalysisRequestId = requestId;
    panel.analyze.disabled = true;
    panel.analyze.textContent = "中断中…";
    setStatus("解析を中断しています…");
    try {
        await requestJson(`/h3_optimizer/analyze/${encodeURIComponent(requestId)}/cancel`, {
            method: "POST",
        });
    } catch (error) {
        if (activeAnalysisRequestId === requestId) {
            setStatus(error.message, "error");
        }
    }
}


async function analyze() {
    if (activeAnalysisRequestId) {
        await cancelAnalysis();
        return;
    }
    let requestId = null;
    try {
        const feedback = panel.feedback.value.trim();
        if (!feedback) {
            throw new Error("改善したい点を入力してください。");
        }
        if (!currentGeneration) {
            throw new Error("レビュー対象動画がありません。");
        }
        const startText = panel.start.value.trim();
        const endText = panel.end.value.trim();
        let timeRange = null;
        if (startText || endText) {
            if (!startText || !endText) {
                throw new Error("時間範囲は開始と終了を両方入力してください。");
            }
            const start = Number(startText);
            const end = Number(endText);
            if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
                throw new Error("時間範囲が不正です。");
            }
            timeRange = {start_sec: start, end_sec: end};
        }

        currentAnalysis = null;
        panel.analysis.replaceChildren();
        panel.apply.disabled = true;
        panel.applyGenerate.disabled = true;
        setStatus("動画を解析中…");
        panel.analyze.disabled = true;
        requestId = `${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
        activeAnalysisRequestId = requestId;
        panel.analyze.textContent = "解析を中断";
        panel.analyze.disabled = false;
        currentAnalysis = await requestJson("/h3_optimizer/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-H3-Analysis-Request-ID": requestId,
            },
            body: JSON.stringify({
                generation_id: currentGeneration.generation_id,
                feedback,
                time_range: timeRange,
                target_state_hash: UNTRACKED_STATE_HASH,
            }),
        });
        renderAnalysis(currentAnalysis);
        setStatus("解析が完了しました。内容を確認してApplyしてください。", "success");
    } catch (error) {
        if (requestId && cancellingAnalysisRequestId === requestId) {
            setStatus("解析を中断しました。");
        } else {
            setStatus(error.message, "error");
        }
    } finally {
        if (panel && (!requestId || activeAnalysisRequestId === requestId)) {
            activeAnalysisRequestId = null;
            cancellingAnalysisRequestId = null;
            panel.analyze.textContent = "動画を解析";
            panel.analyze.disabled = false;
        }
    }
}


async function renderGeneration(generation) {
    const loadToken = ++generationLoadToken;
    currentGeneration = generation;
    currentSnapshot = null;
    currentAnalysis = null;
    panel.analysis.replaceChildren();
    panel.apply.disabled = true;
    panel.applyGenerate.disabled = true;
    panel.restore.disabled = true;
    panel.restoreGenerate.disabled = true;
    panel.meta.textContent = `${generation.project_name} / segment ${generation.segment_id} / ${generation.created_at}`;
    panel.prompt.textContent = generation.prompt;
    panel.settings.textContent = "生成条件を読込中…";
    panel.video.src = api.apiURL(generation.video.url);
    panel.video.load();
    try {
        const snapshot = await requestJson(
            `/h3_optimizer/generations/${encodeURIComponent(generation.generation_id)}/snapshot`,
        );
        if (loadToken !== generationLoadToken) {
            return;
        }
        currentSnapshot = snapshot;
        panel.settings.textContent = snapshotSettings(snapshot);
        panel.restore.disabled = !snapshot.workflow;
        panel.restoreGenerate.disabled = !snapshot.workflow;
    } catch (_error) {
        if (loadToken === generationLoadToken) {
            panel.settings.textContent = "この動画は生成条件の保存機能追加前に登録されています。";
        }
    }
}


async function selectGeneration() {
    const generation = generationsById.get(panel.generation.value);
    if (!generation) {
        return;
    }
    await renderGeneration(generation);
    setStatus("レビュー準備完了", "success");
}


async function refreshGeneration() {
    if (!panel) {
        return;
    }
    const ids = optimizerIds();
    const selected = ids.includes(panel.optimizer.value) ? panel.optimizer.value : ids[0];
    panel.optimizer.replaceChildren(...ids.map((id) => {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        option.selected = id === selected;
        return option;
    }));
    if (!selected) {
        currentGeneration = null;
        currentSnapshot = null;
        generationsById = new Map();
        panel.video.removeAttribute("src");
        panel.generation.replaceChildren();
        panel.restore.disabled = true;
        panel.restoreGenerate.disabled = true;
        panel.meta.textContent = "対象ノードをワークフローへ追加してください。";
        setStatus("H3 Optimizer Ref2VA Prompt Package ノードが見つかりません。", "error");
        return;
    }
    try {
        setStatus("最新動画を確認中…");
        const history = await requestJson(
            `/h3_optimizer/history?optimizer_id=${encodeURIComponent(selected)}`,
        );
        generationsById = new Map(history.generations.map(
            (generation) => [generation.generation_id, generation],
        ));
        panel.generation.replaceChildren(...history.generations.map((generation, index) => {
            const option = document.createElement("option");
            option.value = generation.generation_id;
            option.textContent = `${generation.created_at} / ${generation.frames}f @ ${generation.fps}fps`;
            option.selected = index === 0;
            return option;
        }));
        if (!history.generations.length) {
            throw new Error("No generation has been registered for this optimizer_id.");
        }
        await renderGeneration(history.generations[0]);
        setStatus("レビュー準備完了", "success");
    } catch (error) {
        currentGeneration = null;
        currentSnapshot = null;
        generationsById = new Map();
        panel.video.removeAttribute("src");
        panel.generation.replaceChildren();
        panel.restore.disabled = true;
        panel.restoreGenerate.disabled = true;
        panel.meta.textContent = "まだ動画が登録されていません。";
        setStatus(error.message, "error");
    }
}


function labeledInput(labelText, input) {
    const label = document.createElement("label");
    const title = document.createElement("span");
    title.textContent = labelText;
    label.append(title, input);
    return label;
}


function button(text, onClick) {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = text;
    element.addEventListener("click", onClick);
    return element;
}


function renderSidebar(root) {
    root.classList.add("h3-optimizer-panel");
    const backend = document.createElement("div");
    backend.className = "h3-optimizer-meta";
    backend.textContent = "VLM設定を確認中…";
    const optimizer = document.createElement("select");
    const refresh = button("最新動画を読込", refreshGeneration);
    const generation = document.createElement("select");
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    const meta = document.createElement("div");
    meta.className = "h3-optimizer-meta";
    const settings = document.createElement("pre");
    settings.className = "h3-optimizer-settings";
    const promptDetails = document.createElement("details");
    const promptSummary = document.createElement("summary");
    promptSummary.textContent = "保存したPrompt";
    const prompt = document.createElement("pre");
    promptDetails.append(promptSummary, prompt);
    const restore = button("生成条件を復元", () => restoreGeneration(false));
    const restoreGenerate = button("同じ条件で再生成", () => restoreGeneration(true));
    restore.disabled = true;
    restoreGenerate.disabled = true;
    const restoreActions = document.createElement("div");
    restoreActions.className = "h3-optimizer-actions";
    restoreActions.append(restore, restoreGenerate);
    const feedback = document.createElement("textarea");
    feedback.rows = 5;
    feedback.placeholder = "例: 2.0–3.5秒で右手が不自然に曲がる";
    const start = document.createElement("input");
    start.type = "number";
    start.min = "0";
    start.step = "0.01";
    start.placeholder = "任意";
    const end = start.cloneNode();
    const range = document.createElement("div");
    range.className = "h3-optimizer-range";
    range.append(labeledInput("開始秒", start), labeledInput("終了秒", end));
    const analyzeButton = button("動画を解析", analyze);
    const analysis = document.createElement("section");
    analysis.className = "h3-optimizer-analysis";
    const apply = button("Apply", () => applyAnalysis(false));
    const applyGenerate = button("Apply & Generate", () => applyAnalysis(true));
    apply.disabled = true;
    applyGenerate.disabled = true;
    const actions = document.createElement("div");
    actions.className = "h3-optimizer-actions";
    actions.append(apply, applyGenerate);
    const status = document.createElement("div");
    status.className = "h3-optimizer-status";

    root.replaceChildren(
        backend,
        labeledInput("Optimizer ID", optimizer),
        refresh,
        labeledInput("生成履歴", generation),
        video,
        meta,
        settings,
        promptDetails,
        restoreActions,
        labeledInput("改善したい点", feedback),
        range,
        analyzeButton,
        analysis,
        actions,
        status,
    );
    panel = {root, backend, optimizer, generation, video, meta, settings, prompt, restore,
        restoreGenerate, feedback, start, end, analyze: analyzeButton, analysis, apply,
        applyGenerate, status};
    optimizer.addEventListener("change", refreshGeneration);
    generation.addEventListener("change", selectGeneration);
    refreshBackendStatus();
    refreshGeneration();
}


function addStyles() {
    if (document.getElementById("h3-prompt-optimizer-style")) {
        return;
    }
    const style = document.createElement("style");
    style.id = "h3-prompt-optimizer-style";
    style.textContent = `
        .h3-optimizer-panel { display:flex; flex-direction:column; gap:10px; height:100%; overflow:auto; padding:12px; }
        .h3-optimizer-panel label { display:flex; flex-direction:column; gap:4px; font-size:12px; }
        .h3-optimizer-panel input, .h3-optimizer-panel select, .h3-optimizer-panel textarea,
        .h3-optimizer-panel button { box-sizing:border-box; width:100%; }
        .h3-optimizer-panel video { width:100%; max-height:38vh; background:#000; }
        .h3-optimizer-range, .h3-optimizer-actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
        .h3-optimizer-meta { font-size:11px; opacity:.75; overflow-wrap:anywhere; }
        .h3-optimizer-settings, .h3-optimizer-panel details pre { max-height:180px; overflow:auto; white-space:pre-wrap; font-size:11px; margin:0; }
        .h3-optimizer-analysis pre { max-height:260px; overflow:auto; white-space:pre-wrap; font-size:11px; }
        .h3-optimizer-analysis ul { padding-left:20px; }
        .h3-optimizer-status { min-height:20px; font-size:12px; }
        .h3-optimizer-status[data-kind="error"] { color:#ff7777; }
        .h3-optimizer-status[data-kind="success"] { color:#72d695; }
    `;
    document.head.append(style);
}


app.registerExtension({
    name: "Comfy.H3PromptOptimizer",
    setup() {
        addStyles();
        app.extensionManager.registerSidebarTab({
            id: SIDEBAR_ID,
            icon: "pi pi-video",
            title: "H3 Prompt Optimizer",
            tooltip: "MiniMax H3 video review and prompt improvement",
            type: "custom",
            render: renderSidebar,
        });
        api.addEventListener("execution_success", () => refreshGeneration());
    },
});
