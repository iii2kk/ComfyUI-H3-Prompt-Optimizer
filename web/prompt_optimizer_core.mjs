const SEED_INPUTS = new Set(["seed", "noise_seed"]);


function graphNode(graph, id) {
    const direct = graph?.getNodeById?.(id);
    if (direct) {
        return direct;
    }
    const numericId = Number(id);
    return Number.isInteger(numericId) ? graph?.getNodeById?.(numericId) : null;
}


export function executionNode(rootGraph, executionId) {
    const ids = String(executionId).split(":");
    let graph = rootGraph;
    let node = null;

    for (let index = 0; index < ids.length; index++) {
        node = graphNode(graph, ids[index]);
        if (!node) {
            return null;
        }
        if (index < ids.length - 1) {
            graph = node.subgraph;
            if (!graph) {
                return null;
            }
        }
    }
    return node;
}


export function restoreExecutedWidgetValues(rootGraph, apiPrompt, setWidgetValue) {
    for (const [executionId, promptNode] of Object.entries(apiPrompt || {})) {
        const values = Object.entries(promptNode.inputs || {}).filter(
            ([, value]) => ["string", "number", "boolean"].includes(typeof value),
        );
        if (!values.length) {
            continue;
        }

        const node = executionNode(rootGraph, executionId);
        if (!node) {
            throw new Error(`実行ノード ${executionId} を復元できません。`);
        }

        for (const [name, value] of values) {
            if (!node.widgets?.some((item) => item.name === name)) {
                continue;
            }
            setWidgetValue(node, name, value);
            if (SEED_INPUTS.has(name) &&
                node.widgets?.some((item) => item.name === "control_after_generate")) {
                setWidgetValue(node, "control_after_generate", "fixed");
            }
        }
    }
}


export function restoreGenerationPrompt(node, prompt, setWidgetValue) {
    setWidgetValue(node, "approved_prompt", prompt);
    setWidgetValue(node, "use_approved_prompt", true);
}
