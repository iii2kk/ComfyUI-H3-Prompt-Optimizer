#!/usr/bin/env node

import assert from "node:assert/strict";
import {
    executionNode,
    restoreExecutedWidgetValues,
    restoreGenerationPrompt,
} from "../web/prompt_optimizer_core.mjs";


function widget(name, value) {
    return {name, value};
}


function graph(nodes) {
    const byId = new Map(nodes.map((node) => [String(node.id), node]));
    return {
        getNodeById(id) {
            return byId.get(String(id)) || null;
        },
    };
}


const randomNoise = {
    id: 610,
    widgets: [
        widget("noise_seed", 999),
        widget("control_after_generate", "randomize"),
    ],
};
const scheduler = {
    id: 607,
    widgets: [widget("scheduler", "normal"), widget("steps", 8), widget("denoise", 0.5)],
};
const nestedSampler = {id: 602, subgraph: graph([randomNoise, scheduler]), widgets: []};
const generation = {id: 594, subgraph: graph([nestedSampler]), widgets: []};
const segment = {id: 580, subgraph: graph([generation]), widgets: []};
const promptGenerator = {
    id: 590,
    widgets: [
        widget("seed", 111),
        widget("control_after_generate", "randomize"),
        widget("temperature", 0.5),
        widget("approved_prompt", "old prompt"),
        widget("use_approved_prompt", false),
    ],
};
const root = graph([segment, promptGenerator]);

assert.equal(executionNode(root, "590"), promptGenerator);
assert.equal(executionNode(root, "580:594:602:610"), randomNoise);
assert.equal(executionNode(root, "580:missing"), null);

const changes = [];
restoreExecutedWidgetValues(root, {
    "590": {
        inputs: {
            seed: 411681109320202,
            temperature: 1.0,
            duration_seconds: ["583", 1],
        },
    },
    "580:594:602:607": {
        inputs: {scheduler: "simple", steps: 20, denoise: 1},
    },
    "580:594:602:610": {
        inputs: {noise_seed: 1013058537965327},
    },
}, (node, name, value) => {
    const target = node.widgets.find((item) => item.name === name);
    assert.ok(target);
    target.value = value;
    changes.push([node.id, name, value]);
});

assert.equal(promptGenerator.widgets[0].value, 411681109320202);
assert.equal(promptGenerator.widgets[1].value, "fixed");
assert.equal(promptGenerator.widgets[2].value, 1.0);
assert.deepEqual(scheduler.widgets.map((item) => item.value), ["simple", 20, 1]);
assert.equal(randomNoise.widgets[0].value, 1013058537965327);
assert.equal(randomNoise.widgets[1].value, "fixed");
assert.deepEqual(changes.filter(([, name]) => name === "control_after_generate"), [
    [590, "control_after_generate", "fixed"],
    [610, "control_after_generate", "fixed"],
]);

const executedPrompt = "The exact effective prompt stored with the reviewed generation.";
restoreGenerationPrompt(promptGenerator, executedPrompt, (node, name, value) => {
    const target = node.widgets.find((item) => item.name === name);
    assert.ok(target);
    target.value = value;
});
assert.equal(promptGenerator.widgets.find((item) => item.name === "approved_prompt").value,
    executedPrompt);
assert.equal(promptGenerator.widgets.find((item) => item.name === "use_approved_prompt").value,
    true);

assert.throws(
    () => restoreExecutedWidgetValues(root, {"580:999": {inputs: {seed: 1}}}, () => {}),
    /実行ノード 580:999 を復元できません/,
);

console.log("H3 Prompt Optimizer: workflow restore fixes executed Prompt, subgraphs, and seeds");
