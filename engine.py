import asyncio
import difflib
import hashlib
import json

from .prompt_document import (
    REFERENCE_FIELDS,
    allowed_patch_paths,
    apply_patch,
    compile_ref2va_prompt,
    defined_reference_labels,
    expected_reference_labels,
    parse_ref2va_prompt,
)
from .registry import create_analysis, generation_artifact, get_generation
from .schemas import CriticResult, PatchPlan
from .video_sampling import sample_video
from .vlm import generate_structured, inference_session


CRITIC_SYSTEM_PROMPT = """You are a grounded video critic for MiniMax H3 generations.
Return exactly one JSON object matching this shape:
{
  "issue_type": "one allowed issue type",
  "issue_confirmed": true,
  "confidence": 0.0,
  "localization": {"start_sec": 0.0, "end_sec": 0.0} or null,
  "observations": [{"text": "visible fact only", "start_sec": 0.0, "end_sec": 0.0}],
  "inferences": ["interpretation, kept separate from visible facts"],
  "root_cause": "one allowed root cause",
  "reason": "concise reason"
}
Allowed issue types: appearance, identity_consistency, motion_timing, motion_quality,
pose, object_consistency, camera_motion, shot_timing, action_order, spatial_relation,
audio_sync, dialogue, overall_style, composition, other.
Allowed root causes: PROMPT_UNDERSPECIFIED, PROMPT_AMBIGUOUS, PROMPT_CONFLICT,
PROMPT_TIMING_INSUFFICIENT, MODEL_STOCHASTICITY, MODEL_CAPABILITY_LIMIT,
REFERENCE_CONFLICT, INSUFFICIENT_EVIDENCE.
Classify the prompt before blaming model capability. If the human-requested visible state
is absent from the supplied prompt, use PROMPT_UNDERSPECIFIED. If the prompt requests the
opposite state, use PROMPT_CONFLICT. Do not say the video model ignored human feedback
when that feedback was not present in the generation prompt. Use MODEL_CAPABILITY_LIMIT
only when the requirement is already explicit, unambiguous, and internally consistent;
one failed generation by itself is not evidence of a capability limit.
Do not rewrite the prompt. Observations must state only what the supplied frames show.
Use frame timestamps as evidence. If the evidence is insufficient, say so explicitly."""


PATCH_SYSTEM_PROMPT = """You are a conservative MiniMax H3 Ref2VA Prompt IR patch planner.
Return exactly one JSON object matching this shape:
{
  "action": "PATCH_PROMPT or REGENERATE_SAME_PROMPT or CHANGE_REFERENCE or MANUAL_REVIEW or UNRESOLVED",
  "reason": "concise reason",
  "operations": [{"op": "replace", "path": "/allowed/path", "value": "replacement value"}],
  "changes": ["semantic change"],
  "preserved": ["unchanged intent"]
}
Never return a rewritten full prompt. Use only the allowed JSON Pointer paths supplied by
the caller. Preserve dialogue, visible text, subjects, reference labels, appearance,
camera, scene, audio, and all untargeted actions unless the feedback explicitly targets
that item. Treat the critic's root_cause as advisory: if human feedback identifies a
state that is missing from or contradicted by the prompt, return PATCH_PROMPT whenever
the allowed paths can express the correction, even if the critic selected
MODEL_CAPABILITY_LIMIT. When definition or retention paths are allowed, update them
together with the shot body as needed so the resulting prompt has no internal conflict.
A replacement shot body must remain complete and production-ready. If the prompt is
already sufficiently explicit and the result appears stochastic, use
REGENERATE_SAME_PROMPT with no operations. Use CHANGE_REFERENCE only when the supplied
reference itself conflicts with the requested result and an allowed prompt change cannot
resolve it. Use MANUAL_REVIEW only when no allowed prompt change or same-prompt
regeneration can reasonably address the feedback."""


TEMPORAL_ISSUES = {
    "motion_timing",
    "motion_quality",
    "pose",
    "object_consistency",
    "camera_motion",
    "shot_timing",
    "action_order",
}


def _hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expected_labels(generation):
    package = {"schema_version": 1}
    for name in REFERENCE_FIELDS:
        package[name] = {key: None for key in generation["reference_signature"].get(name, [])}
    return expected_reference_labels(package)


def _time_range(value):
    return value.model_dump() if value is not None else None


def _critic_user_prompt(generation, feedback, requested_range, previous=None):
    parts = [
        "Human feedback:",
        feedback,
        "",
        "Prompt used to generate the video:",
        generation["prompt"],
        "",
        "Video duration: %.3f seconds." % generation["video"]["duration_sec"],
    ]
    if requested_range:
        parts.append(
            "The user selected %.3f to %.3f seconds. Prioritize this interval."
            % (requested_range["start_sec"], requested_range["end_sec"])
        )
    if previous:
        parts.extend([
            "",
            "A global scan produced this preliminary result. Refine it using the denser frames:",
            json.dumps(previous, ensure_ascii=False),
        ])
    return "\n".join(parts)


async def _analyze_frames(generation, feedback, requested_range, path, cancel_event=None):
    duration = generation["video"]["duration_sec"]
    if requested_range:
        start = max(0.0, requested_range["start_sec"] - 0.5)
        end = min(duration, requested_range["end_sec"] + 0.5)
        frames = await asyncio.to_thread(
            sample_video, path, start, end, 24, 48, cancel_event=cancel_event
        )
        return await generate_structured(
            CRITIC_SYSTEM_PROMPT,
            _critic_user_prompt(generation, feedback, requested_range),
            frames,
            CriticResult,
        )

    frames = await asyncio.to_thread(
        sample_video, path, 0.0, duration, 4, 48, cancel_event=cancel_event
    )
    global_result = await generate_structured(
        CRITIC_SYSTEM_PROMPT,
        _critic_user_prompt(generation, feedback, None),
        frames,
        CriticResult,
    )
    if global_result.issue_type not in TEMPORAL_ISSUES or global_result.localization is None:
        return global_result

    localization = global_result.localization
    start = max(0.0, localization.start_sec - 0.5)
    end = min(duration, localization.end_sec + 0.5)
    focused_frames = await asyncio.to_thread(
        sample_video, path, start, end, 24, 48, cancel_event=cancel_event
    )
    return await generate_structured(
        CRITIC_SYSTEM_PROMPT,
        _critic_user_prompt(
            generation,
            feedback,
            {"start_sec": start, "end_sec": end},
            global_result.model_dump(),
        ),
        focused_frames,
        CriticResult,
    )


def _non_patch_plan(critic):
    if critic.issue_confirmed:
        return None
    return PatchPlan(
        action="UNRESOLVED",
        reason=critic.reason,
        operations=[],
        changes=[],
        preserved=[],
    )


async def analyze_generation(request, cancel_event=None):
    generation = get_generation(request.generation_id)
    if generation is None:
        raise ValueError("Generation was not found.")
    requested_range = _time_range(request.time_range)
    duration = generation["video"]["duration_sec"]
    if requested_range and requested_range["end_sec"] > duration + 0.001:
        raise ValueError("Review time range exceeds the generation duration.")

    labels = _expected_labels(generation)
    prompt_ir = parse_ref2va_prompt(
        generation["prompt"], generation["duration_seconds"], labels
    )
    unconnected_references = defined_reference_labels(prompt_ir) - labels
    if unconnected_references:
        names = ", ".join("<%s %d>" % item for item in sorted(unconnected_references))
        raise ValueError(
            "Stored generation prompt has no directly connected reference for: %s."
            % names
        )
    artifact_path = generation_artifact(request.generation_id)
    if artifact_path is None or not artifact_path.is_file():
        raise ValueError("Generation artifact is missing.")
    async with inference_session():
        critic = await _analyze_frames(
            generation, request.feedback.strip(), requested_range, artifact_path, cancel_event
        )

        patch = _non_patch_plan(critic)
        localization = critic.localization.model_dump() if critic.localization else requested_range
        allowed_paths = allowed_patch_paths(prompt_ir, critic.issue_type, localization)
        if patch is None:
            planner_input = {
                "human_feedback": request.feedback.strip(),
                "critic": critic.model_dump(),
                "allowed_paths": sorted(allowed_paths),
                "prompt_ir": prompt_ir,
            }
            patch = await generate_structured(
                PATCH_SYSTEM_PROMPT,
                json.dumps(planner_input, ensure_ascii=False),
                [],
                PatchPlan,
            )

    if patch.action == "PATCH_PROMPT":
        updated_ir = apply_patch(prompt_ir, patch.operations, allowed_paths)
        if defined_reference_labels(updated_ir) != defined_reference_labels(prompt_ir):
            raise ValueError("Prompt patch must preserve all reference labels.")
        optimized_prompt = compile_ref2va_prompt(updated_ir)
        parse_ref2va_prompt(optimized_prompt, generation["duration_seconds"], labels)
    else:
        optimized_prompt = generation["prompt"]

    diff = "\n".join(difflib.unified_diff(
        generation["prompt"].splitlines(),
        optimized_prompt.splitlines(),
        fromfile="current prompt",
        tofile="proposed prompt",
        lineterm="",
    ))
    optimized_hash = _hash_text(optimized_prompt)
    analysis_id = create_analysis(
        generation_id=request.generation_id,
        feedback=request.feedback.strip(),
        time_range=requested_range,
        target_state_hash=request.target_state_hash,
        critic=critic.model_dump(),
        prompt_ir=prompt_ir,
        patch=patch.model_dump(),
        optimized_prompt=optimized_prompt,
        optimized_prompt_hash=optimized_hash,
        diff=diff,
    )
    return {
        "analysis_id": analysis_id,
        "generation_id": request.generation_id,
        "optimizer_id": generation["optimizer_id"],
        "target_state_hash": request.target_state_hash,
        "original_prompt_hash": generation["prompt_hash"],
        "optimized_prompt_hash": optimized_hash,
        "optimized_prompt": optimized_prompt,
        "critic": critic.model_dump(),
        "proposal": patch.model_dump(),
        "allowed_paths": sorted(allowed_paths),
        "diff": diff,
    }
