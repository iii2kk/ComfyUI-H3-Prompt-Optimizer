import hashlib
from importlib import import_module
import json
import logging
import math

import torch

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context

from .prompt_document import (
    REFERENCE_FIELDS,
    compile_ref2va_prompt,
    defined_reference_labels,
    expected_reference_labels,
    parse_ref2va_prompt,
)
from .registry import register_generation


log = logging.getLogger(__name__)

MiniMaxH3Ref2VAPackageIO = io.Custom("MINIMAX_H3_REF2VA_PACKAGE")
H3OptimizerPromptStateIO = io.Custom("H3_OPTIMIZER_PROMPT_STATE")

_Ref2VAPromptPackageGenerator = import_module(
    "custom_nodes.ComfyUI-MiniMaxH3-Prompt-Generator.minimax_ref2va_package"
).MiniMaxH3Ref2VAPromptPackageGenerator


class _Unconnected:
    pass


_UNCONNECTED = _Unconnected()


def _hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _optimizer_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optimizer_id must not be empty.")
    value = value.strip()
    if len(value) > 100 or any(ord(character) < 32 for character in value):
        raise ValueError("optimizer_id is invalid.")
    return value


def _reference_signature(package):
    return {name: list(package[name]) for name in REFERENCE_FIELDS}


def _ordered_dict(values):
    if not values:
        return {}

    def order(item):
        suffix = item[0].rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    return {
        name: value
        for name, value in sorted(values.items(), key=order)
        if value is not None
    }


def _direct_package(prompt, ref_images=None, ref_videos=None,
                    ref_video_audios=None, ref_audios=None):
    return {
        "schema_version": 1,
        "prompt": prompt,
        "ref_images": _ordered_dict(ref_images),
        "ref_videos": _ordered_dict(ref_videos),
        "ref_video_audios": _ordered_dict(ref_video_audios),
        "ref_audios": _ordered_dict(ref_audios),
    }


def _missing_dynamic_inputs(values):
    if not isinstance(values, dict):
        return []
    needed = []
    for entry in values.values():
        if (isinstance(entry, tuple) and len(entry) == 2
                and entry[0] is None and isinstance(entry[1], str)):
            needed.append(entry[1])
    return needed


def _continuation_signature(continuation_frames):
    if continuation_frames is None:
        return {
            "connected": False,
            "frame_count": 0,
            "width": None,
            "height": None,
            "dtype": None,
            "sha256": None,
        }
    if (
        not isinstance(continuation_frames, torch.Tensor)
        or continuation_frames.ndim != 4
        or continuation_frames.shape[0] != 22
        or continuation_frames.shape[1] < 1
        or continuation_frames.shape[2] < 1
        or continuation_frames.shape[-1] != 3
        or not torch.is_floating_point(continuation_frames)
    ):
        raise ValueError(
            "continuation_frames must be a 22-frame ComfyUI IMAGE batch with shape [22, H, W, 3]."
        )
    value = continuation_frames.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(value.view(torch.uint8).numpy()).hexdigest()
    return {
        "connected": True,
        "frame_count": int(value.shape[0]),
        "width": int(value.shape[2]),
        "height": int(value.shape[1]),
        "dtype": str(value.dtype),
        "sha256": digest,
    }


def _frame_plan(duration_seconds, continuation_frame_count):
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be a positive finite number.")
    target_frames = int(math.floor(float(duration_seconds) * 24.0 + 0.5))
    if continuation_frame_count:
        delivered_frames = max(17, ((target_frames + 8) // 17) * 17)
        raw_frames = delivered_frames + continuation_frame_count
    else:
        raw_frames = max(5, (((target_frames - 5) + 8) // 17) * 17 + 5)
        delivered_frames = raw_frames
    return {
        "requested_duration_seconds": float(duration_seconds),
        "target_frames": target_frames,
        "raw_frames": raw_frames,
        "continuation_frames": continuation_frame_count,
        "delivered_frames": delivered_frames,
        "effective_delivered_seconds": delivered_frames / 24.0,
    }


def _optimizer_values(prompt, package, duration_seconds, optimizer_id,
                      approved, source_prompt=None, continuation_frames=None):
    continuation_signature = _continuation_signature(continuation_frames)
    frame_plan = _frame_plan(duration_seconds, continuation_signature["frame_count"])
    expected_labels = expected_reference_labels(package)
    document = parse_ref2va_prompt(prompt, duration_seconds, expected_labels)
    unconnected_references = defined_reference_labels(document) - expected_labels
    if unconnected_references:
        labels = ", ".join("<%s %d>" % item for item in sorted(unconnected_references))
        raise ValueError(
            "MiniMax H3 Ref2VA prompt has no directly connected reference for: %s."
            % labels
        )

    effective_prompt = compile_ref2va_prompt(document)
    effective_package = {**package, "prompt": effective_prompt}
    reference_signature = _reference_signature(effective_package)
    source_hash_text = source_prompt if isinstance(source_prompt, str) else effective_prompt
    state = {
        "schema_version": 1,
        "optimizer_id": optimizer_id,
        "mode": "ref2va",
        "duration_seconds": float(duration_seconds),
        "prompt": effective_prompt,
        "prompt_hash": _hash_text(effective_prompt),
        "source_prompt_hash": _hash_text(source_hash_text),
        "reference_signature": reference_signature,
        "reference_signature_hash": _hash_text(
            json.dumps(reference_signature, ensure_ascii=False, sort_keys=True)
        ),
        "continuation_signature": continuation_signature,
        "frame_plan": frame_plan,
        "approved": bool(approved),
    }
    return effective_prompt, effective_package, state, frame_plan["raw_frames"]


class H3OptimizerSegmentSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3OptimizerSegmentSettingsCS",
            display_name="H3 Optimizer Segment Settings",
            category="MiniMax H3/Prompt Optimizer",
            description=(
                "Provides one shared project and segment configuration for the Prompt "
                "Generator, Video Segment Checkpoint, and Optimizer Video Output."
            ),
            inputs=[
                io.String.Input("project_name", default="my_movie"),
                io.Int.Input("segment_id", default=1, min=1, max=1_000_000),
                io.Int.Input(
                    "resume_from",
                    default=1,
                    min=1,
                    max=1_000_001,
                    tooltip="Segments before this value are loaded from checkpoints.",
                ),
                io.Int.Input(
                    "continuation_frame_count",
                    default=22,
                    min=1,
                    max=4096,
                    tooltip="Frames saved by this segment for the next segment's visual continuation.",
                ),
                io.Int.Input(
                    "blend_frames",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="Boundary frames blended when this segment follows another segment.",
                ),
            ],
            outputs=[
                io.String.Output("optimizer_id"),
                io.String.Output("project_name"),
                io.Int.Output("segment_id"),
                io.Int.Output("resume_from"),
                io.Int.Output("continuation_frame_count"),
                io.Int.Output("blend_frames"),
            ],
        )

    @classmethod
    def execute(cls, project_name, segment_id, resume_from,
                continuation_frame_count, blend_frames):
        if not isinstance(project_name, str) or not project_name:
            raise ValueError("project_name must not be empty.")
        if project_name != project_name.strip():
            raise ValueError("project_name must not have surrounding whitespace.")
        positive_values = (segment_id, resume_from, continuation_frame_count)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in positive_values):
            raise ValueError(
                "segment_id, resume_from, and continuation_frame_count must be positive integers."
            )
        if isinstance(blend_frames, bool) or not isinstance(blend_frames, int) or blend_frames < 0:
            raise ValueError("blend_frames must be a non-negative integer.")
        if blend_frames > continuation_frame_count:
            raise ValueError("blend_frames must not exceed continuation_frame_count.")
        optimizer_id = _optimizer_id(
            "%s_segment_%03d" % (project_name, segment_id)
        )
        return io.NodeOutput(
            optimizer_id,
            project_name,
            segment_id,
            resume_from,
            continuation_frame_count,
            blend_frames,
        )


class H3OptimizerRef2VAPromptPackageGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3OptimizerRef2VAPromptPackageGeneratorCS",
            display_name="H3 Optimizer Ref2VA Prompt Package Generator",
            category="MiniMax H3/Prompt Optimizer",
            description=(
                "Generates the initial Ref2VA prompt with the existing MiniMax H3 Generator. "
                "Approved mode skips prompt generation and reuses the same references."
            ),
            search_aliases=["prompt package", "ref2va", "prompt optimizer"],
            inputs=[
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", optional=True),
                        prefix="ref_image_", min=0, max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_video",
                            optional=True,
                            tooltip="24fpsの参照動画フレーム（IMAGEバッチ）",
                        ),
                        prefix="ref_video_", min=0, max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            optional=True,
                            tooltip="同じ番号のref_videoに対応する音声",
                        ),
                        prefix="ref_video_audio_", min=0, max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_audio",
                            optional=True,
                            tooltip="動画とは独立した参照音声",
                        ),
                        prefix="ref_audio_", min=0, max=3,
                    ),
                ),
                io.Image.Input(
                    "continuation_frames",
                    optional=True,
                    tooltip="Previous segment's exact final 22 frames. Used for prompt context, not as a Ref2VA reference.",
                ),
                io.String.Input(
                    "reference_notes",
                    multiline=True,
                    default="",
                    tooltip="各参照素材の役割。音声では内容とcopy/referenceの指定が必須です。",
                    extra_dict={"widgetType": "MINIMAX_PROMPT_TEXTAREA"},
                ),
                io.String.Input(
                    "instruction",
                    multiline=True,
                    default="",
                    tooltip="作りたい動画を日本語で記述します。台詞は原文のまま保持されます。",
                    extra_dict={"widgetType": "MINIMAX_PROMPT_TEXTAREA"},
                ),
                io.Int.Input("duration_seconds", default=5, min=4, max=15),
                io.String.Input("optimizer_id", default="main_video"),
                io.Boolean.Input("use_approved_prompt", default=False),
                io.String.Input(
                    "approved_prompt",
                    multiline=True,
                    default="",
                    tooltip="Set by the Prompt Optimizer sidebar after human approval.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "temperature", default=1.0, min=0.0, max=2.0, step=0.05, advanced=True,
                ),
                io.Float.Input(
                    "top_p", default=0.95, min=0.0, max=1.0, step=0.01, advanced=True,
                ),
                io.Int.Input(
                    "max_tokens", default=4096, min=256, max=65536, step=256, advanced=True,
                ),
                io.Int.Input(
                    "timeout_seconds", default=300, min=10, max=1800, step=10, advanced=True,
                ),
                io.Int.Input(
                    "max_image_edge", default=1024, min=256, max=2048, step=64, advanced=True,
                ),
                io.Int.Input(
                    "max_video_frames", default=8, min=2, max=32, advanced=True,
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="Effective prompt"),
                MiniMaxH3Ref2VAPackageIO.Output("package", display_name="Effective package"),
                H3OptimizerPromptStateIO.Output("prompt_state"),
                io.Int.Output("raw_frames"),
            ],
        )

    @classmethod
    async def execute(cls, reference_notes="", instruction="", duration_seconds=5,
                      optimizer_id="main_video", use_approved_prompt=False,
                      approved_prompt="", seed=0, temperature=1.0, top_p=0.95,
                      max_tokens=4096, timeout_seconds=300, max_image_edge=1024,
                      max_video_frames=8, ref_images=None, ref_videos=None,
                      ref_video_audios=None, ref_audios=None,
                      continuation_frames=None):
        optimizer_id = _optimizer_id(optimizer_id)
        if use_approved_prompt:
            package = _direct_package(
                approved_prompt,
                ref_images,
                ref_videos,
                ref_video_audios,
                ref_audios,
            )
            return io.NodeOutput(*_optimizer_values(
                approved_prompt,
                package,
                duration_seconds,
                optimizer_id,
                approved=True,
                continuation_frames=continuation_frames,
            ))

        generated = await _Ref2VAPromptPackageGenerator.execute(
            reference_notes=reference_notes,
            instruction=instruction,
            duration_seconds=duration_seconds,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_image_edge=max_image_edge,
            max_video_frames=max_video_frames,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
            continuation_frames=continuation_frames,
        )
        prompt, package = generated[0], generated[1]
        if prompt != package.get("prompt"):
            raise ValueError("Generated MiniMax H3 Ref2VA prompt does not match its package.")
        return io.NodeOutput(*_optimizer_values(
            prompt,
            package,
            duration_seconds,
            optimizer_id,
            approved=False,
            source_prompt=prompt,
            continuation_frames=continuation_frames,
        ))


class H3OptimizerRef2VAPromptPackage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3OptimizerRef2VAPromptPackageCS",
            display_name="H3 Optimizer Ref2VA Prompt Package",
            category="MiniMax H3/Prompt Optimizer",
            description=(
                "Uses the generated Prompt/Package for the initial run. In approved mode it "
                "skips those lazy inputs and builds the Package from direct references."
            ),
            inputs=[
                io.String.Input("source_prompt", force_input=True, optional=True, lazy=True),
                MiniMaxH3Ref2VAPackageIO.Input("source_package", optional=True, lazy=True),
                io.Int.Input("duration_seconds", default=5, min=4, max=15),
                io.String.Input("optimizer_id", default="main_video"),
                io.Boolean.Input("use_approved_prompt", default=False),
                io.String.Input(
                    "approved_prompt",
                    multiline=True,
                    default="",
                    tooltip="Set by the Prompt Optimizer sidebar after human approval.",
                ),
                io.Autogrow.Input(
                    "approved_ref_images",
                    optional=True,
                    lazy=True,
                    tooltip="Connect the same image references as the Generator. Used only in approved mode.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("approved_ref_image", optional=True, lazy=True),
                        prefix="ref_image_", min=0, max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "approved_ref_videos",
                    optional=True,
                    lazy=True,
                    tooltip="Connect the same video references as the Generator. Used only in approved mode.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("approved_ref_video", optional=True, lazy=True),
                        prefix="ref_video_", min=0, max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "approved_ref_video_audios",
                    optional=True,
                    lazy=True,
                    tooltip="Audio paired with approved_ref_videos. Used only in approved mode.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("approved_ref_video_audio", optional=True, lazy=True),
                        prefix="ref_video_audio_", min=0, max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "approved_ref_audios",
                    optional=True,
                    lazy=True,
                    tooltip="Connect the same standalone audio references as the Generator.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("approved_ref_audio", optional=True, lazy=True),
                        prefix="ref_audio_", min=0, max=3,
                    ),
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="Effective prompt"),
                MiniMaxH3Ref2VAPackageIO.Output("package", display_name="Effective package"),
                H3OptimizerPromptStateIO.Output("prompt_state"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, use_approved_prompt=False, source_prompt=_UNCONNECTED,
                          source_package=_UNCONNECTED, approved_ref_images=None,
                          approved_ref_videos=None, approved_ref_video_audios=None,
                          approved_ref_audios=None, **_):
        if not use_approved_prompt:
            needed = []
            if source_prompt is None:
                needed.append("source_prompt")
            if source_package is None:
                needed.append("source_package")
            return needed

        needed = []
        for values in (
            approved_ref_images,
            approved_ref_videos,
            approved_ref_video_audios,
            approved_ref_audios,
        ):
            needed.extend(_missing_dynamic_inputs(values))
        return needed

    @classmethod
    def execute(cls, duration_seconds, optimizer_id, use_approved_prompt=False,
                approved_prompt="", source_prompt=None, source_package=None,
                approved_ref_images=None, approved_ref_videos=None,
                approved_ref_video_audios=None, approved_ref_audios=None):
        optimizer_id = _optimizer_id(optimizer_id)
        if use_approved_prompt:
            effective_prompt = approved_prompt
            effective_package = _direct_package(
                effective_prompt,
                approved_ref_images,
                approved_ref_videos,
                approved_ref_video_audios,
                approved_ref_audios,
            )
        else:
            if source_prompt is None or source_package is None:
                raise ValueError(
                    "Connect source_prompt and source_package, or enable use_approved_prompt."
                )
            package_prompt = source_package.get("prompt")
            if source_prompt != package_prompt:
                raise ValueError("Source MiniMax H3 Ref2VA prompt does not match its package.")
            effective_prompt = source_prompt
            effective_package = source_package
        values = _optimizer_values(
            effective_prompt,
            effective_package,
            duration_seconds,
            optimizer_id,
            approved=use_approved_prompt,
            source_prompt=source_prompt,
        )
        return io.NodeOutput(*values[:3])


class H3OptimizerVideoOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3OptimizerVideoOutputCS",
            display_name="H3 Optimizer Video Output",
            category="MiniMax H3/Prompt Optimizer",
            description=(
                "Registers regenerated checkpoint video with its executed prompt and workflow "
                "without evaluating the prompt branch when the checkpoint is in LOAD mode."
            ),
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.Video.Input("video", lazy=True),
                H3OptimizerPromptStateIO.Input("prompt_state", lazy=True),
                io.String.Input("optimizer_id", default="main_video"),
                io.String.Input("project_name", default="my_movie"),
                io.Int.Input("segment_id", default=1, min=1, max=1_000_000),
                io.Int.Input(
                    "resume_from",
                    default=1,
                    min=1,
                    max=1_000_001,
                    tooltip="Connect the same value used by Video Segment Checkpoint.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **_):
        return float("NaN")

    @classmethod
    def check_lazy_status(cls, segment_id=1, resume_from=1, video=None,
                          prompt_state=None, **_):
        needed = []
        if video is None:
            needed.append("video")
        if int(segment_id) >= int(resume_from) and prompt_state is None:
            needed.append("prompt_state")
        return needed

    @classmethod
    def execute(cls, video, optimizer_id, project_name, segment_id, resume_from,
                prompt_state=None):
        optimizer_id = _optimizer_id(optimizer_id)
        segment_id = int(segment_id)
        resume_from = int(resume_from)
        if segment_id < 1 or resume_from < 1:
            raise ValueError("segment_id and resume_from must be positive integers.")
        if segment_id < resume_from:
            log.info(
                "[H3PromptOptimizer] LOAD project=%s segment=%d; generation not registered",
                project_name, segment_id,
            )
            return io.NodeOutput()
        if not isinstance(prompt_state, dict) or prompt_state.get("schema_version") != 1:
            raise ValueError("Invalid H3 Prompt Optimizer prompt state.")
        if prompt_state.get("optimizer_id") != optimizer_id:
            raise ValueError("H3 Prompt Optimizer IDs do not match.")

        api_prompt = cls.hidden.prompt
        if not isinstance(api_prompt, dict):
            raise RuntimeError("H3 Optimizer Video Output requires the executed ComfyUI prompt.")
        extra_pnginfo = cls.hidden.extra_pnginfo
        workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        if workflow is not None and not isinstance(workflow, dict):
            raise ValueError("ComfyUI workflow metadata must be a JSON object.")

        context = get_executing_context()
        if context is None:
            raise RuntimeError("H3 Optimizer Video Output requires a ComfyUI execution context.")
        generation = register_generation(
            video=video,
            prompt_state=prompt_state,
            optimizer_id=optimizer_id,
            project_name=project_name,
            segment_id=segment_id,
            comfy_prompt_id=str(context.prompt_id),
            recorder_node_id=str(context.node_id),
            api_prompt=api_prompt,
            workflow=workflow,
        )
        log.info(
            "[H3PromptOptimizer] registered %s project=%s segment=%d",
            generation["generation_id"], project_name, segment_id,
        )
        return io.NodeOutput()
