import asyncio
import base64
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
PACKAGE_NAME = "h3_prompt_optimizer_under_test"

if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

if PACKAGE_NAME not in sys.modules:
    package_module = types.ModuleType(PACKAGE_NAME)
    package_module.__path__ = [str(PLUGIN_ROOT)]
    sys.modules[PACKAGE_NAME] = package_module


def plugin_module(name):
    return importlib.import_module("%s.%s" % (PACKAGE_NAME, name))


PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the woman defined by <Picture 1>.\n\n"
    "summary:\nA woman crosses a quiet room.\n\n"
    "retention_analysis:\n<Picture 1> defines her identity and clothing.\n\n"
    "detailed_description:\n"
    "[Shot 1] A static medium shot shows <Subject 1> standing still.\n\n"
    "[Shot 2] At 00:02.000, she walks slowly toward the window.\n\n"
    "overall_soundscape:\nSoft footsteps in a quiet room.\n\n"
    "non_diegetic_music:\nN/A"
)


def ref2va_package(prompt=PROMPT):
    return {
        "schema_version": 1,
        "prompt": prompt,
        "ref_images": {"ref_image_0": object()},
        "ref_videos": {},
        "ref_video_audios": {},
        "ref_audios": {},
    }


def prompt_state(prompt=PROMPT):
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "prompt": prompt,
        "prompt_hash": digest,
        "source_prompt_hash": digest,
        "reference_signature": {
            "ref_images": ["ref_image_0"],
            "ref_videos": [],
            "ref_video_audios": [],
            "ref_audios": [],
        },
        "duration_seconds": 5,
    }


class FakeVideo:
    def save_to(self, path, **_kwargs):
        Path(path).write_bytes(b"fake mp4 payload")

    def get_frame_rate(self):
        return 24.0

    def get_frame_count(self):
        return 120


class PromptDocumentTests(unittest.TestCase):
    def test_round_trip_and_locked_patch(self):
        prompt_document = plugin_module("prompt_document")
        schemas = plugin_module("schemas")
        document = prompt_document.parse_ref2va_prompt(PROMPT, 5, {("Picture", 1)})
        self.assertEqual(prompt_document.compile_ref2va_prompt(document), PROMPT)
        self.assertEqual(
            prompt_document.allowed_patch_paths(
                document,
                "motion_timing",
                {"start_sec": 2.1, "end_sec": 3.0},
            ),
            {"/sections/detailed_description/shots/1/body"},
        )

        operation = schemas.PatchOperation(
            op="replace",
            path="/sections/detailed_description/shots/1/body",
            value="She walks toward the window over three seconds in one smooth motion.",
        )
        updated = prompt_document.apply_patch(
            document,
            [operation],
            {"/sections/detailed_description/shots/1/body"},
        )
        self.assertEqual(
            updated["sections"]["subject_definitions"],
            document["sections"]["subject_definitions"],
        )
        with self.assertRaisesRegex(ValueError, "locked path"):
            prompt_document.apply_patch(
                document,
                [schemas.PatchOperation(
                    op="replace",
                    path="/sections/overall_soundscape",
                    value="Silence.",
                )],
                {"/sections/detailed_description/shots/1/body"},
            )

    def test_object_consistency_can_patch_definitions_retention_and_shot(self):
        prompt_document = plugin_module("prompt_document")
        document = prompt_document.parse_ref2va_prompt(PROMPT, 5, {("Picture", 1)})
        self.assertEqual(
            prompt_document.allowed_patch_paths(
                document,
                "object_consistency",
                {"start_sec": 0.0, "end_sec": 5.0},
            ),
            {
                "/sections/subject_definitions",
                "/sections/retention_analysis",
                "/sections/detailed_description/shots/1/body",
            },
        )


class VideoSamplingTests(unittest.TestCase):
    def test_sampling_stops_before_opening_video_when_cancelled(self):
        video_sampling = plugin_module("video_sampling")
        cancel_event = mock.Mock()
        cancel_event.is_set.return_value = True

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            video_sampling.sample_video(
                "missing.mp4", 0, 5, 4, cancel_event=cancel_event
            )


class NodeContractTests(unittest.TestCase):
    def test_segment_settings_derives_optimizer_id_and_shares_values(self):
        nodes = plugin_module("nodes")
        settings = nodes.H3OptimizerSegmentSettings
        schema = settings.define_schema()

        self.assertEqual(
            [item.id for item in schema.outputs],
            [
                "optimizer_id", "project_name", "segment_id", "resume_from",
                "continuation_frame_count", "blend_frames",
            ],
        )
        self.assertEqual(
            settings.execute("my_movie", 7, 4, 22, 5).result,
            ("my_movie_segment_007", "my_movie", 7, 4, 22, 5),
        )
        self.assertEqual(
            settings.execute("my_movie", 1001, 1, 22, 0)[0],
            "my_movie_segment_1001",
        )
        with self.assertRaisesRegex(ValueError, "project_name"):
            settings.execute(" my_movie", 1, 1, 22, 0)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            settings.execute("my_movie", 1.5, 1, 22, 0)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            settings.execute("my_movie", 1, 1, 4, 5)

    def test_combined_node_generates_initial_prompt_with_existing_generator(self):
        nodes = plugin_module("nodes")
        source_package = ref2va_package()
        image = source_package["ref_images"]["ref_image_0"]
        generated = nodes.io.NodeOutput(PROMPT, source_package)
        generator_execute = mock.AsyncMock(return_value=generated)

        with mock.patch.object(
            nodes._Ref2VAPromptPackageGenerator,
            "execute",
            new=generator_execute,
        ):
            output = asyncio.run(
                nodes.H3OptimizerRef2VAPromptPackageGenerator.execute(
                    reference_notes="Picture 1 defines the subject.",
                    instruction="A woman crosses a quiet room.",
                    duration_seconds=5,
                    optimizer_id="segment_1",
                    seed=42,
                    reasoning_effort="low",
                    ref_images={"ref_image_0": image},
                )
            )

        self.assertEqual(generator_execute.await_count, 1)
        self.assertIs(
            generator_execute.await_args.kwargs["ref_images"]["ref_image_0"],
            image,
        )
        self.assertEqual(
            generator_execute.await_args.kwargs["reasoning_effort"], "low"
        )
        self.assertEqual(output[0], PROMPT)
        self.assertEqual(output[1]["prompt"], PROMPT)
        self.assertFalse(output[2]["approved"])
        self.assertEqual(output[3], 124)
        self.assertEqual(output[2]["frame_plan"]["delivered_frames"], 124)
        self.assertFalse(output[2]["continuation_signature"]["connected"])

    def test_combined_approved_mode_skips_existing_generator(self):
        nodes = plugin_module("nodes")
        image = object()
        generator_execute = mock.AsyncMock()

        with mock.patch.object(
            nodes._Ref2VAPromptPackageGenerator,
            "execute",
            new=generator_execute,
        ):
            output = asyncio.run(
                nodes.H3OptimizerRef2VAPromptPackageGenerator.execute(
                    duration_seconds=5,
                    optimizer_id="segment_1",
                    use_approved_prompt=True,
                    approved_prompt=PROMPT,
                    ref_images={"ref_image_0": image},
                )
            )

        generator_execute.assert_not_awaited()
        self.assertIs(output[1]["ref_images"]["ref_image_0"], image)
        self.assertEqual(output[1]["prompt"], PROMPT)
        self.assertTrue(output[2]["approved"])

        with self.assertRaisesRegex(ValueError, "no directly connected reference"):
            asyncio.run(
                nodes.H3OptimizerRef2VAPromptPackageGenerator.execute(
                    duration_seconds=5,
                    optimizer_id="segment_1",
                    use_approved_prompt=True,
                    approved_prompt=PROMPT,
                )
            )

    def test_combined_node_uses_continuation_for_prompt_and_frame_plan(self):
        nodes = plugin_module("nodes")
        continuation = torch.arange(22 * 2 * 2 * 3, dtype=torch.float32).reshape(22, 2, 2, 3)
        continued_prompt = PROMPT.replace(
            "summary:\nA woman crosses a quiet room.",
            "summary:\nvideo continuation. A woman crosses a quiet room.",
        )
        source_package = ref2va_package(continued_prompt)
        generator_execute = mock.AsyncMock(
            return_value=nodes.io.NodeOutput(continued_prompt, source_package)
        )

        with mock.patch.object(
            nodes._Ref2VAPromptPackageGenerator, "execute", new=generator_execute
        ):
            output = asyncio.run(
                nodes.H3OptimizerRef2VAPromptPackageGenerator.execute(
                    instruction="Continue the motion.",
                    duration_seconds=5,
                    optimizer_id="segment_2",
                    continuation_frames=continuation,
                    ref_images=source_package["ref_images"],
                )
            )

        self.assertIs(
            generator_execute.await_args.kwargs["continuation_frames"], continuation
        )
        self.assertNotIn("continuation_frames", output[1])
        self.assertEqual(output[3], 141)
        self.assertEqual(output[2]["frame_plan"]["continuation_frames"], 22)
        self.assertEqual(output[2]["frame_plan"]["delivered_frames"], 119)
        signature = output[2]["continuation_signature"]
        self.assertTrue(signature["connected"])
        self.assertEqual((signature["width"], signature["height"]), (2, 2))
        self.assertEqual(len(signature["sha256"]), 64)

        with self.assertRaisesRegex(ValueError, "22-frame"):
            asyncio.run(
                nodes.H3OptimizerRef2VAPromptPackageGenerator.execute(
                    duration_seconds=5,
                    optimizer_id="segment_2",
                    use_approved_prompt=True,
                    approved_prompt=continued_prompt,
                    continuation_frames=continuation[:21],
                    ref_images=source_package["ref_images"],
                )
            )

    def test_rebinds_prompt_without_mutating_source_package(self):
        nodes = plugin_module("nodes")
        source_package = ref2va_package()
        output = nodes.H3OptimizerRef2VAPromptPackage.execute(
            source_prompt=PROMPT,
            source_package=source_package,
            duration_seconds=5,
            optimizer_id="segment_1",
        )
        effective_prompt, effective_package, optimizer_state = output.result

        self.assertEqual(effective_prompt, PROMPT)
        self.assertIsNot(effective_package, source_package)
        self.assertEqual(effective_package["prompt"], effective_prompt)
        self.assertIs(
            effective_package["ref_images"]["ref_image_0"],
            source_package["ref_images"]["ref_image_0"],
        )
        self.assertEqual(source_package["prompt"], PROMPT)
        self.assertEqual(optimizer_state["optimizer_id"], "segment_1")
        self.assertFalse(optimizer_state["approved"])

        approved = PROMPT.replace(
            "she walks slowly toward the window.",
            "she walks toward the window over three seconds in one smooth motion.",
        )
        approved_output = nodes.H3OptimizerRef2VAPromptPackage.execute(
            source_prompt=PROMPT,
            source_package=source_package,
            duration_seconds=5,
            optimizer_id="segment_1",
            use_approved_prompt=True,
            approved_prompt=approved,
            approved_ref_images={"ref_image_0": source_package["ref_images"]["ref_image_0"]},
        )
        self.assertEqual(approved_output[0], approved)
        self.assertEqual(approved_output[1]["prompt"], approved)
        self.assertTrue(approved_output[2]["approved"])

    def test_rejects_prompt_package_mismatch_and_missing_reference(self):
        nodes = plugin_module("nodes")
        with self.assertRaisesRegex(ValueError, "does not match"):
            nodes.H3OptimizerRef2VAPromptPackage.execute(
                source_prompt=PROMPT + " ",
                source_package=ref2va_package(),
                duration_seconds=5,
                optimizer_id="segment_1",
            )
        invalid_prompt = PROMPT.replace("<Picture 1>", "the reference", 2)
        with self.assertRaisesRegex(ValueError, "missing supplied reference labels"):
            nodes.H3OptimizerRef2VAPromptPackage.execute(
                source_prompt=invalid_prompt,
                source_package=ref2va_package(invalid_prompt),
                duration_seconds=5,
                optimizer_id="segment_1",
            )
        unconnected_prompt = PROMPT.replace(
            "<Subject 1> is the woman defined by <Picture 1>.",
            "<Subject 1> is the woman defined by <Picture 1>.\n"
            "<Video 1> is not provided; no video references apply.",
        )
        with self.assertRaisesRegex(ValueError, "no directly connected reference"):
            nodes.H3OptimizerRef2VAPromptPackage.execute(
                source_prompt=unconnected_prompt,
                source_package=ref2va_package(unconnected_prompt),
                duration_seconds=5,
                optimizer_id="segment_1",
            )

    def test_existing_type_and_persistent_text_contracts_match(self):
        nodes = plugin_module("nodes")
        prompt_schema = nodes.H3OptimizerRef2VAPromptPackage.define_schema()
        combined_schema = nodes.H3OptimizerRef2VAPromptPackageGenerator.define_schema()
        inputs = {item.id: item for item in prompt_schema.inputs}
        outputs = {item.id: item for item in prompt_schema.outputs}
        combined_inputs = {item.id: item for item in combined_schema.inputs}
        combined_outputs = {item.id: item for item in combined_schema.outputs}
        self.assertEqual(combined_inputs["max_tokens"].max, 65536)
        self.assertEqual(
            combined_inputs["reasoning_effort"].options,
            ["none", "low", "medium", "xhigh"],
        )
        self.assertIn("continuation_frames", combined_inputs)
        self.assertIn("raw_frames", combined_outputs)
        self.assertEqual(
            inputs["source_package"].io_type,
            "MINIMAX_H3_REF2VA_PACKAGE",
        )
        self.assertEqual(
            outputs["package"].io_type,
            "MINIMAX_H3_REF2VA_PACKAGE",
        )
        self.assertEqual(
            combined_outputs["package"].io_type,
            "MINIMAX_H3_REF2VA_PACKAGE",
        )

        persistent_path = PLUGIN_ROOT.parent / "ComfyUI-My-Custom-Scripts" / "persistent_text.py"
        spec = importlib.util.spec_from_file_location("persistent_text_under_test", persistent_path)
        persistent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(persistent_module)
        result = persistent_module.PersistentText().store(stored_text="old", text=PROMPT)
        self.assertEqual(result["result"], (PROMPT,))

    def test_video_output_lazy_contract_preserves_checkpoint_load_skip(self):
        nodes = plugin_module("nodes")
        recorder = nodes.H3OptimizerVideoOutput
        self.assertEqual(
            recorder.check_lazy_status(
                segment_id=1,
                resume_from=2,
                video=object(),
                prompt_state=None,
            ),
            [],
        )
        self.assertEqual(
            recorder.check_lazy_status(
                segment_id=2,
                resume_from=2,
                video=object(),
                prompt_state=None,
            ),
            ["prompt_state"],
        )
        self.assertEqual(
            recorder.check_lazy_status(
                segment_id=2,
                resume_from=2,
                video=None,
                prompt_state=None,
            ),
            ["video", "prompt_state"],
        )

    def test_video_output_records_executed_prompt_and_workflow(self):
        nodes = plugin_module("nodes")
        recorder = nodes.H3OptimizerVideoOutput
        state = prompt_state()
        state.update({"schema_version": 1, "optimizer_id": "segment_1"})
        api_prompt = {"12": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}}}
        workflow = {"nodes": [{"id": 12, "type": "RandomNoise"}], "links": []}
        previous_hidden = recorder.hidden
        recorder.hidden = types.SimpleNamespace(
            prompt=api_prompt,
            extra_pnginfo={"workflow": workflow},
        )
        try:
            with mock.patch.object(
                nodes,
                "get_executing_context",
                return_value=types.SimpleNamespace(prompt_id="prompt_1", node_id="node_1"),
            ), mock.patch.object(
                nodes,
                "register_generation",
                return_value={"generation_id": "generation_1"},
            ) as register:
                recorder.execute(
                    video=FakeVideo(),
                    prompt_state=state,
                    optimizer_id="segment_1",
                    project_name="movie",
                    segment_id=1,
                    resume_from=1,
                )
        finally:
            recorder.hidden = previous_hidden

        self.assertEqual(register.call_args.kwargs["api_prompt"], api_prompt)
        self.assertEqual(register.call_args.kwargs["workflow"], workflow)

    def test_approved_mode_skips_generator_inputs_and_requests_direct_references(self):
        nodes = plugin_module("nodes")
        optimizer = nodes.H3OptimizerRef2VAPromptPackage
        self.assertEqual(
            optimizer.check_lazy_status(
                use_approved_prompt=False,
                source_prompt=None,
                source_package=None,
                approved_ref_images={
                    "ref_image_0": (None, "approved_ref_images.ref_image_0")
                },
            ),
            ["source_prompt", "source_package"],
        )
        self.assertEqual(
            optimizer.check_lazy_status(
                use_approved_prompt=True,
                source_prompt=None,
                source_package=None,
                approved_ref_images={
                    "ref_image_0": (None, "approved_ref_images.ref_image_0")
                },
                approved_ref_videos={},
                approved_ref_video_audios={},
                approved_ref_audios={},
            ),
            ["approved_ref_images.ref_image_0"],
        )

        direct_image = object()
        output = optimizer.execute(
            duration_seconds=5,
            optimizer_id="segment_1",
            use_approved_prompt=True,
            approved_prompt=PROMPT,
            source_prompt=None,
            source_package=None,
            approved_ref_images={"ref_image_0": direct_image},
        )
        self.assertIs(output[1]["ref_images"]["ref_image_0"], direct_image)
        self.assertEqual(output[1]["prompt"], PROMPT)

        with self.assertRaisesRegex(ValueError, "no directly connected reference"):
            optimizer.execute(
                duration_seconds=5,
                optimizer_id="segment_1",
                use_approved_prompt=True,
                approved_prompt=PROMPT,
                source_prompt=None,
                source_package=None,
            )


class RegistryTests(unittest.TestCase):
    def test_snapshots_deduplicates_and_tracks_parent(self):
        registry = plugin_module("registry")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                registry.folder_paths,
                "get_output_directory",
                return_value=directory,
            ):
                first = registry.register_generation(
                    FakeVideo(), prompt_state(), "segment_1", "movie", 1, "prompt_a", "node_1",
                    api_prompt={
                        "12": {
                            "class_type": "RandomNoise",
                            "inputs": {"noise_seed": 42},
                            "is_changed": [float("nan")],
                        }
                    },
                    workflow={
                        "nodes": [{"id": 12, "type": "RandomNoise"}],
                        "links": [],
                        "last_execution": float("inf"),
                    },
                )
                duplicate = registry.register_generation(
                    FakeVideo(), prompt_state(), "segment_1", "movie", 1, "prompt_a", "node_1"
                )
                second_prompt = PROMPT.replace("quiet room", "sunlit room")
                second = registry.register_generation(
                    FakeVideo(), prompt_state(second_prompt), "segment_1", "movie", 1,
                    "prompt_b", "node_1"
                )

                self.assertEqual(duplicate["generation_id"], first["generation_id"])
                self.assertEqual(second["parent_generation_id"], first["generation_id"])
                self.assertEqual(
                    registry.latest_generation("segment_1")["generation_id"],
                    second["generation_id"],
                )
                self.assertEqual(
                    [item["generation_id"] for item in registry.generation_history("segment_1")],
                    [second["generation_id"], first["generation_id"]],
                )
                artifact = registry.generation_artifact(first["generation_id"])
                self.assertEqual(artifact.read_bytes(), b"fake mp4 payload")
                self.assertEqual(
                    registry.generation_snapshot(first["generation_id"]),
                    {
                        "api_prompt": {
                            "12": {
                                "class_type": "RandomNoise",
                                "inputs": {"noise_seed": 42},
                                "is_changed": [None],
                            }
                        },
                        "workflow": {
                            "nodes": [{"id": 12, "type": "RandomNoise"}],
                            "links": [],
                            "last_execution": None,
                        },
                    },
                )
                self.assertNotIn("NaN", (artifact.parent / "api_prompt.json").read_text("utf-8"))
                self.assertNotIn("Infinity", (artifact.parent / "workflow.json").read_text("utf-8"))

                (artifact.parent / "api_prompt.json").write_text(
                    '{"12":{"inputs":{"noise_seed":42},"is_changed":[NaN]}}',
                    encoding="utf-8",
                )
                self.assertEqual(
                    registry.generation_snapshot(first["generation_id"])["api_prompt"]["12"][
                        "is_changed"
                    ],
                    [None],
                )
                self.assertIsNone(registry.generation_snapshot(second["generation_id"]))

                analysis_id = registry.create_analysis(
                    generation_id=second["generation_id"],
                    feedback="slow the motion",
                    time_range=None,
                    target_state_hash="a" * 64,
                    critic={"issue_type": "motion_timing"},
                    prompt_ir={"schema_version": 1},
                    patch={"action": "PATCH_PROMPT"},
                    optimized_prompt=second_prompt,
                    optimized_prompt_hash=hashlib.sha256(
                        second_prompt.encode("utf-8")
                    ).hexdigest(),
                    diff="diff",
                )
                analysis = registry.get_analysis(analysis_id)
                self.assertIsNone(analysis["time_range"])
                self.assertNotIn("time_range_json", analysis)
                self.assertEqual(analysis["proposal"]["action"], "PATCH_PROMPT")
                self.assertTrue(registry.mark_analysis_applied(analysis_id))
                self.assertFalse(registry.mark_analysis_applied(analysis_id))

    def test_successful_child_generation_marks_matching_analysis_applied(self):
        registry = plugin_module("registry")
        optimized_prompt = PROMPT.replace("quiet room", "sunlit room")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                registry.folder_paths,
                "get_output_directory",
                return_value=directory,
            ):
                generation = registry.register_generation(
                    FakeVideo(), prompt_state(), "segment_1", "movie", 1,
                    "prompt_a", "node_1"
                )
                analysis_id = registry.create_analysis(
                    generation_id=generation["generation_id"],
                    feedback="use sunlight",
                    time_range=None,
                    target_state_hash="a" * 64,
                    critic={"issue_type": "lighting"},
                    prompt_ir={"schema_version": 1},
                    patch={"action": "PATCH_PROMPT"},
                    optimized_prompt=optimized_prompt,
                    optimized_prompt_hash=hashlib.sha256(
                        optimized_prompt.encode("utf-8")
                    ).hexdigest(),
                    diff="diff",
                )
                self.assertIsNone(registry.get_analysis(analysis_id)["applied_at"])

                registry.register_generation(
                    FakeVideo(), prompt_state(optimized_prompt), "segment_1", "movie", 1,
                    "prompt_b", "node_1"
                )

                self.assertIsNotNone(registry.get_analysis(analysis_id)["applied_at"])


class SchemaTests(unittest.TestCase):
    def test_analysis_request_strips_feedback_and_validates_hash(self):
        schemas = plugin_module("schemas")
        request = schemas.AnalyzeRequest(
            generation_id=" generation_1 ",
            feedback=" fix the motion ",
            target_state_hash="a" * 64,
        )
        self.assertEqual(request.generation_id, "generation_1")
        self.assertEqual(request.feedback, "fix the motion")
        self.assertEqual(request.reasoning_effort, "none")
        self.assertEqual(
            schemas.AnalyzeRequest(
                generation_id="generation_1",
                feedback="fix",
                target_state_hash="a" * 64,
                reasoning_effort="xhigh",
            ).reasoning_effort,
            "xhigh",
        )
        with self.assertRaises(ValueError):
            schemas.AnalyzeRequest(
                generation_id="generation_1",
                feedback="fix",
                target_state_hash="z" * 64,
            )
        with self.assertRaises(ValueError):
            schemas.AnalyzeRequest(
                generation_id="generation_1",
                feedback="fix",
                target_state_hash="a" * 64,
                reasoning_effort="high",
            )

    def test_analysis_token_budget_defaults_and_manual_validation(self):
        schemas = plugin_module("schemas")
        inputs = dict(generation_id="generation_1", feedback="fix", target_state_hash="a" * 64)
        self.assertEqual(schemas.AnalyzeRequest(**inputs).resolved_max_tokens, 4096)
        for effort, expected in [("none", 4096), ("low", 8192), ("medium", 12288), ("xhigh", 16384)]:
            request = schemas.AnalyzeRequest(**inputs, reasoning_effort=effort, max_tokens=None)
            self.assertEqual(request.resolved_max_tokens, expected)
            request = schemas.AnalyzeRequest(**inputs, reasoning_effort=effort, max_tokens=23456)
            self.assertEqual(request.resolved_max_tokens, 23456)
        for invalid in [0, -1, 1.5, 4096.0, "8192", True, False]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                schemas.AnalyzeRequest(**inputs, max_tokens=invalid)

    def test_frame_analysis_passes_budget_to_global_and_focused_calls(self):
        engine = plugin_module("engine")
        schemas = plugin_module("schemas")
        critic = types.SimpleNamespace(
            issue_type="motion_timing",
            localization=schemas.Localization(start_sec=1, end_sec=2),
            model_dump=lambda: {},
        )
        for requested_range, expected_calls in [(None, 2), ({"start_sec": 1, "end_sec": 2}, 1)]:
            generate = mock.AsyncMock(return_value=critic)
            with mock.patch.object(engine, "sample_video", return_value=[]), \
                 mock.patch.object(engine, "_critic_user_prompt", return_value="review"), \
                 mock.patch.object(engine, "generate_structured", new=generate):
                asyncio.run(engine._analyze_frames(
                    {"video": {"duration_sec": 5}}, "fix", requested_range,
                    Path("/video"), "medium", 23456,
                ))
            self.assertEqual(generate.await_count, expected_calls)
            for call in generate.await_args_list:
                self.assertEqual(call.kwargs["max_tokens"], 23456)
                self.assertEqual(call.kwargs["reasoning_effort"], "medium")

    def test_vlm_parser_accepts_server_channel_prefix(self):
        from pydantic import BaseModel

        vlm = plugin_module("vlm")

        class ProbeResult(BaseModel):
            status: str

        result = vlm._decode_json(
            '<|channel>thought\nThe answer follows.\n<channel|>{"status":"ok"}',
            ProbeResult,
        )
        self.assertEqual(result.status, "ok")

    def test_vlm_parser_removes_llama_cli_reasoning_before_json_candidates(self):
        from pydantic import BaseModel

        vlm = plugin_module("vlm")

        class ProbeResult(BaseModel):
            status: str

        result = vlm._decode_json(
            "[Start thinking]\n{\"status\":\"wrong\"}\n[End thinking]\n"
            "{\"status\":\"ok\"}",
            ProbeResult,
        )
        self.assertEqual(result.status, "ok")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            vlm._decode_json(
                "[Start thinking]\n{\"status\":\"wrong\"}\n[End thinking]\nnot json",
                ProbeResult,
            )
        with self.assertRaisesRegex(ValueError, "unterminated"):
            vlm._decode_json("[Start thinking]\nprivate", ProbeResult)
        with self.assertRaisesRegex(ValueError, "empty final"):
            vlm._decode_json("[Start thinking]\nprivate[End thinking]", ProbeResult)


class VLMBackendTests(unittest.TestCase):
    def test_llama_cli_lock_is_shared_with_prompt_generator(self):
        vlm = plugin_module("vlm")
        adapter = importlib.import_module(
            "custom_nodes.ComfyUI-MiniMaxH3-Prompt-Generator.llama_cli"
        )
        self.assertIs(vlm._LLAMA_SESSION_LOCK, adapter.SESSION_LOCK)

    def test_llama_cli_reasoning_effort_command_mapping(self):
        vlm = plugin_module("vlm")
        config = vlm.LlamaCliConfig(
            Path("/cli"), Path("/model"), Path("/mmproj"), "CUDA0", "auto",
            None, None, None, 30,
        )
        for effort in ("none", "low", "medium", "xhigh"):
            command = vlm._llama_cli_command(
                config,
                Path("/system"),
                Path("/prompt"),
                Path("/output"),
                Path("/grammar"),
                [],
                512,
                effort,
            )
            self.assertEqual(command[command.index("--predict") + 1], "512")
            if effort == "none":
                self.assertEqual(command[command.index("--reasoning") + 1], "off")
                self.assertNotIn("--reasoning-effort", command)
            else:
                self.assertEqual(
                    command[command.index("--reasoning-effort") + 1], effort
                )
                self.assertNotIn("--reasoning", command)

    def test_openai_request_includes_reasoning_effort(self):
        vlm = plugin_module("vlm")
        captured = {}
        response_choice = {"message": {"content": "{}"}}

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def text(self):
                return json.dumps({"choices": [response_choice]})

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def post(self, url, **kwargs):
                captured["url"] = url
                captured.update(kwargs)
                return FakeResponse()

        with mock.patch.object(vlm.aiohttp, "ClientSession", return_value=FakeSession()), \
             mock.patch.object(vlm, "chat_completions_url", return_value="http://llm/v1/chat/completions"):
            response = asyncio.run(
                vlm._request_openai([{"role": "user", "content": "review"}], 512, "medium")
            )
            for content in (None, "", "{}"):
                response_choice.update(finish_reason="length", message={"content": content})
                with self.assertRaisesRegex(ValueError, "max_tokens=512.*reasoning_effort=medium"):
                    asyncio.run(vlm._request_openai([], 512, "medium"))
            response_choice.update(finish_reason="stop", message={"content": ""})
            with self.assertRaisesRegex(ValueError, "returned no content"):
                asyncio.run(vlm._request_openai([], 512, "medium"))

        self.assertEqual(response.content, "{}")
        self.assertEqual(captured["url"], "http://llm/v1/chat/completions")
        self.assertEqual(captured["json"]["reasoning_effort"], "medium")
        self.assertEqual(captured["json"]["max_tokens"], 512)

    def test_legacy_llama_cli_settings_are_loaded_from_plugin_env(self):
        vlm = plugin_module("vlm")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = root / "llama-cli"
            model = root / "model.gguf"
            mmproj = root / "mmproj.gguf"
            cli.write_text("#!/bin/sh\n", encoding="utf-8")
            cli.chmod(0o700)
            model.write_bytes(b"model")
            mmproj.write_bytes(b"mmproj")
            (root / ".env").write_text(
                "\n".join((
                    "H3_OPTIMIZER_VLM_BACKEND=llama_cli",
                    "H3_OPTIMIZER_LLAMA_CLI_PATH=%s" % cli,
                    "H3_OPTIMIZER_LLAMA_MODEL_PATH=%s" % model,
                    "H3_OPTIMIZER_LLAMA_MMPROJ_PATH=%s" % mmproj,
                    "H3_OPTIMIZER_LLAMA_CTX_SIZE=32768",
                )),
                encoding="utf-8",
            )
            names = {
                "H3_OPTIMIZER_VLM_BACKEND",
                "H3_OPTIMIZER_LLAMA_CLI_PATH",
                "H3_OPTIMIZER_LLAMA_MODEL_PATH",
                "H3_OPTIMIZER_LLAMA_MMPROJ_PATH",
                "H3_OPTIMIZER_LLAMA_CTX_SIZE",
            }
            environment = {key: value for key, value in vlm.os.environ.items() if key not in names}
            with mock.patch.object(vlm, "PLUGIN_DIRECTORY", root), mock.patch.dict(
                vlm.os.environ, environment, clear=True
            ):
                config = vlm._llama_cli_config()
                self.assertEqual(vlm.vlm_backend(), "llama_cli")

            self.assertEqual(config.cli_path, cli)
            self.assertEqual(config.model_path, model)
            self.assertEqual(config.mmproj_path, mmproj)
            self.assertEqual(config.ctx_size, 32768)

    def test_shared_llama_cli_settings_are_loaded_from_generator_env(self):
        vlm = plugin_module("vlm")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer_root = root / "optimizer"
            generator_root = root / "generator"
            optimizer_root.mkdir()
            generator_root.mkdir()
            cli = root / "llama-cli"
            model = root / "model.gguf"
            mmproj = root / "mmproj.gguf"
            cli.write_text("#!/bin/sh\n", encoding="utf-8")
            cli.chmod(0o700)
            model.write_bytes(b"model")
            mmproj.write_bytes(b"mmproj")
            generator_env = generator_root / ".env"
            generator_env.write_text(
                "\n".join((
                    "MINIMAX_VLM_BACKEND=llama_cli",
                    "MINIMAX_LLAMA_CLI_PATH=%s" % cli,
                    "MINIMAX_LLAMA_MODEL_PATH=%s" % model,
                    "MINIMAX_LLAMA_MMPROJ_PATH=%s" % mmproj,
                    "MINIMAX_LLAMA_CTX_SIZE=24576",
                )),
                encoding="utf-8",
            )

            with mock.patch.object(vlm, "PLUGIN_DIRECTORY", optimizer_root), mock.patch.object(
                vlm, "PROMPT_GENERATOR_ENV", generator_env
            ), mock.patch.dict(vlm.os.environ, {}, clear=True):
                config = vlm._llama_cli_config()
                self.assertEqual(vlm.vlm_backend(), "llama_cli")

            self.assertEqual(config.cli_path, cli)
            self.assertEqual(config.model_path, model)
            self.assertEqual(config.mmproj_path, mmproj)
            self.assertEqual(config.ctx_size, 24576)

    def test_shared_setting_priority_does_not_overwrite_env_files(self):
        vlm = plugin_module("vlm")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer_root = root / "optimizer"
            generator_root = root / "generator"
            optimizer_root.mkdir()
            generator_root.mkdir()
            optimizer_env = optimizer_root / ".env"
            generator_env = generator_root / ".env"
            optimizer_text = "MINIMAX_LLAMA_MODEL_PATH=/optimizer/model.gguf\n"
            generator_text = "MINIMAX_LLAMA_MODEL_PATH=/generator/model.gguf\n"
            optimizer_env.write_text(optimizer_text, encoding="utf-8")
            generator_env.write_text(generator_text, encoding="utf-8")

            with mock.patch.object(vlm, "PLUGIN_DIRECTORY", optimizer_root), mock.patch.object(
                vlm, "PROMPT_GENERATOR_ENV", generator_env
            ), mock.patch.dict(vlm.os.environ, {}, clear=True):
                self.assertEqual(
                    vlm.optimizer_setting("LLAMA_MODEL_PATH"),
                    "/optimizer/model.gguf",
                )
                vlm.os.environ["MINIMAX_LLAMA_MODEL_PATH"] = "/process/shared.gguf"
                self.assertEqual(
                    vlm.optimizer_setting("LLAMA_MODEL_PATH"),
                    "/process/shared.gguf",
                )
                vlm.os.environ["H3_OPTIMIZER_LLAMA_MODEL_PATH"] = "/process/optimizer.gguf"
                self.assertEqual(
                    vlm.optimizer_setting("LLAMA_MODEL_PATH"),
                    "/process/optimizer.gguf",
                )

            self.assertEqual(optimizer_env.read_text(encoding="utf-8"), optimizer_text)
            self.assertEqual(generator_env.read_text(encoding="utf-8"), generator_text)

    def test_llama_cli_request_uses_json_grammar_images_and_temporary_files(self):
        vlm = plugin_module("vlm")

        config = vlm.LlamaCliConfig(
            cli_path=Path("/opt/llama/llama-cli"),
            model_path=Path("/models/vlm.gguf"),
            mmproj_path=Path("/models/mmproj.gguf"),
            device="CUDA0",
            gpu_layers="auto",
            ctx_size=32768,
            fit_target_mib=1024,
            image_max_tokens=768,
            timeout_seconds=30,
        )
        jpeg = b"fake jpeg bytes"
        messages = [{
            "role": "system",
            "content": "Return JSON.",
        }, {
            "role": "user",
            "content": [
                {"type": "text", "text": "Review the frame."},
                {"type": "text", "text": "Frame timestamp: 1.250 seconds"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
                    },
                },
            ],
        }]
        captured = {}

        class FakeProcess:
            pid = 12345
            returncode = None

            async def communicate(self):
                self.returncode = 0
                return b"console output", b"llama diagnostic"

            async def wait(self):
                self.returncode = 0
                return 0

        async def start_process(*command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            system_path = Path(command[command.index("--system-prompt-file") + 1])
            prompt_path = Path(command[command.index("--file") + 1])
            output_path = Path(command[command.index("--output-file") + 1])
            grammar_path = Path(command[command.index("--grammar-file") + 1])
            image_path = Path(command[command.index("--image") + 1])
            captured["temporary_directory"] = prompt_path.parent
            captured["system"] = system_path.read_text(encoding="utf-8")
            captured["prompt"] = prompt_path.read_text(encoding="utf-8")
            output_path.write_text(
                'prompt transcript\n\nAssistant:\n{"status":"ok"}', encoding="utf-8"
            )
            captured["grammar"] = grammar_path.read_text(encoding="utf-8")
            captured["image"] = image_path.read_bytes()
            return FakeProcess()

        with mock.patch.object(
            vlm.asyncio, "create_subprocess_exec", new=start_process
        ), mock.patch.object(vlm, "_comfy_queue_busy", return_value=False):
            output = asyncio.run(
                vlm._request_llama_cli(messages, 512, config, "none")
            )

        self.assertEqual(output.content, '{"status":"ok"}')
        self.assertEqual(output.diagnostics, "llama diagnostic")
        self.assertEqual(captured["system"], "Return JSON.")
        self.assertIn("Image 1: 1.250 seconds", captured["prompt"])
        self.assertIn("root ::= object", captured["grammar"])
        self.assertEqual(captured["image"], jpeg)
        self.assertIn("--single-turn", captured["command"])
        self.assertIn("--offline", captured["command"])
        self.assertIn("--output-file", captured["command"])
        self.assertIn("--grammar-file", captured["command"])
        self.assertNotIn("--json-schema-file", captured["command"])
        self.assertIn("--mmproj", captured["command"])
        self.assertTrue(captured["kwargs"]["start_new_session"])
        self.assertFalse(captured["temporary_directory"].exists())

    def test_invalid_structured_responses_log_content_and_backend_diagnostics(self):
        from pydantic import BaseModel

        vlm = plugin_module("vlm")

        class ProbeResult(BaseModel):
            status: str

        responses = [
            vlm.VLMResponse("initial non-JSON output", "initial llama stderr"),
            vlm.VLMResponse("repair non-JSON output", "repair llama stderr"),
        ]
        with mock.patch.object(vlm, "_request", side_effect=responses), self.assertLogs(
            vlm.log, level="WARNING"
        ) as captured:
            with self.assertRaisesRegex(ValueError, "VLM response is not valid JSON"):
                asyncio.run(vlm.generate_structured("system", "user", [], ProbeResult))

        messages = "\n".join(captured.output)
        self.assertIn("ProbeResult response is invalid during initial generation", messages)
        self.assertIn("initial non-JSON output", messages)
        self.assertIn("initial llama stderr", messages)
        self.assertIn("ProbeResult response is invalid during repair generation", messages)
        self.assertIn("repair non-JSON output", messages)
        self.assertIn("repair llama stderr", messages)

    def test_generate_structured_propagates_reasoning_effort_to_repair(self):
        from pydantic import BaseModel

        vlm = plugin_module("vlm")

        class ProbeResult(BaseModel):
            status: str

        request = mock.AsyncMock(side_effect=[
            vlm.VLMResponse("not json"),
            vlm.VLMResponse('{"status":"ok"}'),
        ])
        with mock.patch.object(vlm, "_request", new=request):
            result = asyncio.run(
                vlm.generate_structured(
                    "system", "user", [], ProbeResult, max_tokens=23456, reasoning_effort="xhigh"
                )
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(request.await_count, 2)
        self.assertEqual([call.args[1] for call in request.await_args_list], [23456, 23456])
        self.assertEqual([call.args[2] for call in request.await_args_list], ["xhigh", "xhigh"])

    def test_llama_cli_empty_response_logs_context_and_timings(self):
        vlm = plugin_module("vlm")

        config = vlm.LlamaCliConfig(
            Path("/cli"), Path("/model"), Path("/mmproj"), "CUDA0", "auto",
            65536, None, None, 30,
        )
        captured = {}

        class FakeProcess:
            pid = 12345
            returncode = None

            async def communicate(self):
                self.returncode = 0
                return b"", b"[ Prompt: 12.0 t/s | Generation: 0.0 t/s ]"

        async def start_process(*command, **_kwargs):
            captured["command"] = command
            output_path = Path(command[command.index("--output-file") + 1])
            output_path.write_text("User:\nrequest\n\nAssistant:\n\n", encoding="utf-8")
            return FakeProcess()

        with mock.patch.object(
            vlm.asyncio, "create_subprocess_exec", new=start_process
        ), mock.patch.object(vlm, "_comfy_queue_busy", return_value=False), self.assertLogs(
            vlm.log, level="ERROR"
        ) as logs:
            with self.assertRaisesRegex(vlm.VLMProcessError, "no assistant content"):
                asyncio.run(vlm._request_llama_cli(
                    [{"role": "user", "content": "request"}], 4096, config, "none"
                ))

        message = "\n".join(logs.output)
        self.assertIn("images=0, max_tokens=4096, ctx_size=65536", message)
        self.assertIn("Generation: 0.0 t/s", message)
        self.assertNotIn("--no-show-timings", captured["command"])

    def test_local_inference_session_unloads_comfy_models_once(self):
        vlm = plugin_module("vlm")
        config = vlm.LlamaCliConfig(
            Path("/cli"), Path("/model"), Path("/mmproj"), "CUDA0", "auto",
            None, None, None, 300,
        )

        async def enter_session():
            async with vlm.inference_session():
                self.assertEqual(vlm._ACTIVE_BACKEND.get(), "llama_cli")
                self.assertIs(vlm._ACTIVE_LLAMA_CONFIG.get(), config)

        with mock.patch.object(
            vlm, "vlm_backend", return_value="llama_cli"
        ), mock.patch.object(
            vlm, "_llama_cli_config", return_value=config
        ), mock.patch.object(
            vlm, "_comfy_queue_busy", return_value=False
        ), mock.patch.object(
            vlm.comfy.model_management, "unload_all_models"
        ) as unload, mock.patch.object(
            vlm.comfy.model_management, "soft_empty_cache"
        ) as empty_cache:
            asyncio.run(enter_session())

        unload.assert_called_once_with()
        empty_cache.assert_called_once_with()
        self.assertIsNone(vlm._ACTIVE_BACKEND.get())
        self.assertIsNone(vlm._ACTIVE_LLAMA_CONFIG.get())

    def test_cancelling_llama_cli_request_stops_the_process(self):
        vlm = plugin_module("vlm")

        with tempfile.TemporaryDirectory() as directory:
            cli = Path(directory) / "llama-cli"
            cli.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
            cli.chmod(0o700)
            config = vlm.LlamaCliConfig(
                cli, Path("/model"), Path("/mmproj"), "CUDA0", "auto",
                None, None, None, 300,
            )
            started = asyncio.Event()
            process = None
            create_subprocess = asyncio.create_subprocess_exec

            async def start_process(*command, **kwargs):
                nonlocal process
                process = await create_subprocess(*command, **kwargs)
                started.set()
                return process

            async def cancel_request():
                task = asyncio.create_task(
                    vlm._request_llama_cli(
                        [{"role": "user", "content": "wait"}], 16, config, "none"
                    )
                )
                await started.wait()
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            with mock.patch.object(
                vlm.asyncio, "create_subprocess_exec", new=start_process
            ), mock.patch.object(vlm, "_comfy_queue_busy", return_value=False):
                asyncio.run(cancel_request())

            self.assertIsNotNone(process.returncode)


class EngineTests(unittest.TestCase):
    def test_analysis_rejects_stored_prompt_with_unconnected_reference(self):
        engine = plugin_module("engine")
        registry = plugin_module("registry")
        schemas = plugin_module("schemas")
        invalid_prompt = PROMPT.replace(
            "<Subject 1> is the woman defined by <Picture 1>.",
            "<Subject 1> is the woman defined by <Picture 1>.\n"
            "<Video 1> is not provided; no video references apply.",
        )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                registry.folder_paths,
                "get_output_directory",
                return_value=directory,
            ):
                generation = registry.register_generation(
                    FakeVideo(), prompt_state(invalid_prompt), "segment_1", "movie", 1,
                    "prompt_a", "node_1"
                )
                request = schemas.AnalyzeRequest(
                    generation_id=generation["generation_id"],
                    feedback="Review the motion.",
                    target_state_hash="a" * 64,
                )
                with self.assertRaisesRegex(ValueError, "no directly connected reference"):
                    asyncio.run(engine.analyze_generation(request))

    def test_capability_classification_still_allows_prompt_correction(self):
        engine = plugin_module("engine")
        registry = plugin_module("registry")
        schemas = plugin_module("schemas")
        critic = schemas.CriticResult(
            issue_type="object_consistency",
            issue_confirmed=True,
            confidence=1.0,
            localization={"start_sec": 0.0, "end_sec": 5.0},
            observations=[{
                "text": "The loose pom-pom disappears when the subject stands.",
                "start_sec": 2.8,
                "end_sec": 3.1,
            }],
            inferences=["The prompt does not preserve the loose object."],
            root_cause="MODEL_CAPABILITY_LIMIT",
            reason="The generated object is not persistent.",
        )
        definitions = (
            "<Subject 1> is the woman defined by <Picture 1>, wearing socks and no boots.\n"
            "<Subject 2> is a loose white pom-pom resting on the bed."
        )
        retention = (
            "<Picture 1> defines her identity and clothing except footwear.\n"
            "<Subject 2> remains stationary on the bed."
        )
        shot = (
            "A static medium shot shows <Subject 1> standing up in socks with no shoes or "
            "boots. <Subject 2> remains visible in the same place on the bed."
        )
        patch = schemas.PatchPlan(
            action="PATCH_PROMPT",
            reason="Make the requested object and footwear states explicit.",
            operations=[
                {"op": "replace", "path": "/sections/subject_definitions", "value": definitions},
                {"op": "replace", "path": "/sections/retention_analysis", "value": retention},
                {
                    "op": "replace",
                    "path": "/sections/detailed_description/shots/1/body",
                    "value": shot,
                },
            ],
            changes=["Keep the pom-pom on the bed and replace boots with socks."],
            preserved=["Identity, scene, sound, and reference labels."],
        )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                registry.folder_paths,
                "get_output_directory",
                return_value=directory,
            ), mock.patch.object(
                engine,
                "_analyze_frames",
                new=mock.AsyncMock(return_value=critic),
            ) as analyze_frames, mock.patch.object(
                engine,
                "generate_structured",
                new=mock.AsyncMock(return_value=patch),
            ) as planner:
                generation = registry.register_generation(
                    FakeVideo(), prompt_state(), "segment_1", "movie", 1, "prompt_a", "node_1"
                )
                request = schemas.AnalyzeRequest(
                    generation_id=generation["generation_id"],
                    feedback="Keep the pom-pom on the bed and show socks, not boots.",
                    target_state_hash="a" * 64,
                    reasoning_effort="xhigh",
                )
                result = asyncio.run(engine.analyze_generation(request))

        self.assertEqual(planner.await_count, 1)
        self.assertEqual(analyze_frames.await_args.kwargs["max_tokens"], 16384)
        self.assertEqual(planner.await_args.kwargs["max_tokens"], 16384)
        self.assertEqual(
            analyze_frames.await_args.kwargs["reasoning_effort"], "xhigh"
        )
        self.assertEqual(
            planner.await_args.kwargs["reasoning_effort"], "xhigh"
        )
        planner_input = json.loads(planner.await_args.args[1])
        self.assertEqual(planner_input["critic"]["root_cause"], "MODEL_CAPABILITY_LIMIT")
        self.assertIn("/sections/subject_definitions", planner_input["allowed_paths"])
        self.assertEqual(result["proposal"]["action"], "PATCH_PROMPT")
        self.assertIn("wearing socks and no boots", result["optimized_prompt"])
        self.assertIn("remains visible in the same place", result["optimized_prompt"])
        self.assertIn("/sections/subject_definitions", result["allowed_paths"])

    def test_analysis_applies_only_the_localized_shot_patch(self):
        engine = plugin_module("engine")
        registry = plugin_module("registry")
        schemas = plugin_module("schemas")
        critic = schemas.CriticResult(
            issue_type="motion_timing",
            issue_confirmed=True,
            confidence=0.9,
            localization={"start_sec": 2.1, "end_sec": 2.8},
            observations=[{
                "text": "The walk completes abruptly.",
                "start_sec": 2.1,
                "end_sec": 2.8,
            }],
            inferences=["The requested slow timing is not visible."],
            root_cause="PROMPT_TIMING_INSUFFICIENT",
            reason="The second shot has no explicit duration.",
        )
        patch = schemas.PatchPlan(
            action="PATCH_PROMPT",
            reason="Make only the second shot timing explicit.",
            operations=[{
                "op": "replace",
                "path": "/sections/detailed_description/shots/1/body",
                "value": "She walks toward the window over three seconds in one smooth motion.",
            }],
            changes=["Second shot walking duration is explicit."],
            preserved=["Subject, identity, first shot, sound and music."],
        )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                registry.folder_paths,
                "get_output_directory",
                return_value=directory,
            ), mock.patch.object(
                engine,
                "_analyze_frames",
                new=mock.AsyncMock(return_value=critic),
            ), mock.patch.object(
                engine,
                "generate_structured",
                new=mock.AsyncMock(return_value=patch),
            ):
                generation = registry.register_generation(
                    FakeVideo(), prompt_state(), "segment_1", "movie", 1, "prompt_a", "node_1"
                )
                request = schemas.AnalyzeRequest(
                    generation_id=generation["generation_id"],
                    feedback="The walking motion is too fast.",
                    target_state_hash="a" * 64,
                    reasoning_effort="medium",
                    max_tokens=23456,
                )
                result = asyncio.run(engine.analyze_generation(request))

                self.assertEqual(engine._analyze_frames.await_args.kwargs["max_tokens"], 23456)
                self.assertEqual(engine.generate_structured.await_args.kwargs["max_tokens"], 23456)
                self.assertEqual(result["proposal"]["action"], "PATCH_PROMPT")
                self.assertIn("over three seconds", result["optimized_prompt"])
                self.assertIn("<Picture 1> defines her identity", result["optimized_prompt"])
                self.assertIn("Soft footsteps", result["optimized_prompt"])
                self.assertEqual(
                    result["allowed_paths"],
                    ["/sections/detailed_description/shots/1/body"],
                )
                stored = registry.get_analysis(result["analysis_id"])
                self.assertEqual(stored["optimized_prompt"], result["optimized_prompt"])


class PackageLoadTests(unittest.TestCase):
    def test_extension_entrypoint_and_routes_load(self):
        from aiohttp import web
        from server import PromptServer

        sentinel = object()
        previous = getattr(PromptServer, "instance", sentinel)
        PromptServer.instance = types.SimpleNamespace(routes=web.RouteTableDef())
        module_name = "h3_prompt_optimizer_package_probe"
        spec = importlib.util.spec_from_file_location(
            module_name,
            PLUGIN_ROOT / "__init__.py",
            submodule_search_locations=[str(PLUGIN_ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            self.assertEqual(
                set(module.NODE_CLASS_MAPPINGS),
                {
                    "H3OptimizerSegmentSettingsCS",
                    "H3OptimizerRef2VAPromptPackageGeneratorCS",
                    "H3OptimizerRef2VAPromptPackageCS",
                    "H3OptimizerVideoOutputCS",
                },
            )
            self.assertEqual(len(PromptServer.instance.routes), 10)

            route_module = sys.modules[module_name + ".routes"]
            with mock.patch.object(route_module, "backend_status", return_value={}):
                status = asyncio.run(route_module.optimizer_status(None))
            self.assertEqual(json.loads(status.text)["analysis_max_tokens"], {
                "none": 4096, "low": 8192, "medium": 12288, "xhigh": 16384,
            })

            async def check_cancel():
                started = asyncio.Event()

                async def wait_for_cancel(_payload, cancel_event):
                    started.set()
                    await asyncio.Event().wait()

                analyze_request = types.SimpleNamespace(
                    headers={"X-H3-Analysis-Request-ID": "request_1"},
                    json=mock.AsyncMock(return_value={
                        "generation_id": "generation_1",
                        "feedback": "review motion",
                        "target_state_hash": "a" * 64,
                    }),
                )
                cancel_request = types.SimpleNamespace(
                    match_info={"request_id": "request_1"}
                )
                with mock.patch.object(
                    route_module,
                    "analyze_generation",
                    side_effect=wait_for_cancel,
                ):
                    analysis_task = asyncio.create_task(
                        route_module.optimizer_analyze(analyze_request)
                    )
                    await started.wait()
                    cancel_response = await route_module.optimizer_cancel_analysis(
                        cancel_request
                    )
                    analysis_response = await analysis_task

                self.assertEqual(cancel_response.status, 200)
                self.assertEqual(analysis_response.status, 409)
                self.assertNotIn("request_1", route_module._ANALYSIS_TASKS)

            asyncio.run(check_cancel())
        finally:
            for name in list(sys.modules):
                if name == module_name or name.startswith(module_name + "."):
                    del sys.modules[name]
            if previous is sentinel:
                del PromptServer.instance
            else:
                PromptServer.instance = previous


if __name__ == "__main__":
    unittest.main(verbosity=2)
