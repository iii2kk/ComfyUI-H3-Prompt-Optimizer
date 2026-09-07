import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import comfy.model_management
from pydantic import ValidationError
from server import PromptServer


DEFAULT_BASE_URL = "http://localhost:1234"
OPENAI_COMPATIBLE_BACKEND = "openai_compatible"
LLAMA_CLI_BACKEND = "llama_cli"
PLUGIN_DIRECTORY = Path(__file__).resolve().parent
PROMPT_GENERATOR_ENV = PLUGIN_DIRECTORY.parent / "ComfyUI-MiniMaxH3-Prompt-Generator" / ".env"
_ACTIVE_BACKEND = ContextVar("h3_optimizer_vlm_backend", default=None)
_ACTIVE_LLAMA_CONFIG = ContextVar("h3_optimizer_llama_config", default=None)
_LLAMA_SESSION_LOCK = import_module(
    "custom_nodes.ComfyUI-MiniMaxH3-Prompt-Generator.llama_cli"
).SESSION_LOCK
log = logging.getLogger(__name__)
_DIAGNOSTIC_LIMIT = 1600
JSON_OBJECT_GRAMMAR = r'''root ::= object
value ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
'''


class VLMConfigurationError(ValueError):
    pass


class VLMBackendBusyError(RuntimeError):
    pass


class VLMProcessError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlamaCliConfig:
    cli_path: Path
    model_path: Path
    mmproj_path: Path
    device: str
    gpu_layers: str
    ctx_size: int | None
    fit_target_mib: int | None
    image_max_tokens: int | None
    timeout_seconds: int


@dataclass(frozen=True)
class VLMResponse:
    content: str
    diagnostics: str = ""


def _env_file_value(path, names):
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    for name in names:
        value = values.get(name)
        if value not in (None, ""):
            return value
    return None


def optimizer_setting(suffix, fallback=None):
    """Read an Optimizer override or the shared Prompt Generator setting."""
    optimizer_name = "H3_OPTIMIZER_" + suffix
    shared_name = "MINIMAX_" + suffix
    names = (optimizer_name, shared_name)

    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value

    value = _env_file_value(PLUGIN_DIRECTORY / ".env", names)
    if value not in (None, ""):
        return value

    value = _env_file_value(PROMPT_GENERATOR_ENV, (shared_name,))
    return value if value not in (None, "") else fallback


def optimizer_only_setting(suffix, fallback=None):
    name = "H3_OPTIMIZER_" + suffix
    value = os.getenv(name)
    if value in (None, ""):
        value = _env_file_value(PLUGIN_DIRECTORY / ".env", (name,))
    return value if value not in (None, "") else fallback


def vlm_backend():
    backend = optimizer_setting("VLM_BACKEND", OPENAI_COMPATIBLE_BACKEND).strip().lower()
    if backend not in (OPENAI_COMPATIBLE_BACKEND, LLAMA_CLI_BACKEND):
        raise VLMConfigurationError(
            "MINIMAX_VLM_BACKEND must be openai_compatible or llama_cli."
        )
    return backend


def _positive_int_setting(name, fallback=None, shared=True):
    value = optimizer_setting(name) if shared else optimizer_only_setting(name)
    if value is None:
        return fallback
    setting_name = ("MINIMAX_" if shared else "H3_OPTIMIZER_") + name
    try:
        parsed = int(value)
    except ValueError as error:
        raise VLMConfigurationError("%s must be a positive integer." % setting_name) from error
    if parsed <= 0:
        raise VLMConfigurationError("%s must be a positive integer." % setting_name)
    return parsed


def _absolute_file_setting(name, executable=False):
    value = optimizer_setting(name)
    if value is None:
        raise VLMConfigurationError(
            "MINIMAX_%s is required for the llama_cli backend." % name
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise VLMConfigurationError("MINIMAX_%s must be an absolute path." % name)
    if not path.is_file():
        raise VLMConfigurationError("MINIMAX_%s does not point to a file." % name)
    if executable and not os.access(path, os.X_OK):
        raise VLMConfigurationError("MINIMAX_%s is not executable." % name)
    return path


def _llama_cli_config():
    gpu_layers = optimizer_setting("LLAMA_GPU_LAYERS", "auto").strip().lower()
    if gpu_layers not in ("auto", "all"):
        try:
            if int(gpu_layers) < 0:
                raise ValueError
        except ValueError as error:
            raise VLMConfigurationError(
                "MINIMAX_LLAMA_GPU_LAYERS must be auto, all, or a non-negative integer."
            ) from error
    device = optimizer_setting("LLAMA_DEVICE", "CUDA0").strip()
    if not device:
        raise VLMConfigurationError("MINIMAX_LLAMA_DEVICE must not be empty.")
    return LlamaCliConfig(
        cli_path=_absolute_file_setting("LLAMA_CLI_PATH", executable=True),
        model_path=_absolute_file_setting("LLAMA_MODEL_PATH"),
        mmproj_path=_absolute_file_setting("LLAMA_MMPROJ_PATH"),
        device=device,
        gpu_layers=gpu_layers,
        ctx_size=_positive_int_setting("LLAMA_CTX_SIZE"),
        fit_target_mib=_positive_int_setting("LLAMA_FIT_TARGET_MIB"),
        image_max_tokens=_positive_int_setting("LLAMA_IMAGE_MAX_TOKENS"),
        timeout_seconds=_positive_int_setting("TIMEOUT_SECONDS", 300, shared=False),
    )


def backend_status():
    backend = vlm_backend()
    if backend == OPENAI_COMPATIBLE_BACKEND:
        return {
            "critic_backend": backend,
            "critic_endpoint": chat_completions_url(),
            "critic_model": optimizer_only_setting("MODEL", "server default"),
        }
    config = _llama_cli_config()
    return {
        "critic_backend": backend,
        "critic_endpoint": config.cli_path.name,
        "critic_model": config.model_path.name,
    }


def _comfy_queue_busy():
    server = getattr(PromptServer, "instance", None)
    queue = getattr(server, "prompt_queue", None)
    return queue is not None and queue.get_tasks_remaining() > 0


def _prepare_local_inference():
    if _comfy_queue_busy():
        raise VLMBackendBusyError(
            "Local llama-cli analysis requires the ComfyUI queue to be empty."
        )
    try:
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
    except Exception as error:
        raise VLMProcessError("Could not release ComfyUI models before llama-cli inference.") from error
    if _comfy_queue_busy():
        raise VLMBackendBusyError(
            "A ComfyUI job started while preparing local llama-cli analysis."
        )


@asynccontextmanager
async def inference_session():
    if _ACTIVE_BACKEND.get() is not None:
        yield
        return

    backend = vlm_backend()
    backend_token = _ACTIVE_BACKEND.set(backend)
    config_token = None
    lock_acquired = False
    try:
        if backend == LLAMA_CLI_BACKEND:
            config = _llama_cli_config()
            lock_acquired = _LLAMA_SESSION_LOCK.acquire(blocking=False)
            if not lock_acquired:
                raise VLMBackendBusyError("Another local llama-cli analysis is already running.")
            _prepare_local_inference()
            config_token = _ACTIVE_LLAMA_CONFIG.set(config)
        yield
    finally:
        if config_token is not None:
            _ACTIVE_LLAMA_CONFIG.reset(config_token)
        _ACTIVE_BACKEND.reset(backend_token)
        if lock_acquired:
            _LLAMA_SESSION_LOCK.release()


def chat_completions_url():
    base_url = optimizer_setting("BASE_URL", DEFAULT_BASE_URL)
    base_url = base_url.strip()
    if "://" not in base_url:
        base_url = "http://" + base_url
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise VLMConfigurationError(
            "MINIMAX_BASE_URL must be an HTTP or HTTPS URL."
        )
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        final_path = path
    elif path.endswith("/v1"):
        final_path = path + "/chat/completions"
    else:
        final_path = path + "/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, final_path, "", ""))


def _strip_response(text):
    value = (text or "").strip().lstrip("\ufeff").strip()
    wrappers = (
        ("<think>", "</think>"),
        ("<|channel>thought", "<channel|>"),
        ("[Start thinking]", "[End thinking]"),
    )
    matched_wrapper = False
    for opener, closer in wrappers:
        if not value.startswith(opener):
            continue
        matched_wrapper = True
        end = value.find(closer, len(opener))
        if end < 0:
            raise ValueError("VLM response has an unterminated reasoning block.")
        value = value[end + len(closer):].strip()
    if not matched_wrapper and "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    if not value:
        raise ValueError("VLM response has an empty final response.")
    return value


def _decode_json(text, schema):
    value = _strip_response(text)
    try:
        data = json.loads(value)
    except json.JSONDecodeError as error:
        decoder = json.JSONDecoder()
        candidates = []
        for index, character in enumerate(value):
            if character != "{":
                continue
            try:
                candidate, _end = decoder.raw_decode(value, index)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                candidates.append(candidate)
        if not candidates:
            raise ValueError("VLM response is not valid JSON: %s" % error) from error
        validation_error = None
        for candidate in reversed(candidates):
            try:
                return schema.model_validate(candidate)
            except ValidationError as candidate_error:
                validation_error = candidate_error
        raise ValueError(
            "VLM response contains JSON, but it does not match the required schema: %s"
            % validation_error
        ) from validation_error
    try:
        return schema.model_validate(data)
    except ValidationError as error:
        raise ValueError("VLM response does not match the required schema: %s" % error) from error


def _message_content(user_text, images):
    if not images:
        return user_text
    content = [{"type": "text", "text": user_text}]
    for frame in images:
        content.append({
            "type": "text",
            "text": "Frame timestamp: %.3f seconds" % frame["timestamp"],
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + frame["jpeg_base64"]},
        })
    return content


async def _request(messages, max_tokens, reasoning_effort):
    backend = _ACTIVE_BACKEND.get()
    if backend is None:
        async with inference_session():
            return await _request(messages, max_tokens, reasoning_effort)
    if backend == LLAMA_CLI_BACKEND:
        config = _ACTIVE_LLAMA_CONFIG.get()
        if config is None:
            raise VLMConfigurationError("The llama-cli inference session is not initialized.")
        return await _request_llama_cli(messages, max_tokens, config, reasoning_effort)
    return await _request_openai(messages, max_tokens, reasoning_effort)


async def _request_openai(messages, max_tokens, reasoning_effort):
    payload = {
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": int(max_tokens),
        "stream": False,
        "reasoning_effort": reasoning_effort,
    }
    model = optimizer_only_setting("MODEL")
    if model:
        payload["model"] = model
    timeout_seconds = _positive_int_setting("TIMEOUT_SECONDS", 300, shared=False)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {}
    api_key = optimizer_only_setting("API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(chat_completions_url(), json=payload, headers=headers) as response:
                body = await response.text()
                if response.status != 200:
                    raise ValueError("H3 Prompt Optimizer VLM HTTP %d: %s" % (response.status, body[:800]))
                try:
                    data = json.loads(body)
                    choice = data["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    content = choice["message"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as error:
                    raise ValueError("H3 Prompt Optimizer VLM returned an unexpected response shape.") from error
                if finish_reason == "length":
                    raise ValueError(
                        "VLM reached max_tokens=%d (reasoning_effort=%s). "
                        "Increase Max tokens or lower Reasoning effort."
                        % (max_tokens, reasoning_effort)
                    )
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("H3 Prompt Optimizer VLM returned no content.")
                return VLMResponse(content)
    except aiohttp.InvalidURL as error:
        raise ValueError("H3 Prompt Optimizer VLM URL is invalid.") from error
    except aiohttp.ClientConnectionError as error:
        raise ValueError("Could not connect to the H3 Prompt Optimizer VLM at %s." % chat_completions_url()) from error
    except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as error:
        raise ValueError("H3 Prompt Optimizer VLM did not answer within %d seconds." % timeout_seconds) from error


def _decode_image_url(value):
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str):
        raise VLMConfigurationError("llama-cli received an invalid image URL.")
    header, separator, encoded = value.partition(",")
    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise VLMConfigurationError("llama-cli only accepts embedded base64 images.")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise VLMConfigurationError("llama-cli received invalid base64 image data.") from error


def _llama_cli_messages(messages):
    system_parts = []
    transcript = []
    images = []
    timestamps = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        text_parts = []
        last_timestamp = None
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text = part.get("text", "")
                    text_parts.append(text)
                    if text.startswith("Frame timestamp:"):
                        last_timestamp = text.removeprefix("Frame timestamp:").strip()
                elif part.get("type") == "image_url":
                    images.append(_decode_image_url(part.get("image_url")))
                    timestamps.append(last_timestamp)
                    last_timestamp = None
        else:
            raise VLMConfigurationError("llama-cli received unsupported message content.")

        text = "\n".join(item for item in text_parts if item)
        if role == "system":
            if text:
                system_parts.append(text)
        elif text:
            transcript.append("%s:\n%s" % (role.capitalize(), text))

    if images:
        image_order = ["Attached images are ordered as follows:"]
        for index, timestamp in enumerate(timestamps, 1):
            label = timestamp if timestamp else "timestamp not provided"
            image_order.append("Image %d: %s" % (index, label))
        transcript.insert(0, "\n".join(image_order))
    return "\n\n".join(system_parts), "\n\n".join(transcript), images


def _llama_cli_command(config, system_path, prompt_path, output_path, grammar_path,
                       image_paths, max_tokens, reasoning_effort):
    command = [
        str(config.cli_path),
        "--offline",
        "--model", str(config.model_path),
    ]
    if image_paths:
        command.extend(("--mmproj", str(config.mmproj_path)))
    command.extend([
        "--device", config.device,
        "--gpu-layers", config.gpu_layers,
        "--fit", "on",
        "--single-turn",
        "--simple-io",
        "--system-prompt-file", str(system_path),
        "--file", str(prompt_path),
        "--output-file", str(output_path),
        "--grammar-file", str(grammar_path),
        "--temperature", "0.1",
        "--top-p", "0.9",
        "--predict", str(int(max_tokens)),
        "--no-display-prompt",
        "--color", "off",
        "--log-colors", "off",
        "--log-verbosity", "1",
    ])
    if reasoning_effort == "none":
        command.extend(("--reasoning", "off"))
    else:
        command.extend(("--reasoning-effort", reasoning_effort))
    if config.ctx_size is not None:
        command.extend(("--ctx-size", str(config.ctx_size)))
    if config.fit_target_mib is not None:
        command.extend(("--fit-target", str(config.fit_target_mib)))
    if config.image_max_tokens is not None:
        command.extend(("--image-max-tokens", str(config.image_max_tokens)))
    for path in image_paths:
        command.extend(("--image", str(path)))
    return command


async def _terminate_process(process, communication=None):
    waiter = communication
    if waiter is None:
        waiter = asyncio.create_task(process.wait())
    if process.returncode is not None:
        return await asyncio.shield(waiter)
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return await asyncio.shield(waiter)
    else:
        try:
            process.terminate()
        except ProcessLookupError:
            return await asyncio.shield(waiter)
    try:
        return await asyncio.wait_for(asyncio.shield(waiter), timeout=5)
    except asyncio.TimeoutError:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    return await asyncio.shield(waiter)


async def _request_llama_cli(messages, max_tokens, config, reasoning_effort):
    if _comfy_queue_busy():
        raise VLMBackendBusyError(
            "A ComfyUI job was queued during local llama-cli analysis."
        )
    system_prompt, user_prompt, images = _llama_cli_messages(messages)
    with tempfile.TemporaryDirectory(prefix="h3_optimizer_llama_") as directory:
        temporary_directory = Path(directory)
        system_path = temporary_directory / "system.txt"
        prompt_path = temporary_directory / "prompt.txt"
        output_path = temporary_directory / "output.txt"
        grammar_path = temporary_directory / "json_object.gbnf"
        system_path.write_text(system_prompt, encoding="utf-8")
        prompt_path.write_text(user_prompt, encoding="utf-8")
        grammar_path.write_text(JSON_OBJECT_GRAMMAR, encoding="utf-8")
        image_paths = []
        for index, data in enumerate(images, 1):
            image_path = temporary_directory / ("frame_%03d.jpg" % index)
            image_path.write_bytes(data)
            image_paths.append(image_path)

        command = _llama_cli_command(
            config,
            system_path,
            prompt_path,
            output_path,
            grammar_path,
            image_paths,
            max_tokens,
            reasoning_effort,
        )
        process_kwargs = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if os.name == "posix":
            process_kwargs["start_new_session"] = True
        else:
            process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = await asyncio.create_subprocess_exec(*command, **process_kwargs)
        except OSError as error:
            raise VLMProcessError("Could not start llama-cli: %s" % error) from error
        communication = asyncio.create_task(process.communicate())
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communication), timeout=config.timeout_seconds
            )
        except asyncio.TimeoutError as error:
            await asyncio.shield(_terminate_process(process, communication))
            raise VLMProcessError(
                "llama-cli did not answer within %d seconds." % config.timeout_seconds
            ) from error
        except asyncio.CancelledError:
            await asyncio.shield(_terminate_process(process, communication))
            raise

        output = stdout.decode("utf-8", errors="replace").strip()
        error_output = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            detail = error_output[-1600:] or output[-1600:] or "no error output"
            raise VLMProcessError("llama-cli exited with code %d: %s" % (process.returncode, detail))
        if output_path.is_file():
            recorded = output_path.read_text(encoding="utf-8")
            marker = "\n\nAssistant:\n"
            if marker in recorded:
                output = recorded.rsplit(marker, 1)[1].strip()
        if not output:
            log.error(
                "[H3PromptOptimizer] llama-cli returned no assistant content "
                "(images=%d, max_tokens=%d, ctx_size=%s).\n"
                "stdout (first %d chars): %r\n"
                "llama-cli diagnostics (last %d chars): %r",
                len(images),
                max_tokens,
                config.ctx_size if config.ctx_size is not None else "default",
                _DIAGNOSTIC_LIMIT,
                stdout.decode("utf-8", errors="replace")[:_DIAGNOSTIC_LIMIT],
                _DIAGNOSTIC_LIMIT,
                error_output[-_DIAGNOSTIC_LIMIT:],
            )
            raise VLMProcessError(
                "llama-cli returned no assistant content. Check the ComfyUI log for "
                "token and context diagnostics."
            )
        return VLMResponse(output, error_output)


def _log_invalid_response(stage, schema, response, error):
    backend = _ACTIVE_BACKEND.get() or "unknown backend"
    log.warning(
        "[H3PromptOptimizer] %s %s response is invalid during %s: %s\n"
        "response content (first %d chars): %r\n"
        "backend diagnostics (last %d chars): %r",
        backend,
        schema.__name__,
        stage,
        error,
        _DIAGNOSTIC_LIMIT,
        response.content[:_DIAGNOSTIC_LIMIT],
        _DIAGNOSTIC_LIMIT,
        response.diagnostics[-_DIAGNOSTIC_LIMIT:],
    )


async def generate_structured(
    system_prompt, user_text, images, schema, max_tokens=4096, reasoning_effort="none"
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _message_content(user_text, images)},
    ]
    response = await _request(messages, max_tokens, reasoning_effort)
    try:
        return _decode_json(response.content, schema)
    except ValueError as initial_error:
        _log_invalid_response("initial generation", schema, response, initial_error)
        repair_messages = [
            *messages,
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": (
                    "The response was invalid: %s\nReturn one corrected JSON object only. "
                    "Do not add Markdown or explanation." % initial_error
                ),
            },
        ]
        repaired = await _request(repair_messages, max_tokens, reasoning_effort)
        try:
            return _decode_json(repaired.content, schema)
        except ValueError as repair_error:
            _log_invalid_response("repair generation", schema, repaired, repair_error)
            raise
