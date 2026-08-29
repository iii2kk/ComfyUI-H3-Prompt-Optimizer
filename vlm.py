import asyncio
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from pydantic import ValidationError


DEFAULT_BASE_URL = "http://localhost:1234"
PLUGIN_DIRECTORY = Path(__file__).resolve().parent
PROMPT_GENERATOR_ENV = PLUGIN_DIRECTORY.parent / "ComfyUI-MiniMaxH3-Prompt-Generator" / ".env"


def _env_file_value(name):
    paths = [PLUGIN_DIRECTORY / ".env"]
    if name == "MINIMAX_BASE_URL":
        paths.append(PROMPT_GENERATOR_ENV)
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return None


def setting(name, fallback=None):
    value = os.getenv(name)
    if value is None:
        value = _env_file_value(name)
    return value if value not in (None, "") else fallback


def chat_completions_url():
    base_url = setting("H3_OPTIMIZER_BASE_URL", setting("MINIMAX_BASE_URL", DEFAULT_BASE_URL))
    base_url = base_url.strip()
    if "://" not in base_url:
        base_url = "http://" + base_url
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("H3_OPTIMIZER_BASE_URL must be an HTTP or HTTPS URL.")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        final_path = path
    elif path.endswith("/v1"):
        final_path = path + "/chat/completions"
    else:
        final_path = path + "/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, final_path, "", ""))


def _strip_response(text):
    value = (text or "").strip()
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
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


async def _request(messages, max_tokens):
    payload = {
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    model = setting("H3_OPTIMIZER_MODEL")
    if model:
        payload["model"] = model
    timeout_seconds = int(setting("H3_OPTIMIZER_TIMEOUT_SECONDS", "300"))
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {}
    api_key = setting("H3_OPTIMIZER_API_KEY")
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
                    content = data["choices"][0]["message"]["content"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                    raise ValueError("H3 Prompt Optimizer VLM returned an unexpected response shape.") from error
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("H3 Prompt Optimizer VLM returned no content.")
                return content
    except aiohttp.InvalidURL as error:
        raise ValueError("H3 Prompt Optimizer VLM URL is invalid.") from error
    except aiohttp.ClientConnectionError as error:
        raise ValueError("Could not connect to the H3 Prompt Optimizer VLM at %s." % chat_completions_url()) from error
    except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as error:
        raise ValueError("H3 Prompt Optimizer VLM did not answer within %d seconds." % timeout_seconds) from error


async def generate_structured(system_prompt, user_text, images, schema, max_tokens=4096):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _message_content(user_text, images)},
    ]
    raw = await _request(messages, max_tokens)
    try:
        return _decode_json(raw, schema)
    except ValueError as initial_error:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "The response was invalid: %s\nReturn one corrected JSON object only. "
                    "Do not add Markdown or explanation." % initial_error
                ),
            },
        ]
        repaired = await _request(repair_messages, max_tokens)
        return _decode_json(repaired, schema)
