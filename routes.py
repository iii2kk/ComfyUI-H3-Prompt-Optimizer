import asyncio
import logging
import re
import threading
import uuid

from aiohttp import web
from pydantic import ValidationError

from server import PromptServer

from .engine import analyze_generation
from .registry import (
    generation_artifact,
    generation_history,
    generation_snapshot,
    get_analysis,
    get_generation,
    latest_generation,
    mark_analysis_applied,
)
from .schemas import AnalyzeRequest
from .vlm import (
    VLMBackendBusyError,
    VLMConfigurationError,
    VLMProcessError,
    backend_status,
)


log = logging.getLogger(__name__)
routes = PromptServer.instance.routes
_ANALYSIS_TASKS = {}
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def _error(message, status=400):
    return web.json_response({"error": str(message)}, status=status)


def _analysis_request_id(request):
    request_id = (request.headers.get("X-H3-Analysis-Request-ID") or "").strip()
    if not request_id:
        return uuid.uuid4().hex
    if not _REQUEST_ID_RE.fullmatch(request_id):
        raise ValueError("X-H3-Analysis-Request-ID is invalid.")
    return request_id


def _cancel_analysis(request_id):
    running = _ANALYSIS_TASKS.get(request_id)
    if running is None or running[0].done():
        return False
    running[1].set()
    running[0].cancel()
    return True


@routes.get("/h3_optimizer/status")
async def optimizer_status(_request):
    try:
        vlm_status = backend_status()
    except VLMConfigurationError as error:
        return _error(error, status=503)
    return web.json_response({
        "status": "ready",
        "version": "0.8.1",
        **vlm_status,
    })


@routes.get("/h3_optimizer/generations/latest")
async def optimizer_latest_generation(request):
    optimizer_id = (request.query.get("optimizer_id") or "").strip()
    if not optimizer_id:
        return _error("optimizer_id is required.")
    generation = latest_generation(optimizer_id)
    if generation is None:
        return _error("No generation has been registered for this optimizer_id.", status=404)
    return web.json_response(generation)


@routes.get("/h3_optimizer/generations/{generation_id}")
async def optimizer_generation(request):
    generation = get_generation(request.match_info["generation_id"])
    if generation is None:
        return _error("Generation was not found.", status=404)
    return web.json_response(generation)


@routes.get("/h3_optimizer/generations/{generation_id}/video")
async def optimizer_generation_video(request):
    path = generation_artifact(request.match_info["generation_id"])
    if path is None or not path.is_file():
        return _error("Generation video was not found.", status=404)
    return web.FileResponse(path)


@routes.get("/h3_optimizer/generations/{generation_id}/snapshot")
async def optimizer_generation_snapshot(request):
    try:
        snapshot = generation_snapshot(request.match_info["generation_id"])
    except ValueError as error:
        return _error(error, status=500)
    if snapshot is None:
        return _error("Generation snapshot was not found.", status=404)
    return web.json_response(snapshot)


@routes.get("/h3_optimizer/history")
async def optimizer_history(request):
    optimizer_id = (request.query.get("optimizer_id") or "").strip()
    if not optimizer_id:
        return _error("optimizer_id is required.")
    return web.json_response({"generations": generation_history(optimizer_id)})


@routes.post("/h3_optimizer/analyze")
async def optimizer_analyze(request):
    try:
        request_id = _analysis_request_id(request)
        payload = AnalyzeRequest.model_validate(await request.json())
        if request_id in _ANALYSIS_TASKS:
            return _error("An analysis with this request ID is already running.", status=409)
        cancel_event = threading.Event()
        task = asyncio.create_task(analyze_generation(payload, cancel_event))
        _ANALYSIS_TASKS[request_id] = (task, cancel_event)
        try:
            result = await task
        except asyncio.CancelledError:
            cancel_event.set()
            return _error("Analysis was cancelled.", status=409)
        finally:
            running = _ANALYSIS_TASKS.get(request_id)
            if running is not None and running[0] is task:
                del _ANALYSIS_TASKS[request_id]
        return web.json_response(result)
    except VLMBackendBusyError as error:
        return _error(error, status=409)
    except VLMConfigurationError as error:
        return _error(error, status=503)
    except VLMProcessError as error:
        return _error(error, status=502)
    except (ValidationError, ValueError) as error:
        return _error(error)
    except Exception as error:
        log.exception("[H3PromptOptimizer] analysis failed")
        return _error("Analysis failed: %s" % error, status=500)


@routes.post("/h3_optimizer/analyze/{request_id}/cancel")
async def optimizer_cancel_analysis(request):
    request_id = request.match_info["request_id"]
    if not _REQUEST_ID_RE.fullmatch(request_id):
        return _error("Analysis request ID is invalid.")
    if not _cancel_analysis(request_id):
        return _error("Analysis is not running.", status=409)
    return web.json_response({"status": "cancelling"})


@routes.get("/h3_optimizer/analyses/{analysis_id}")
async def optimizer_analysis(request):
    analysis = get_analysis(request.match_info["analysis_id"])
    if analysis is None:
        return _error("Analysis was not found.", status=404)
    return web.json_response(analysis)


@routes.post("/h3_optimizer/analyses/{analysis_id}/applied")
async def optimizer_analysis_applied(request):
    if not mark_analysis_applied(request.match_info["analysis_id"]):
        return _error("Analysis was not found or was already marked applied.", status=409)
    return web.json_response({"status": "applied"})
