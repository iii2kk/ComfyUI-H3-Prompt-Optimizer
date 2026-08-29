from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import threading
import uuid

import folder_paths
from comfy_api.latest import Types


STORE_DIRECTORY = "h3_prompt_optimizer"
DATABASE_FILENAME = "registry.sqlite3"
API_PROMPT_FILENAME = "api_prompt.json"
WORKFLOW_FILENAME = "workflow.json"

_LOCK = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _store_root():
    output_root = Path(folder_paths.get_output_directory()).resolve()
    root = (output_root / STORE_DIRECTORY).resolve()
    try:
        root.relative_to(output_root)
    except ValueError as error:
        raise ValueError("H3 Prompt Optimizer storage resolves outside the output directory.") from error
    root.mkdir(parents=True, exist_ok=True)
    return root


def _connect():
    connection = sqlite3.connect(_store_root() / DATABASE_FILENAME)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS generations (
            generation_id TEXT PRIMARY KEY,
            optimizer_id TEXT NOT NULL,
            comfy_prompt_id TEXT NOT NULL,
            recorder_node_id TEXT NOT NULL,
            project_name TEXT NOT NULL,
            segment_id INTEGER NOT NULL,
            parent_generation_id TEXT,
            prompt TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            source_prompt_hash TEXT NOT NULL,
            reference_signature TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            fps REAL NOT NULL,
            frames INTEGER NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(comfy_prompt_id, recorder_node_id),
            FOREIGN KEY(parent_generation_id) REFERENCES generations(generation_id)
        );

        CREATE INDEX IF NOT EXISTS generations_optimizer_created
        ON generations(optimizer_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY,
            generation_id TEXT NOT NULL,
            feedback TEXT NOT NULL,
            time_range_json TEXT,
            target_state_hash TEXT NOT NULL,
            critic_json TEXT NOT NULL,
            prompt_ir_json TEXT NOT NULL,
            patch_json TEXT NOT NULL,
            optimized_prompt TEXT NOT NULL,
            optimized_prompt_hash TEXT NOT NULL,
            diff TEXT NOT NULL,
            applied_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(generation_id) REFERENCES generations(generation_id)
        );
        """
    )
    return connection


@contextmanager
def _database():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(_json_safe(value), file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def _row_dict(row):
    if row is None:
        return None
    value = dict(row)
    value["reference_signature"] = json.loads(value["reference_signature"])
    value["video"] = {
        "url": "/h3_optimizer/generations/%s/video" % value["generation_id"],
        "fps": value["fps"],
        "frames": value["frames"],
        "duration_sec": value["frames"] / value["fps"],
        "sha256": value["artifact_sha256"],
    }
    value.pop("artifact_path", None)
    return value


def get_generation(generation_id):
    with _LOCK, _database() as connection:
        row = connection.execute(
            "SELECT * FROM generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
    return _row_dict(row)


def latest_generation(optimizer_id):
    with _LOCK, _database() as connection:
        row = connection.execute(
            "SELECT * FROM generations WHERE optimizer_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (optimizer_id,),
        ).fetchone()
    return _row_dict(row)


def generation_history(optimizer_id, limit=50):
    with _LOCK, _database() as connection:
        rows = connection.execute(
            "SELECT * FROM generations WHERE optimizer_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (optimizer_id, int(limit)),
        ).fetchall()
    return [_row_dict(row) for row in rows]


def generation_artifact(generation_id):
    with _LOCK, _database() as connection:
        row = connection.execute(
            "SELECT artifact_path FROM generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
    if row is None:
        return None
    root = _store_root()
    path = (root / row["artifact_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Generation artifact resolves outside H3 Prompt Optimizer storage.") from error
    return path


def generation_snapshot(generation_id):
    artifact_path = generation_artifact(generation_id)
    if artifact_path is None:
        return None
    api_prompt_path = artifact_path.parent / API_PROMPT_FILENAME
    if not api_prompt_path.is_file():
        return None

    try:
        with api_prompt_path.open("r", encoding="utf-8") as file:
            api_prompt = _json_safe(json.load(file))
        workflow_path = artifact_path.parent / WORKFLOW_FILENAME
        if workflow_path.is_file():
            with workflow_path.open("r", encoding="utf-8") as file:
                workflow = _json_safe(json.load(file))
        else:
            workflow = None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Generation snapshot is not readable: %s" % error) from error
    return {"api_prompt": api_prompt, "workflow": workflow}


def register_generation(video, prompt_state, optimizer_id, project_name, segment_id,
                        comfy_prompt_id, recorder_node_id, api_prompt=None, workflow=None):
    with _LOCK:
        with _database() as connection:
            existing = connection.execute(
                "SELECT * FROM generations WHERE comfy_prompt_id = ? AND recorder_node_id = ?",
                (comfy_prompt_id, recorder_node_id),
            ).fetchone()
            if existing is not None:
                return _row_dict(existing)

        generation_id = "gen_" + uuid.uuid4().hex
        root = _store_root()
        artifact_directory = root / "artifacts" / generation_id
        artifact_directory.mkdir(parents=True, exist_ok=False)
        artifact_path = artifact_directory / "video.mp4"
        try:
            video.save_to(
                str(artifact_path),
                format=Types.VideoContainer.MP4,
                codec=Types.VideoCodec.AUTO,
            )
            if api_prompt is not None:
                _write_json(artifact_directory / API_PROMPT_FILENAME, api_prompt)
            if workflow is not None:
                _write_json(artifact_directory / WORKFLOW_FILENAME, workflow)
            artifact_sha256 = _sha256_file(artifact_path)
            fps = float(video.get_frame_rate())
            frames = int(video.get_frame_count())
            prompt = prompt_state["prompt"]
            with _database() as connection:
                parent = connection.execute(
                    "SELECT generation_id FROM generations WHERE optimizer_id = ? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                    (optimizer_id,),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO generations (
                        generation_id, optimizer_id, comfy_prompt_id, recorder_node_id,
                        project_name, segment_id, parent_generation_id, prompt, prompt_hash,
                        source_prompt_hash, reference_signature, duration_seconds, fps, frames,
                        artifact_path, artifact_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation_id,
                        optimizer_id,
                        comfy_prompt_id,
                        recorder_node_id,
                        project_name,
                        int(segment_id),
                        parent["generation_id"] if parent else None,
                        prompt,
                        prompt_state["prompt_hash"],
                        prompt_state["source_prompt_hash"],
                        json.dumps(prompt_state["reference_signature"], ensure_ascii=False, sort_keys=True),
                        float(prompt_state["duration_seconds"]),
                        fps,
                        frames,
                        artifact_path.relative_to(root).as_posix(),
                        artifact_sha256,
                        _now(),
                    ),
                )
                if parent is not None:
                    connection.execute(
                        """
                        UPDATE analyses SET applied_at = ? WHERE analysis_id = (
                            SELECT analysis_id FROM analyses
                            WHERE generation_id = ? AND optimized_prompt_hash = ?
                                AND applied_at IS NULL
                            ORDER BY created_at DESC, rowid DESC LIMIT 1
                        )
                        """,
                        (_now(), parent["generation_id"], prompt_state["prompt_hash"]),
                    )
                row = connection.execute(
                    "SELECT * FROM generations WHERE generation_id = ?", (generation_id,)
                ).fetchone()
            return _row_dict(row)
        except Exception:
            if artifact_directory.is_dir():
                shutil.rmtree(artifact_directory)
            raise


def create_analysis(generation_id, feedback, time_range, target_state_hash, critic,
                    prompt_ir, patch, optimized_prompt, optimized_prompt_hash, diff):
    analysis_id = "analysis_" + uuid.uuid4().hex
    with _LOCK, _database() as connection:
        connection.execute(
            """
            INSERT INTO analyses (
                analysis_id, generation_id, feedback, time_range_json, target_state_hash,
                critic_json, prompt_ir_json, patch_json, optimized_prompt,
                optimized_prompt_hash, diff, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                generation_id,
                feedback,
                json.dumps(time_range, ensure_ascii=False) if time_range else None,
                target_state_hash,
                json.dumps(critic, ensure_ascii=False),
                json.dumps(prompt_ir, ensure_ascii=False),
                json.dumps(patch, ensure_ascii=False),
                optimized_prompt,
                optimized_prompt_hash,
                diff,
                _now(),
            ),
        )
    return analysis_id


def get_analysis(analysis_id):
    with _LOCK, _database() as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        return None
    value = dict(row)
    time_range_json = value.pop("time_range_json")
    value["time_range"] = json.loads(time_range_json) if time_range_json else None
    value["critic"] = json.loads(value.pop("critic_json"))
    value["prompt_ir"] = json.loads(value.pop("prompt_ir_json"))
    value["proposal"] = json.loads(value.pop("patch_json"))
    return value


def mark_analysis_applied(analysis_id):
    with _LOCK, _database() as connection:
        cursor = connection.execute(
            "UPDATE analyses SET applied_at = ? WHERE analysis_id = ? AND applied_at IS NULL",
            (_now(), analysis_id),
        )
    return cursor.rowcount == 1
