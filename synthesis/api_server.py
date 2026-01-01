"""Persistent API server for synthesis jobs."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import requests
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

from config.config import get_config

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = APP_ROOT / "input-images"
OUTPUT_ROOT = APP_ROOT / "augmented-output"
SERVICE_ROOT = OUTPUT_ROOT / ".service"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RETRIEVAL_BASE_URL = os.getenv(
    "SYNTHESIS_RETRIEVAL_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SERVICE_ROOT.mkdir(parents=True, exist_ok=True)

CONFIG = get_config()


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse a boolean-like value."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid boolean value: {value}")


def ensure_list(value: Any) -> list[Any]:
    """Normalize a scalar-or-list field into a list."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def sanitize_filename(name: str, fallback_stem: str) -> str:
    """Return a filesystem-safe image filename."""
    raw_name = Path(name or "").name
    stem = Path(raw_name).stem or fallback_stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or fallback_stem

    suffix = Path(raw_name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".png"

    return f"{stem}{suffix}"


def resolve_input_dir(raw_value: str) -> Path:
    """Resolve an input directory relative to /app if needed."""
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = APP_ROOT / candidate
    return candidate.resolve()


def resolve_output_dir(raw_value: str | None, job_id: str) -> Path:
    """Resolve an output directory anchored under augmented-output/."""
    if not raw_value:
        return (OUTPUT_ROOT / job_id).resolve()

    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = OUTPUT_ROOT / candidate

    resolved = candidate.resolve()
    if resolved != OUTPUT_ROOT and OUTPUT_ROOT not in resolved.parents:
        raise ValueError("output_dir must stay under ./augmented-output")
    return resolved


def count_images(input_dir: Path) -> int:
    """Count supported images in a directory."""
    total = 0
    for path in input_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            total += 1
    return total


def resolve_image_url(raw_url: str) -> str:
    """Resolve retrieval-relative image URLs against the configured base URL."""
    if raw_url.startswith("/"):
        return f"{RETRIEVAL_BASE_URL}{raw_url}"

    parsed = urlparse(raw_url)
    if parsed.scheme:
        return raw_url

    return urljoin(f"{RETRIEVAL_BASE_URL}/", raw_url.lstrip("/"))


def safe_relative_to(root: Path, candidate: Path) -> Path | None:
    """Return the candidate path relative to root if it is contained within root."""
    try:
        return candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file if it exists and is valid."""
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def log_tail(path: Path, max_chars: int = 4000) -> str:
    """Return the tail of a log file."""
    if not path.exists():
        return ""

    try:
        return path.read_text(errors="replace")[-max_chars:]
    except OSError:
        return ""


def config_summary() -> dict[str, Any]:
    """Expose the active provider and model configuration."""
    return {
        "provider": CONFIG.azure.provider,
        "vision_model": (
            CONFIG.azure.vision_deployment if CONFIG.azure.is_azure else CONFIG.azure.vision_model
        ),
        "image_edit_model": (
            CONFIG.azure.image_edit_deployment if CONFIG.azure.is_azure else CONFIG.azure.image_edit_model
        ),
        "requests_per_minute": CONFIG.processing.requests_per_minute,
        "max_concurrent": CONFIG.processing.max_concurrent,
        "retrieval_base_url": RETRIEVAL_BASE_URL,
    }


@dataclass
class JobRecord:
    """A synthesis job tracked by the service."""

    job_id: str
    status: str
    created_at: str
    source_mode: str
    input_dir: str
    output_dir: str
    domain: str
    num_variants: int
    resume: bool
    source_count: int
    command: list[str]
    staged_files: list[str] = field(default_factory=list)
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    log_file: str = ""
    progress_file: str = ""
    metadata_file: str = ""
    suggestions_file: str = ""


class SynthesisJobManager:
    """Manage at most one active synthesis process at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._active_job_id: str | None = None

    def start_job(
        self,
        *,
        job_id: str,
        source_mode: str,
        input_dir: Path,
        output_dir: Path,
        domain: str,
        num_variants: int,
        resume: bool,
        staged_files: list[str],
        source_count: int,
    ) -> JobRecord:
        """Start a new synthesis job."""
        with self._lock:
            active_job = self.current_job_record()
            if active_job and active_job.status in {"starting", "running", "cancelling"}:
                raise RuntimeError(f"Job {active_job.job_id} is already active")

            service_dir = SERVICE_ROOT / job_id
            service_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            log_path = service_dir / "job.log"
            progress_path = output_dir / "progress.json"
            metadata_path = output_dir / "metadata.json"
            suggestions_path = output_dir / "suggestions.json"

            command = [
                sys.executable,
                "run_augmentation.py",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--domain",
                domain,
                "--num-variants",
                str(num_variants),
            ]
            if resume:
                command.append("--resume")

            job = JobRecord(
                job_id=job_id,
                status="starting",
                created_at=utc_now(),
                source_mode=source_mode,
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                domain=domain,
                num_variants=num_variants,
                resume=resume,
                source_count=source_count,
                command=command,
                staged_files=staged_files,
                log_file=str(log_path),
                progress_file=str(progress_path),
                metadata_file=str(metadata_path),
                suggestions_file=str(suggestions_path),
            )
            self._jobs[job_id] = job

            log_handle = open(log_path, "a", buffering=1)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(APP_ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except Exception:
                log_handle.close()
                raise

            job.pid = process.pid
            job.status = "running"
            job.started_at = utc_now()
            self._processes[job_id] = process
            self._active_job_id = job_id

            thread = threading.Thread(
                target=self._wait_for_completion,
                args=(job_id, process, log_handle),
                daemon=True,
            )
            thread.start()
            return job

    def _wait_for_completion(
        self,
        job_id: str,
        process: subprocess.Popen[str],
        log_handle: Any,
    ) -> None:
        """Wait for a subprocess and update its final state."""
        returncode = process.wait()
        log_handle.close()

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return

            job.returncode = returncode
            job.finished_at = utc_now()

            if job.status == "cancelling":
                job.status = "cancelled"
            elif returncode == 0:
                job.status = "completed"
            else:
                job.status = "failed"
                job.error = log_tail(Path(job.log_file), 1200) or f"Process exited with code {returncode}"

            if self._active_job_id == job_id:
                self._active_job_id = None
            self._processes.pop(job_id, None)

    def current_job_record(self) -> JobRecord | None:
        """Return the currently active job, if any."""
        if not self._active_job_id:
            return None
        return self._jobs.get(self._active_job_id)

    def get_job(self, job_id: str) -> JobRecord | None:
        """Return a job record by ID."""
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> JobRecord:
        """Cancel a running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in {"running", "starting"}:
                raise RuntimeError(f"Job {job_id} is not running")

            process = self._processes.get(job_id)
            job.status = "cancelling"

        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        return job

    def artifact_url(self, file_path: str) -> str | None:
        """Convert an output path into a downloadable URL."""
        relative_path = safe_relative_to(OUTPUT_ROOT, Path(file_path))
        if relative_path is None:
            return None
        return f"/artifacts/{relative_path.as_posix()}"

    def serialize_job(self, job: JobRecord, include_log_tail: bool = False) -> dict[str, Any]:
        """Serialize a job with progress and artifact metadata."""
        payload = asdict(job)
        payload["status_url"] = f"/jobs/{job.job_id}"
        payload["cancel_url"] = f"/jobs/{job.job_id}/cancel"
        payload["logs_url"] = f"/jobs/{job.job_id}/logs"
        payload["progress"] = self._progress_summary(Path(job.progress_file))
        payload["result"] = self._result_summary(job)
        if include_log_tail:
            payload["log_tail"] = log_tail(Path(job.log_file))
        return payload

    def status_payload(self) -> dict[str, Any]:
        """Return the overall service status."""
        jobs = sorted(
            self._jobs.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )
        current_job = self.current_job_record()
        return {
            "service": "labelingcopilot-synthesis",
            "config": config_summary(),
            "current_job": self.serialize_job(current_job) if current_job else None,
            "recent_jobs": [self.serialize_job(job) for job in jobs[:10]],
        }

    def _progress_summary(self, progress_path: Path) -> dict[str, Any] | None:
        data = read_json(progress_path)
        if not data:
            return None
        return {
            "session_id": data.get("session_id"),
            "started_at": data.get("started_at"),
            "last_updated": data.get("last_updated"),
            "stats": data.get("stats", {}),
        }

    def _result_summary(self, job: JobRecord) -> dict[str, Any] | None:
        metadata = read_json(Path(job.metadata_file))
        if not metadata:
            return None

        images = metadata.get("images", [])
        artifacts = []
        for item in images[:50]:
            output_image = item.get("output_image")
            artifact = {
                "source_image": item.get("source_image"),
                "output_image": output_image,
                "category": item.get("category"),
                "prompt": item.get("prompt"),
            }
            if output_image:
                artifact["download_url"] = self.artifact_url(output_image)
            artifacts.append(artifact)

        return {
            "metadata_file": job.metadata_file,
            "metadata_url": self.artifact_url(job.metadata_file),
            "aggregate_scores": metadata.get("aggregate_scores"),
            "total_original_images": metadata.get("total_original_images"),
            "total_augmented_images": metadata.get("total_augmented_images"),
            "generated_images": artifacts,
        }


JOB_MANAGER = SynthesisJobManager()


async def parse_synthesize_request(request: Request) -> dict[str, Any]:
    """Parse a synthesis trigger request from JSON or multipart form-data."""
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Malformed JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        return {
            "domain": payload.get("domain"),
            "num_variants": payload.get("num_variants"),
            "resume": payload.get("resume"),
            "input_dir": payload.get("input_dir"),
            "output_dir": payload.get("output_dir"),
            "image_paths": ensure_list(payload.get("image_paths")),
            "image_urls": ensure_list(payload.get("image_urls")),
            "files": [],
        }

    form = await request.form()
    files = [item for item in form.getlist("images") if isinstance(item, UploadFile)]
    return {
        "domain": form.get("domain"),
        "num_variants": form.get("num_variants"),
        "resume": form.get("resume"),
        "input_dir": form.get("input_dir"),
        "output_dir": form.get("output_dir"),
        "image_paths": form.getlist("image_paths"),
        "image_urls": form.getlist("image_urls"),
        "files": files,
    }


async def save_uploaded_file(upload: UploadFile, destination: Path, index: int) -> str:
    """Save an uploaded image to the staging directory."""
    file_name = sanitize_filename(upload.filename or "", f"upload_{index}")
    output_path = destination / file_name

    with output_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)

    await upload.close()
    return file_name


def stage_source_path(source_path: str, destination: Path, index: int) -> str:
    """Stage a container-visible seed image into the job input directory."""
    resolved_source = resolve_input_dir(source_path)
    if not resolved_source.exists() or not resolved_source.is_file():
        raise ValueError(f"Seed image does not exist: {source_path}")
    if resolved_source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported seed image type: {source_path}")

    file_name = sanitize_filename(resolved_source.name, f"seed_{index}")
    output_path = destination / file_name
    try:
        output_path.symlink_to(resolved_source)
    except OSError:
        shutil.copy2(resolved_source, output_path)
    return file_name


def stage_source_url(source_url: str, destination: Path, index: int) -> str:
    """Fetch and stage a remote seed image."""
    resolved_url = resolve_image_url(source_url)
    response = requests.get(resolved_url, timeout=60)
    response.raise_for_status()

    parsed = urlparse(resolved_url)
    file_name = sanitize_filename(unquote(Path(parsed.path).name), f"remote_{index}")
    output_path = destination / file_name
    output_path.write_bytes(response.content)
    return file_name


async def prepare_job(request: Request) -> dict[str, Any]:
    """Validate the incoming request and stage seed images if needed."""
    payload = await parse_synthesize_request(request)

    try:
        num_variants = int(payload.get("num_variants") or 3)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="num_variants must be an integer") from exc

    if num_variants < 1:
        raise HTTPException(status_code=400, detail="num_variants must be >= 1")

    try:
        resume = parse_bool(payload.get("resume"), default=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    domain = payload.get("domain") or "general computer vision imagery"
    input_dir_value = (payload.get("input_dir") or "").strip()
    output_dir_value = (payload.get("output_dir") or "").strip()
    image_paths = payload.get("image_paths") or []
    image_urls = payload.get("image_urls") or []
    files = payload.get("files") or []

    if input_dir_value and (image_paths or image_urls or files):
        raise HTTPException(
            status_code=400,
            detail="input_dir cannot be combined with image_paths, image_urls, or uploaded images",
        )

    if resume and not output_dir_value:
        raise HTTPException(
            status_code=400,
            detail="resume=true requires an explicit output_dir under ./augmented-output",
        )

    job_id = uuid.uuid4().hex[:12]
    source_mode = "default_directory"
    staged_files: list[str] = []

    if files or image_paths or image_urls:
        source_mode = "staged_images"
        input_dir = (SERVICE_ROOT / job_id / "input").resolve()
        input_dir.mkdir(parents=True, exist_ok=True)

        for index, upload in enumerate(files, start=1):
            staged_files.append(await save_uploaded_file(upload, input_dir, index))
        for index, source_path in enumerate(image_paths, start=len(staged_files) + 1):
            try:
                staged_files.append(stage_source_path(source_path, input_dir, index))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        for index, source_url in enumerate(image_urls, start=len(staged_files) + 1):
            try:
                staged_files.append(stage_source_url(source_url, input_dir, index))
            except requests.RequestException as exc:
                raise HTTPException(status_code=400, detail=f"Failed to fetch seed image: {source_url}") from exc
    elif input_dir_value:
        source_mode = "input_directory"
        input_dir = resolve_input_dir(input_dir_value)
    else:
        input_dir = DEFAULT_INPUT_DIR.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Input directory does not exist: {input_dir}")

    source_count = count_images(input_dir)
    if source_count == 0:
        raise HTTPException(status_code=400, detail=f"No .jpg/.jpeg/.png images found in {input_dir}")

    try:
        output_dir = resolve_output_dir(output_dir_value or None, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "job_id": job_id,
        "source_mode": source_mode,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "domain": domain,
        "num_variants": num_variants,
        "resume": resume,
        "staged_files": staged_files,
        "source_count": source_count,
    }


async def health(_: Request) -> JSONResponse:
    """Health endpoint."""
    status = "busy" if JOB_MANAGER.current_job_record() else "ok"
    return JSONResponse({"status": status, "config": config_summary()})


async def status(_: Request) -> JSONResponse:
    """Overall service status."""
    return JSONResponse(JOB_MANAGER.status_payload())


async def synthesize(request: Request) -> JSONResponse:
    """Start a synthesis job."""
    job_request = await prepare_job(request)

    try:
        job = JOB_MANAGER.start_job(**job_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return JSONResponse(JOB_MANAGER.serialize_job(job, include_log_tail=True), status_code=202)


async def get_job(request: Request) -> JSONResponse:
    """Return a job by ID."""
    job_id = request.path_params["job_id"]
    job = JOB_MANAGER.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return JSONResponse(JOB_MANAGER.serialize_job(job, include_log_tail=True))


async def cancel_job(request: Request) -> JSONResponse:
    """Cancel a running job."""
    job_id = request.path_params["job_id"]
    try:
        job = JOB_MANAGER.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(JOB_MANAGER.serialize_job(job, include_log_tail=True))


async def get_job_logs(request: Request) -> PlainTextResponse:
    """Return the current log tail for a job."""
    job_id = request.path_params["job_id"]
    job = JOB_MANAGER.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

    tail_chars = request.query_params.get("tail")
    try:
        tail_size = int(tail_chars) if tail_chars else 4000
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="tail must be an integer") from exc

    return PlainTextResponse(log_tail(Path(job.log_file), max_chars=max(256, tail_size)))


async def get_artifact(request: Request) -> FileResponse:
    """Serve generated artifacts from augmented-output/."""
    relative_path = request.path_params["path"]
    candidate = (OUTPUT_ROOT / relative_path).resolve()
    if safe_relative_to(OUTPUT_ROOT, candidate) is None or not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(candidate)


app = Starlette(
    debug=False,
    routes=[
        Route("/health", health),
        Route("/status", status),
        Route("/synthesize", synthesize, methods=["POST"]),
        Route("/jobs/{job_id:str}", get_job, methods=["GET"]),
        Route("/jobs/{job_id:str}/cancel", cancel_job, methods=["POST"]),
        Route("/jobs/{job_id:str}/logs", get_job_logs, methods=["GET"]),
        Route("/artifacts/{path:path}", get_artifact, methods=["GET"]),
    ],
)
