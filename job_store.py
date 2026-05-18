import uuid
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# In-Memory Job Store
# In production, replace this with PostgreSQL or Redis persistence.
# Dict[job_id -> job_record]
# ─────────────────────────────────────────────
_jobs: Dict[str, Dict[str, Any]] = {}


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"


def create_job(total_files: int, filenames: list[str]) -> str:
    """Create a new bulk job record and return its job_id."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "total": total_files,
        "completed": 0,
        "failed": 0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "results": [],
        # Pre-populate file states so the frontend can render a progress grid immediately
        "files": {name: {"status": "pending", "doc_type": None, "error": None} for name in filenames},
        "original_files": filenames,
        "original_total": total_files,
    }
    logger.info(f"[Job Store] Created job {job_id} with {total_files} files.")
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    job = _jobs.get(job_id)
    if not job:
        return None
    
    # Calculate parent PDF completion counts dynamically
    original_files = job.get("original_files", [])
    files = job.get("files", {})
    
    completed_parents = 0
    for name in original_files:
        f_state = files.get(name, {})
        if f_state.get("status") in ("success", "failed", "error"):
            completed_parents += 1
            
    job_copy = dict(job)
    job_copy["completed_parents"] = completed_parents
    job_copy["total_parents"] = job.get("original_total", len(original_files))
    return job_copy


def update_job_file(job_id: str, filename: str, status: str, doc_type: str = None,
                    data: dict = None, error: str = None, ledger_hash: str = None,
                    raw_text: str = None):
    """Atomically update a single file's result inside a job."""
    job = _jobs.get(job_id)
    if not job:
        return

    # If this is a dynamically added sub-file, track it in the total
    if filename not in job["files"]:
        job["total"] += 1
        job["files"][filename] = {}

    job["files"][filename] = {
        "status": status,
        "doc_type": doc_type,
        "error": error,
        "ledger_hash": ledger_hash,
    }

    if status == "success" and data:
        job["results"].append({
            "filename": filename,
            "doc_type": doc_type,
            "data": data,
            "ledger_hash": ledger_hash,
            "raw_text": raw_text,
        })
        job["completed"] += 1
    elif status in ("failed", "error"):
        job["failed"] += 1

    # Determine overall job status
    finished = job["completed"] + job["failed"]
    if finished >= job["total"]:
        job["status"] = JobStatus.DONE
    else:
        job["status"] = JobStatus.RUNNING

    job["updated_at"] = datetime.utcnow().isoformat()


def mark_job_running(job_id: str):
    job = _jobs.get(job_id)
    if job:
        job["status"] = JobStatus.RUNNING
        job["updated_at"] = datetime.utcnow().isoformat()


def mark_job_failed(job_id: str, error: str):
    job = _jobs.get(job_id)
    if job:
        job["status"] = JobStatus.FAILED
        job["error"] = error
        job["updated_at"] = datetime.utcnow().isoformat()
