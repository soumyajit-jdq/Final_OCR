"""
worker.py — Async Background Worker for Bulk ZIP Processing

Architecture:
  1. FastAPI route receives ZIP → saves bytes → creates job record → returns 202 immediately.
  2. asyncio.create_task() launches process_zip_job() in the background.
  3. process_zip_job() uses a strict asyncio.Semaphore(1) + per-task delays to rate-limit
     all AI API calls to ~1 request every N seconds, staying safely inside free-tier limits.
  4. Each file result is written to job_store so the frontend can poll /api/v1/job/{job_id}.
"""

import asyncio
import os
import zipfile
import shutil
import tempfile
import logging
import re
import anyio

from job_store import (
    create_job, get_job,
    update_job_file, mark_job_running, mark_job_failed
)
from service import ProcessingService

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# RATE LIMIT CONFIG
#
# Gemini Free Tier:  15 requests/minute  → 1 req every 4 seconds
# Cerebras Free:     30 requests/minute  → 1 req every 2 seconds
#
# We use 1 concurrent AI call at a time + 5 second delay between PDFs.
# This keeps us at ~12 AI calls/minute, safely under the 15 RPM limit.
# ─────────────────────────────────────────────────────────────────
AI_CONCURRENCY   = asyncio.Semaphore(1)  # Only 1 AI extraction at a time
INTER_PDF_DELAY  = 5.0                   # Seconds to wait between each PDF


async def process_zip_job(job_id: str, zip_bytes: bytes):
    """
    Background coroutine: processes each PDF in the ZIP sequentially
    with rate-limiting, updating the job store as results arrive.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        mark_job_running(job_id)

        # Extract ZIP
        zip_path = os.path.join(temp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        extract_path = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)

        pdf_files = []
        for root, _, files in os.walk(extract_path):
            for fname in sorted(files):
                if fname.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, fname))

        logger.info(f"[Worker] Job {job_id}: Processing {len(pdf_files)} PDFs sequentially.")

        for idx, pdf_path in enumerate(pdf_files):
            filename = os.path.basename(pdf_path)
            logger.info(f"[Worker] Job {job_id}: Starting file {idx+1}/{len(pdf_files)}: {filename}")

            try:
                with open(pdf_path, "rb") as f:
                    file_bytes = f.read()

                # ── Validation ──────────────────────────────────────
                validation = await ProcessingService.validate_document(file_bytes, filename)
                if not validation.is_valid:
                    update_job_file(job_id, filename, "failed", error=f"Validation: {validation.instruction}")
                    continue

                # ── OCR (page-by-page) ──────────────────────────────
                img_list, raw_text = await ProcessingService.process_pdf_pages(file_bytes, max_pages=50)
                if not img_list:
                    update_job_file(job_id, filename, "failed", error="Could not extract pages from PDF")
                    continue

                # Run OCR on each page (I/O bound, fast, no AI quota)
                ocr_pages = []
                for img in img_list:
                    text = await ProcessingService.run_ocr(img)
                    ocr_pages.append(text)
                ocr_text = "\n\n".join(ocr_pages)

                # ── Classification (keyword-first, no AI cost) ───────
                doc_type = await ProcessingService.classify_document(ocr_text)
                logger.info(f"[Worker] {filename} classified as: {doc_type}")

                if doc_type == "unknown":
                    update_job_file(job_id, filename, "failed", error="Could not classify document type")
                    continue

                # ── AI Extraction (rate-limited) ─────────────────────
                async with AI_CONCURRENCY:
                    logger.info(f"[Worker] Extracting {filename} (type={doc_type})...")
                    if doc_type == "marksheet":
                        structured = await ProcessingService.extract_with_ai(img_list, ocr_text)
                    elif doc_type == "transcript":
                        structured = await ProcessingService.extract_transcript_with_ai(img_list, ocr_text)
                    elif doc_type == "certificate":
                        structured = await ProcessingService.extract_certificate_with_ai(img_list, ocr_text)
                    else:
                        raise ValueError(f"Unsupported doc_type: {doc_type}")

                # ── Hash ─────────────────────────────────────────────
                ledger_hash = await ProcessingService.generate_ledger_hash(structured, doc_type)

                update_job_file(
                    job_id, filename,
                    status="success",
                    doc_type=doc_type,
                    data=structured,
                    ledger_hash=ledger_hash,
                )
                logger.info(f"[Worker] ✓ {filename} done. Hash: {ledger_hash[:20]}...")

            except Exception as e:
                logger.error(f"[Worker] ✗ {filename} failed: {e}")
                update_job_file(job_id, filename, "error", error=str(e))

            # ── Rate-limit delay between PDFs ─────────────────────────
            if idx < len(pdf_files) - 1:
                logger.info(f"[Worker] Waiting {INTER_PDF_DELAY}s before next PDF to respect API rate limits...")
                await anyio.sleep(INTER_PDF_DELAY)

        logger.info(f"[Worker] Job {job_id} complete.")

    except Exception as e:
        logger.error(f"[Worker] Job {job_id} failed catastrophically: {e}")
        mark_job_failed(job_id, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
