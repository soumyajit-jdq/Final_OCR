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
import fitz

from job_store import (
    create_job, get_job,
    update_job_file, mark_job_running, mark_job_failed
)
from service import ProcessingService

logger = logging.getLogger(__name__)

AI_CONCURRENCY   = asyncio.Semaphore(3)  # Run up to 3 AI extractions in parallel
INTER_PDF_DELAY  = 0.5                   # Extremely short wait; we rely on 429 retry logic instead


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

                # ── OCR / Native Text Extraction (page-by-page) ──────────────────────
                img_list, raw_text = await ProcessingService.process_pdf_pages(file_bytes, max_pages=50)
                if not img_list:
                    update_job_file(job_id, filename, "failed", error="Could not extract pages from PDF")
                    continue

                # Identify if this is a native text-based PDF (extremely fast local extraction)
                is_native = len(raw_text) > 100 and any(c.isalpha() for c in raw_text)
                
                ocr_pages = []
                if is_native:
                    logger.info(f"[Worker] Native text detected for {filename}. Skipping OCR for maximum performance!")
                    def extract_native_pages():
                        doc = fitz.open(stream=file_bytes, filetype="pdf")
                        pages_text = [page.get_text().strip() for page in doc]
                        doc.close()
                        return pages_text
                    ocr_pages = await anyio.to_thread.run_sync(extract_native_pages)
                else:
                    logger.info(f"[Worker] Scanned/Image PDF detected for {filename}. Running parallel high-speed OCR...")
                    tasks = [ProcessingService.run_ocr(img) for img in img_list]
                    ocr_pages = await asyncio.gather(*tasks)
                    
                ocr_text = "\n\n".join(ocr_pages)

                # ── Page-level Classification ───────
                page_types = await ProcessingService.classify_pages(ocr_pages)
                logger.info(f"[Worker] {filename} page classifications: {page_types}")
                
                unique_types = set(page_types)
                if not unique_types:
                    update_job_file(job_id, filename, "failed", error="Could not classify document type")
                    continue

                logger.info(f"\n==================== OCR TEXT FOR {filename} ====================\n{ocr_text}\n=======================================================================\n")

                # ── AI Extraction (rate-limited, fully concurrent) ───────────────────
                extraction_tasks = []

                # 1. Process Marksheets
                marksheet_indices = [i for i, t in enumerate(page_types) if t == "marksheet"]
                if marksheet_indices:
                    logger.info(f"[Worker] Queueing {len(marksheet_indices)} Marksheets from {filename} for concurrent AI extraction...")
                    async def extract_single_marksheet(m_idx, page_idx):
                        async with AI_CONCURRENCY:
                            page_img = [img_list[page_idx]]
                            page_text = ocr_pages[page_idx]
                            
                            structured_collection = await ProcessingService.extract_with_ai(page_img, page_text)
                            marksheets = structured_collection.get("marksheets", [])
                            for idx, ms_data in enumerate(marksheets):
                                ledger_hash = await ProcessingService.generate_ledger_hash(ms_data, "marksheet")
                                sub_name = f"{filename} (Marksheet {m_idx+1})"
                                update_job_file(
                                    job_id, sub_name,
                                    status="success",
                                    doc_type="marksheet",
                                    data=ms_data,
                                    ledger_hash=ledger_hash,
                                    raw_text=page_text,
                                )
                                logger.info(f"[Worker] ✓ {sub_name} done. Hash: {ledger_hash[:20]}...")

                    for m_idx, page_idx in enumerate(marksheet_indices):
                        extraction_tasks.append(extract_single_marksheet(m_idx, page_idx))

                # 2. Process Transcripts
                transcript_indices = [i for i, t in enumerate(page_types) if t == "transcript"]
                if transcript_indices:
                    async def extract_transcript_task():
                        logger.info(f"[Worker] Queueing Transcript from {filename} for AI extraction...")
                        async with AI_CONCURRENCY:
                            trans_images = [img_list[i] for i in transcript_indices]
                            trans_text = "\n\n".join([ocr_pages[i] for i in transcript_indices])
                            
                            structured = await ProcessingService.extract_transcript_with_ai(trans_images, trans_text)
                            ledger_hash = await ProcessingService.generate_ledger_hash(structured, "transcript")
                            sub_filename = f"{filename} (Transcript)" if len(unique_types) > 1 else filename
                            
                            update_job_file(
                                job_id, sub_filename,
                                status="success",
                                doc_type="transcript",
                                data=structured,
                                ledger_hash=ledger_hash,
                                raw_text=trans_text,
                            )
                            logger.info(f"[Worker] ✓ {sub_filename} done. Hash: {ledger_hash[:20]}...")

                    extraction_tasks.append(extract_transcript_task())

                # 3. Process Certificates
                certificate_indices = [i for i, t in enumerate(page_types) if t == "certificate"]
                if certificate_indices:
                    async def extract_certificate_task():
                        logger.info(f"[Worker] Queueing Certificate from {filename} for AI extraction...")
                        async with AI_CONCURRENCY:
                            cert_images = [img_list[i] for i in certificate_indices]
                            cert_text = "\n\n".join([ocr_pages[i] for i in certificate_indices])
                            
                            structured = await ProcessingService.extract_certificate_with_ai(cert_images, cert_text)
                            ledger_hash = await ProcessingService.generate_ledger_hash(structured, "certificate")
                            sub_filename = f"{filename} (Certificate)" if len(unique_types) > 1 else filename
                            
                            update_job_file(
                                job_id, sub_filename,
                                status="success",
                                doc_type="certificate",
                                data=structured,
                                ledger_hash=ledger_hash,
                                raw_text=cert_text,
                            )
                            logger.info(f"[Worker] ✓ {sub_filename} done. Hash: {ledger_hash[:20]}...")

                    extraction_tasks.append(extract_certificate_task())

                # Run ALL sub-tasks concurrently (throttled by AI_CONCURRENCY limit of 3)
                if extraction_tasks:
                    await asyncio.gather(*extraction_tasks)

                # Mark original parent file as done if we split it
                if len(unique_types) > 1:
                    update_job_file(job_id, filename, "success", doc_type="mixed", data={"msg": f"Split into {len(unique_types)} sub-records"})

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
