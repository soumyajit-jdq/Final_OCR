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
                
                # Group pages into distinct logical documents (groups)
                doc_groups = []
                current_group = None
                
                for i, page_type in enumerate(page_types):
                    text = ocr_pages[i]
                    img = img_list[i]
                    
                    is_new_doc = False
                    if not current_group:
                        is_new_doc = True
                    elif current_group['type'] != page_type:
                        is_new_doc = True
                    else:
                        # Same type. Let's see if we should split.
                        if page_type == 'certificate':
                            # Certificates are 1 page each. Every certificate page is a new document.
                            is_new_doc = True
                        elif page_type == 'marksheet':
                            # Marksheet pages are processed individually.
                            is_new_doc = True
                        elif page_type == 'transcript':
                            # For transcripts, start a new one if we detect student registration/header keywords
                            text_lower = text.lower()
                            has_start_keyword = any(kw in text_lower for kw in [
                                "official transcript", "transcript of academic record", 
                                "consolidated statement", "transcript of record"
                            ])
                            has_header_field = any(kw in text_lower for kw in [
                                "name of student", "name of candidate", "registration no", "regn. no", 
                                "roll no", "enrollment no", "admission year", "completion year"
                            ])
                            if has_start_keyword:
                                is_new_doc = True
                            elif has_header_field and not any(kw in text_lower for kw in ["continued", "page 2", "page 3", "page 4"]):
                                is_new_doc = True

                    if is_new_doc:
                        current_group = {
                            'type': page_type,
                            'pages': [i],
                            'images': [img],
                            'text': text
                        }
                        doc_groups.append(current_group)
                    else:
                        current_group['pages'].append(i)
                        current_group['images'].append(img)
                        current_group['text'] += "\n\n" + text

                if not doc_groups:
                    update_job_file(job_id, filename, "failed", error="Could not classify document type")
                    continue

                logger.info(f"\n==================== OCR TEXT FOR {filename} ====================\n{ocr_text}\n=======================================================================\n")

                # ── AI Extraction (rate-limited, fully concurrent) ───────────────────
                extraction_tasks = []

                # Count counts per type for naming
                type_counts = {}
                for group in doc_groups:
                    t = group['type']
                    type_counts[t] = type_counts.get(t, 0) + 1

                type_indices = {}

                for group in doc_groups:
                    doc_type = group['type']
                    ocr_text_chunk = group['text']
                    images = group['images']

                    # Track index of this document type
                    type_indices[doc_type] = type_indices.get(doc_type, 0) + 1
                    current_idx = type_indices[doc_type]
                    total_of_type = type_counts[doc_type]

                    # Determine filename label
                    if total_of_type > 1:
                        type_label = doc_type.capitalize()
                        sub_name = f"{filename} ({type_label} {current_idx})"
                    else:
                        if len(doc_groups) > 1:
                            sub_name = f"{filename} ({doc_type.capitalize()})"
                        else:
                            sub_name = filename

                    async def extract_task(g_type, g_imgs, g_text, name_label):
                        async with AI_CONCURRENCY:
                            logger.info(f"[Worker] Queueing {g_type.capitalize()} ({name_label}) for AI extraction...")
                            if g_type == "marksheet":
                                structured_collection = await ProcessingService.extract_with_ai(g_imgs, g_text)
                                marksheets = structured_collection.get("marksheets", [])
                                for ms_idx, ms_data in enumerate(marksheets):
                                    ledger_hash = await ProcessingService.generate_ledger_hash(ms_data, "marksheet")
                                    final_name = f"{name_label} Part {ms_idx+1}" if len(marksheets) > 1 else name_label
                                    update_job_file(
                                        job_id, final_name,
                                        status="success",
                                        doc_type="marksheet",
                                        data=ms_data,
                                        ledger_hash=ledger_hash,
                                        raw_text=g_text,
                                    )
                                    logger.info(f"[Worker] ✓ {final_name} done. Hash: {ledger_hash[:20]}...")
                            elif g_type == "transcript":
                                structured = await ProcessingService.extract_transcript_with_ai(g_imgs, g_text)
                                ledger_hash = await ProcessingService.generate_ledger_hash(structured, "transcript")
                                update_job_file(
                                    job_id, name_label,
                                    status="success",
                                    doc_type="transcript",
                                    data=structured,
                                    ledger_hash=ledger_hash,
                                    raw_text=g_text,
                                )
                                logger.info(f"[Worker] ✓ {name_label} done. Hash: {ledger_hash[:20]}...")
                            elif g_type == "certificate":
                                structured = await ProcessingService.extract_certificate_with_ai(g_imgs, g_text)
                                ledger_hash = await ProcessingService.generate_ledger_hash(structured, "certificate")
                                update_job_file(
                                    job_id, name_label,
                                    status="success",
                                    doc_type="certificate",
                                    data=structured,
                                    ledger_hash=ledger_hash,
                                    raw_text=g_text,
                                )
                                logger.info(f"[Worker] ✓ {name_label} done. Hash: {ledger_hash[:20]}...")

                    extraction_tasks.append(extract_task(doc_type, images, ocr_text_chunk, sub_name))

                # Run ALL sub-tasks concurrently (throttled by AI_CONCURRENCY limit of 3)
                if extraction_tasks:
                    await asyncio.gather(*extraction_tasks)

                # Mark original parent file as done if we split it
                if len(doc_groups) > 1:
                    update_job_file(job_id, filename, "success", doc_type="mixed", data={"msg": f"Split into {len(doc_groups)} sub-records"})

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
