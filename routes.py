import asyncio
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from service import ProcessingService
from models import MarkSheetData, ValidationResponse, TranscriptData, CertificateData, BulkProcessingResponse
from job_store import create_job, get_job
from worker import process_zip_job
import zipfile
import io
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/validate", response_model=ValidationResponse)
async def validate_document(file: UploadFile = File()):
    """
    Step 1: Lightweight quality check
    """
    try:
        file_bytes = await file.read()
        # Non-blocking call to the validation service
        validation = await ProcessingService.validate_document(file_bytes, file.filename)
        return validation
    except Exception as e:
        logger.exception("Validation route failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/marksheet_data_extraction", response_model=MarkSheetData, response_model_exclude_none=True)
async def extract_document(file: UploadFile = File()):
    """
    Step 2: Full Extraction pipeline
    """
    try:
        file_bytes = await file.read()
        
        # 1. Automatic Validation
        validation = await ProcessingService.validate_document(file_bytes, file.filename)
        if not validation.is_valid:
            logger.warning(f"Quality Check Failed: {validation.instruction}")
            raise HTTPException(status_code=400, detail=validation.instruction)
            
        # 2. Proceed to Extraction
        if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
            logger.info("Extracting data from PDF")
            img_list, raw_text = await ProcessingService.process_pdf_pages(file_bytes)
            if img_list:
                processing_image = img_list
                ocr_text = raw_text
        else:
            processing_image = [file_bytes]
            ocr_text = ""
            page_results = []
        # ocr_source = processing_image[0] if isinstance(processing_image, list) else processing_image
            for i, img in enumerate(processing_image):
                logger.info(f"Processing OCR for page {i+1}/{len(processing_image)}...")
                page_text = await ProcessingService.run_ocr(img)
                page_results.append(page_text)
            ocr_text = "\n\n".join(page_results)
        if len(ocr_text) < 60 or not any(c.isalpha() for c in ocr_text):
            logger.info(f"Scanned PDF detected. Running OCR on ALL {len(processing_image)} pages...")
            page_results = []
            # Loop through every extracted page image
            for i, img in enumerate(processing_image):
                logger.info(f"Processing OCR for page {i+1}/{len(processing_image)}...")
                page_text = await ProcessingService.run_ocr(img)
                page_results.append(page_text)
            
            # Combine all pages into one full text block
            ocr_text = "\n\n".join(page_results)
            
        logger.info(f"--- RAW OCR TEXT START ---\n{ocr_text}\n--- RAW OCR TEXT END ---")
            
        # 3. AI Structured Extraction
        doc_type = await ProcessingService.classify_document(ocr_text)
        logger.info(f"Step 2 (Classification): Identified as {doc_type}")
        
        if "marksheet" not in doc_type and "unknown" not in doc_type:
            logger.warning(f"Classification Mismatch: Expected marksheet, found {doc_type}. Proceeding anyway.")
            # raise HTTPException(status_code=400, detail="Please upload the correct document.")
            
        structured_dict = await ProcessingService.extract_with_ai(processing_image, ocr_text)
        
        # 4. Final Object Construction
        # Since extract_with_ai returns a MarkSheetCollection dict containing a 'marksheets' list,
        # we extract the first marksheet for single-document ingestion routes.
        marksheets = structured_dict.get("marksheets", [])
        if not marksheets:
            raise HTTPException(status_code=500, detail="No marksheet data extracted from the document.")
        
        return MarkSheetData(**marksheets[0])
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Extraction route failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/certificate", response_model=CertificateData, response_model_exclude_none=True)
async def extract_certificate(file: UploadFile = File()):
    """
    Step 2: Certificate Extraction pipeline
    """
    try:
        file_bytes = await file.read()

        # 1. Automatic Validation
        validation = await ProcessingService.validate_document(file_bytes, file.filename)
        if not validation.is_valid:
            logger.warning(f"Quality Check Failed: {validation.instruction}")
            raise HTTPException(status_code=400, detail=validation.instruction)

        # 2. Proceed to Extraction
        ocr_text = ""
        processing_image = file_bytes
        
        if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
            img_list, raw_text = await ProcessingService.process_pdf_pages(file_bytes, max_pages=1)
            if img_list:
                processing_image = img_list
                ocr_text = raw_text
        
        # ocr_source = processing_image[0] if isinstance(processing_image, list) else processing_image
        # if len(ocr_text) < 60 or not any(c.isalpha() for c in ocr_text):
        #     ocr_text = await ProcessingService.run_ocr(ocr_source)
        if len(ocr_text) < 60 or not any(c.isalpha() for c in ocr_text):
            logger.info(f"Scanned PDF detected. Running OCR on ALL {len(processing_image)} pages...")
            page_results = []
            # Loop through every extracted page image
            for i, img in enumerate(processing_image):
                logger.info(f"Processing OCR for page {i+1}/{len(processing_image)}...")
                page_text = await ProcessingService.run_ocr(img)
                page_results.append(page_text)
            
            # Combine all pages into one full text block
            ocr_text = "\n\n".join(page_results)

        logger.info(f"--- RAW CERTIFICATE OCR TEXT START ---\n{ocr_text}\n--- RAW CERTIFICATE OCR TEXT END ---")

        # 3. Classification Gate
        doc_type = await ProcessingService.classify_document(ocr_text)
        logger.info(f"Step 2 (Classification): Identified as {doc_type}")
        if "certificate" not in doc_type and "unknown" not in doc_type:
            logger.warning(f"Classification Mismatch: Expected certificate, found {doc_type}. Proceeding anyway.")
            # raise HTTPException(status_code=400, detail="Please upload the correct document.")

        # 4. Extraction
        structured_dict = await ProcessingService.extract_certificate_with_ai(processing_image, ocr_text)
        
        return CertificateData(**structured_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Certificate extraction route failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transcript", response_model=TranscriptData, response_model_exclude_none=True)
async def extract_transcript(file: UploadFile = File()):
    """
    Step 2: Transcript Extraction pipeline (Hierarchical)
    """
    try:
        file_bytes = await file.read()

        # 1. Automatic Validation
        validation = await ProcessingService.validate_document(file_bytes, file.filename)
        if not validation.is_valid:
            logger.warning(f"Quality Check Failed: {validation.instruction}")
            raise HTTPException(status_code=400, detail=validation.instruction)

        # 2. Proceed to Extraction
        ocr_text = ""
        processing_image = file_bytes
        
        if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
            logger.info("Extracting data from Transcript PDF (up to 50 pages)")
            img_list, raw_text = await ProcessingService.process_pdf_pages(file_bytes, max_pages=50)
            if img_list:
                processing_image = img_list
                ocr_text = raw_text
        
        # ocr_source = processing_image[0] if isinstance(processing_image, list) else processing_image
        # if len(ocr_text) < 60 or not any(c.isalpha() for c in ocr_text):
        #     logger.info("Scanned Transcript PDF detected, running full OCR")
        #     ocr_text = await ProcessingService.run_ocr(ocr_source)
        # The logic now handles all pages instead of just index [0]
        if len(ocr_text) < 60 or not any(c.isalpha() for c in ocr_text):
            logger.info(f"Scanned PDF detected. Running OCR on ALL {len(processing_image)} pages...")
            page_results = []
            # Loop through every extracted page image
            for i, img in enumerate(processing_image):
                logger.info(f"Processing OCR for page {i+1}/{len(processing_image)}...")
                page_text = await ProcessingService.run_ocr(img)
                page_results.append(page_text)
            
            # Combine all pages into one full text block
            ocr_text = "\n\n".join(page_results)

            
        logger.info(f"--- RAW TRANSCRIPT OCR TEXT START ---\n{ocr_text}\n--- RAW TRANSCRIPT OCR TEXT END ---")
            
        # 3. Classification Gate
        doc_type = await ProcessingService.classify_document(ocr_text)
        logger.info(f"Step 2 (Classification): Identified as {doc_type}")
        if "transcript" not in doc_type and "unknown" not in doc_type:
            logger.warning(f"Classification Mismatch: Expected transcript, found {doc_type}. Proceeding anyway.")
            # raise HTTPException(status_code=400, detail="Please upload the correct document.")

        # 4. AI Hierarchical Extraction
        structured_dict = await ProcessingService.extract_transcript_with_ai(processing_image, ocr_text)
        
        return TranscriptData(**structured_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Transcript extraction route failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk_process_zip")
async def bulk_process_zip(file: UploadFile = File()):
    """
    Upload a ZIP of PDFs and STREAM the results back in real-time.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported.")
    
    try:
        zip_bytes = await file.read()
        return StreamingResponse(
            ProcessingService.bulk_process_zip_streaming(zip_bytes),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        logger.exception("Bulk ZIP processing failed")
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# ASYNC JOB QUEUE ROUTES  (Production-grade, rate-limit safe)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/bulk_process_zip_async", status_code=202)
async def bulk_process_zip_async(file: UploadFile = File()):
    """
    Accepts a ZIP of PDFs, immediately returns a job_id (202 Accepted).
    Processing happens in the background with strict rate-limiting.
    Poll GET /api/v1/job/{job_id} for real-time progress.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported.")

    zip_bytes = await file.read()

    # Peek inside the ZIP to get filenames for the job record
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Use os.path.basename to normalize — worker.py also uses basename
            # This prevents key mismatches like "folder/file.pdf" vs "file.pdf"
            raw_names = [
                name for name in zf.namelist()
                if name.lower().endswith(".pdf") and not name.startswith("__MACOSX")
            ]
            pdf_names = sorted(set(os.path.basename(name) for name in raw_names))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP.")

    if not pdf_names:
        raise HTTPException(status_code=400, detail="No PDF files found inside the ZIP.")

    # Create job record (returns instantly)
    job_id = create_job(total_files=len(pdf_names), filenames=pdf_names)
    logger.info(f"[Route] Created job {job_id} for {len(pdf_names)} PDFs. Launching background worker...")

    # Fire-and-forget — does not block the HTTP response
    asyncio.create_task(process_zip_job(job_id, zip_bytes))

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "pending",
            "total_files": len(pdf_names),
            "filenames": pdf_names,
            "message": f"Job accepted. {len(pdf_names)} PDFs queued for processing.",
            "poll_url": f"/api/v1/job/{job_id}",
        }
    )


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll this endpoint to get real-time progress of a bulk processing job.
    Returns overall status, per-file states, and completed results.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JSONResponse(content=job)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY STREAMING ROUTE (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/log_merkle")
async def log_merkle(payload: dict):
    """
    Utility endpoint to log Merkle Root construction to the backend terminal.
    """
    root = payload.get("root")
    count = payload.get("leaves_count")
    print(f"\n" + "="*60)
    print(f" [TERMINAL] MERKLE ROOT CONSTRUCTED")
    print(f" ROOT:  {root}")
    print(f" LEAVES: {count}")
    print("="*60 + "\n")
    logger.info(f"Merkle Root Constructed: {root} ({count} leaves)")
    return {"status": "logged"}
