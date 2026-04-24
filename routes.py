from fastapi import APIRouter, File, UploadFile, HTTPException
from service import ProcessingService
from models import MarkSheetData, ValidationResponse, TranscriptData, CertificateData
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

@router.post("/marksheet_data_extraction", response_model=MarkSheetData)
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
        
        if doc_type != "marksheet" and doc_type != "unknown":
            logger.warning(f"Classification Mismatch: Expected marksheet, found {doc_type}. Proceeding anyway.")
            # raise HTTPException(status_code=400, detail="Please upload the correct document.")
            
        structured_dict = await ProcessingService.extract_with_ai(processing_image, ocr_text)
        
        # 4. Final Object Construction
        return MarkSheetData(**structured_dict)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Extraction route failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/certificate", response_model=CertificateData)
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
        if doc_type != "certificate" and doc_type != "unknown":
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

@router.post("/transcript", response_model=TranscriptData)
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
            logger.info("Extracting data from Transcript PDF (up to 10 pages)")
            img_list, raw_text = await ProcessingService.process_pdf_pages(file_bytes, max_pages=10)
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
        if doc_type != "transcript" and doc_type != "unknown":
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
