from fastapi import APIRouter, File, UploadFile, HTTPException
from service import ProcessingService
from models import MarkSheetData, ValidationResponse
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

@router.post("/extract", response_model=MarkSheetData)
async def extract_document(file: UploadFile = File()):
    """
    Step 2: Full Extraction pipeline
    """
    try:
        file_bytes = await file.read()
        
        # 1. Automatic Validation (Awaited)
        validation = await ProcessingService.validate_document(file_bytes, file.filename)
        if not validation.is_valid:
            logger.warning(f"Quality Check Failed: {validation.instruction}")
            raise HTTPException(status_code=400, detail=validation.instruction)
            
        # 2. Proceed to Extraction (Non-blocking)
        ocr_text = ""
        processing_image = file_bytes
        
        # Handle PDF vs Image
        if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
            logger.info("Extracting data from PDF")
            img_list, raw_text = await ProcessingService.process_pdf_pages(file_bytes)
            if img_list:
                processing_image = img_list
                ocr_text = raw_text
        
        # Secondary OCR if text is sparse
        ocr_source = processing_image[0] if isinstance(processing_image, list) else processing_image
        if len(ocr_text) < 50:
            logger.info("Sparse text detected, running async OCR")
            ocr_text = await ProcessingService.run_ocr(ocr_source)
            
        # 3. AI Structured Extraction (Awaited)
        structured_dict = await ProcessingService.extract_with_ai(processing_image, ocr_text)
        
        # 4. Final Security Hash (Awaited)
        norm_text = ProcessingService.normalize_for_hash(structured_dict)
        structured_dict["merkle_hash"] = await ProcessingService.generate_keccak256(norm_text)
        
        return MarkSheetData(**structured_dict)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Extraction route failed")
        raise HTTPException(status_code=500, detail=str(e))
