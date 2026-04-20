import os
import json
import re
import httpx
import base64
import io
import logging
import fitz
import anyio
from PIL import Image
from web3 import Web3
from collections import OrderedDict
from dotenv import load_dotenv
from models import MarkSheetData, ValidationResponse
from preprocessing import validate_image_quality

load_dotenv()

# Setup Logging
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")

class ProcessingService:
    @staticmethod
    async def validate_document(file_bytes: bytes, filename: str) -> ValidationResponse:
        """Runs the preprocessing quality checks in a separate thread to avoid blocking."""
        def sync_validate():
            is_valid, msg = validate_image_quality(file_bytes, filename)
            file_type = "PDF" if file_bytes.startswith(b"%PDF") else "Image"
            return is_valid, msg, file_type
            
        is_valid, msg, file_type = await anyio.to_thread.run_sync(sync_validate)
        return ValidationResponse(is_valid=is_valid, instruction=msg, file_type=file_type)

    @staticmethod
    async def compress_image(image_bytes: bytes, max_kb: int = 1000):
        """Image compression is CPU bound, running in thread."""
        def sync_compress():
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            if len(image_bytes) <= max_kb * 1024:
                return image_bytes
            quality = 90
            buffer = io.BytesIO()
            while quality > 10:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                if len(buffer.getvalue()) <= max_kb * 1024:
                    return buffer.getvalue()
                quality -= 10
            img.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=20)
            return buffer.getvalue()
            
        return await anyio.to_thread.run_sync(sync_compress)

    @staticmethod
    async def run_ocr(image_bytes: bytes):
        """Asynchronous call to OCR.space API."""
        compressed_bytes = await ProcessingService.compress_image(image_bytes)
        url = "https://api.ocr.space/parse/image"
        
        # Prepare multipart/form-data
        files = {"file": ("image.jpg", compressed_bytes, "image/jpeg")}
        data = {"apikey": OCR_API_KEY, "language": "eng", "isTable": True, "OCREngine": 2}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, files=files, data=data, timeout=60)
                result = response.json()
                if result.get("OCRExitCode") != 1:
                    return f"OCR Failed: {result.get('ErrorMessage')}"
                return result["ParsedResults"][0]["ParsedText"]
            except Exception as e:
                return f"OCR Error: {e}"

    @staticmethod
    def encode_image(image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    @staticmethod
    def build_canonical_payload(data: dict) -> str:
        """
        Builds a canonical JSON string with STRICT key ordering:
          registration_no -> name -> gpa -> subjects
        Each subject maintains: code -> title -> credit_points -> grade
        """
        subjects = []
        for s in data.get("subjects", []):
            ordered_subject = OrderedDict([
                ("code", str(s.get("code", ""))),
                ("title", str(s.get("title", ""))),
                ("credit_points", str(s.get("credit_points", ""))),
                # ("grade", str(s.get("grade", "")))
            ])
            subjects.append(ordered_subject)

        payload = OrderedDict([
            ("registration_no", str(data.get("registration_no", ""))),
            ("name", str(data.get("name", ""))),
            ("gpa", str(data.get("gpa", ""))),
            ("subjects", subjects)
        ])

        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    async def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
        """PDF processing is CPU intensive, running in thread pool."""
        def sync_pdf_process():
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages = min(len(doc), max_pages)
                if num_pages == 0:
                    return [], ""
                
                all_text = []
                all_images = []
                for i in range(num_pages):
                    page = doc[i]
                    all_text.append(page.get_text().strip())
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    all_images.append(pix.tobytes("jpg"))
                doc.close()
                return all_images, "\n\n".join(all_text)
            except Exception as e:
                logger.error(f"PDF processing failed: {e}")
                return [], ""
        
        return await anyio.to_thread.run_sync(sync_pdf_process)

    @staticmethod
    async def generate_keccak256(text: str):
        """Generates an Ethereum-standard Keccak-256 hash using Web3.py."""
        def sync_hash():
            hash_bytes = Web3.keccak(text=text)
            return Web3.to_hex(hash_bytes)
        return await anyio.to_thread.run_sync(sync_hash)

    @staticmethod
    async def generate_with_cerebras(prompt: str):
        """High-speed text-only extraction using Cerebras."""
        try:
            url = "https://api.cerebras.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # --- PASS 1: Initial Extraction ---
            payload_1 = {
                "model": "",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a VERBATIM marksheet parser. Extract exactly as visible in OCR. Return ONLY JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            }
            async with httpx.AsyncClient() as client:
                resp_1_raw = await client.post(url, headers=headers, json=payload_1, timeout=30)
                resp_1 = resp_1_raw.json()
                
            if "choices" not in resp_1:
                return None
            initial_json = resp_1["choices"][0]["message"]["content"]
            
            # --- PASS 2: Self-Correction Loop ---
            correction_system_prompt = (
                "You are a character-level QA auditor. Compare the provided JSON against the Raw OCR Text.\n"
                "STRICT AUDIT RULE: Check if 'credit_points' contains the Total Credit Points (usually 10-20) and 'grade' contains the Grade Point.\n"
                "If you see a shift (e.g., 'credit_points' has 2 instead of 17.6), fix it immediately.\n"
                "Identify and fix any truncated words or misaligned columns.\n"
                "Return ONLY the corrected JSON object."
            )
            correction_user_prompt = f"RAW OCR TEXT:\n{prompt}\n\nINITIAL JSON TO CORRECT:\n{initial_json}"
            
            payload_2 = {
                "model": "",
                "messages": [
                    {"role": "system", "content": correction_system_prompt},
                    {"role": "user", "content": correction_user_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            }
            async with httpx.AsyncClient() as client:
                resp_2_raw = await client.post(url, headers=headers, json=payload_2, timeout=30)
                resp_2 = resp_2_raw.json()
            
            if "choices" in resp_2:
                corrected_content = resp_2["choices"][0]["message"]["content"]
                logger.info("Self-Correction loop completed.")
                return json.loads(corrected_content)
            
            return json.loads(initial_json)
        except Exception as e:
            logger.warning(f"Cerebras extraction/correction failed: {e}")
            return None

    @staticmethod
    async def extract_with_ai(image_data, ocr_text: str):
        """Handles the AI generation logic"""
        primary_image_bytes = image_data[0] if isinstance(image_data, list) else image_data
        base64_img = ProcessingService.encode_image(primary_image_bytes)
        
        prompt = f"""
You are an expert VERBATIM marksheet parser. Extract ALL details from the provided OCR text and image with 100% character-level precision.
STRICT RULE: Do not correct spelling, do not format dates, do not normalize case, and do not truncate or shorten words. Extract text EXACTLY as it appears.
The document may contain trilingual text (English, Gujarati, Hindi). Extract exactly as visible.

#### COLUMN MAPPING RULE ####
A typical row looks like: [SR NO] [COURSE CATEGORY] [COURSE CODE] [TITLE] [CREDIT HOURS] [GRADE POINTS] [CREDIT POINTS]
Example: "1 ALLIED ABM 517 AGRICULTURAL MARKETING MANAGEMENT 2 8.8 17.6"
- "code": "ABM 517"
- "title": "AGRICULTURAL MARKETING MANAGEMENT"
- "credit_points": "17.6" (This is the total Credit Points. Always use the last value.)
- "grade": "8.8" (This is the Grade Points. Always use this value.)
- DO NOT use the middle number (2) which is the credit hours.

OCR TEXT:
{ocr_text}

JSON FORMAT:
{{
  "registration_no": "Enrollment/Reg No",
  "name": "Full Name",
  "gpa": "GPA/SGPA/CGPA",
  "subjects": [
    {{
      "code": "Code", 
      "title": "Subject Title", 
      "credit_points": "Total Credit Points ONLY", 
    }}
  ]
}}
Return ONLY the JSON.
"""

        # 1. Try Cerebras First
        if CEREBRAS_API_KEY:
            logger.info("Attempting primary extraction with Cerebras")
            cerebras_result = await ProcessingService.generate_with_cerebras(prompt)
            if cerebras_result:
                return cerebras_result

        # 2. Try Gemini (Async Client)
        if GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=GEMINI_API_KEY)
                
                # Convert bytes to PIL for Gemini
                def bytes_to_pil(b_list):
                    if isinstance(b_list, list):
                        return [Image.open(io.BytesIO(b)) for b in b_list]
                    return Image.open(io.BytesIO(b_list))
                
                pil_images = await anyio.to_thread.run_sync(bytes_to_pil, image_data)
                
                contents = [prompt]
                if isinstance(pil_images, list):
                    contents.extend(pil_images)
                else:
                    contents.append(pil_images)
                
                # Use the asynchronous generation capability
                response = await client.aio.models.generate_content(
                    model='gemini-3.1-flash-lite-preview',
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=MarkSheetData,
                        temperature=0.1
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                logger.warning(f"Gemini Async failed: {e}")

        # 3. Fallback to OpenRouter (Reasoning Models) using httpx
        models = [
            "nvidia/nemotron-nano-12b-v2-vl:free", 
            "liquid/lfm-2.5-1.2b-thinking:free",
            "google/gemma-4-31b-it:free"
        ]
        async with httpx.AsyncClient() as client:
            for model in models:
                try:
                    logger.info(f"Attempting async extraction with fallback {model}...")
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]}],
                        "reasoning": {"enabled": True}
                    }
                    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                    res = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
                    res_json = res.json()
                    
                    if "choices" in res_json:
                        raw_content = res_json["choices"][0]["message"]["content"]
                        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                        if match:
                            return json.loads(match.group())
                except Exception as e:
                    logger.warning(f"{model} failed: {e}")
        
        raise ValueError("AI Extraction failed across all async paths.")
