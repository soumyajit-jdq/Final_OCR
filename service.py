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
from Crypto.Hash import keccak
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
    def normalize_for_hash(data: dict):
        """Creates a deterministic string for hashing."""
        parts = []
        parts.append(str(data.get("university", "")).strip())
        parts.append(str(data.get("college", "")).strip())
        parts.append(str(data.get("name", "")).strip())
        parts.append(str(data.get("registration_no", "")).strip())
        parts.append(str(data.get("semester", "")).strip())
        
        subs = data.get("subjects", [])
        sorted_subs = sorted(subs, key=lambda x: x.get("code", ""))
        for s in sorted_subs:
            parts.append(f"{s.get('code')}{s.get('grade')}{s.get('credits')}")
        
        parts.append(str(data.get("gpa", "")).strip())
        return "|".join(parts).lower()

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
        def sync_hash():
            k = keccak.new(digest_bits=256)
            k.update(text.encode('utf-8'))
            return k.hexdigest()
        return await anyio.to_thread.run_sync(sync_hash)

    @staticmethod
    async def extract_with_ai(image_data, ocr_text: str):
        """Handles the AI generation logic"""
        primary_image_bytes = image_data[0] if isinstance(image_data, list) else image_data
        base64_img = ProcessingService.encode_image(primary_image_bytes)
        
        prompt = f"Extract marksheet data into JSON format accurately. OCR Text: {ocr_text}"

        # 1. Try Gemini (Async Client)
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
                        temperature=0.0
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                logger.warning(f"Gemini Async failed: {e}")

        # 2. Fallback to OpenRouter (Reasoning Models) using httpx
        models = ["nvidia/nemotron-nano-12b-v2-vl:free", "liquid/lfm-2.5-1.2b-thinking:free"]
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
