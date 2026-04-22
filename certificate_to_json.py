import os
import json
import re
import requests
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from PIL import Image
import io
import uvicorn
import logging
import psutil
import signal
import fitz
from web3 import Web3
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CEREBRAS_API_KEY=os.getenv("CEREBRAS_API_KEY", "")

app = FastAPI(title="Certificate AI Parser")

# DATA MODELS
class CertificateData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    certificate_no: str = Field(..., description="Certificate Number")
    no: str = Field(..., description="Reference Number")
    # university: Optional[str] = Field(None, description="Issuing University Name")
    name: str = Field(..., description="Student Name")
    degree: str = Field(..., description="Degree conferred")
    ogpa: Optional[str] = Field(None, description="Overall Grade Point Average")
    year: Optional[str] = Field(None, description="Academic Session Year")
    date: str = Field(..., description="Issue Date")
    # class_division: Optional[str] = Field(None, description="Class/Division obtained")
    # merkle_hash: Optional[str] = Field(None, description="Keccak-256 Verification Hash")

# PORT UTILITIES
def force_free_port(port: int):
    """Detects and kills any process using the specified port."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            if conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    logger.info(f"Port {port} is occupied by {p.name()} (PID: {conn.pid}). Terminating...")
                    p.terminate()
                    p.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    try:
                        p.kill()
                    except:
                        pass

# HELPER FUNCTIONS
def compress_image(image_bytes: bytes, max_kb: int = 1000):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if len(image_bytes) <= max_kb * 1024:
        return image_bytes
    quality = 90
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

def run_ocr(image_bytes: bytes):
    compressed_bytes = compress_image(image_bytes)
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", compressed_bytes, "image/jpeg")}
    data = {"apikey": OCR_API_KEY, "language": "eng", "isTable": True, "OCREngine": 2}
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        if result.get("OCRExitCode") != 1:
            return f"[OCR Error: {result.get('ErrorMessage')}]"
        return result["ParsedResults"][0]["ParsedText"]
    except Exception as e:
        return f"[OCR Exception: {e}]"

def run_ocr_on_pdf(pdf_bytes: bytes):
    if len(pdf_bytes) > 1024 * 1024:
        logger.info("PDF > 1MB. Switching to page-by-page OCR...")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = []
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
            all_text.append(run_ocr(pix.tobytes("jpg")))
        doc.close()
        return "\n\n".join(all_text)

    url = "https://api.ocr.space/parse/image"
    files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
    data = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "isTable": True,
        "OCREngine": 2
    }
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        if result.get("OCRExitCode") != 1:
            return run_ocr_on_pdf(pdf_bytes + b'\0') 
        return "\n\n".join([res.get("ParsedText", "") for res in result.get("ParsedResults", [])])
    except Exception as e:
        return f"OCR Error: {e}"

# HASHING UTILITIES
def build_canonical_payload(data: dict) -> str:
    """Builds a canonical JSON string for the certificate structure."""
    payload = OrderedDict([
        ("certificate_no", str(data.get("certificate_no", ""))),
        ("no", str(data.get("no", ""))),
        ("university", str(data.get("university", ""))),
        ("name", str(data.get("name", ""))),
        ("degree", str(data.get("degree", ""))),
        ("ogpa", str(data.get("ogpa", ""))),
        ("year", str(data.get("year", ""))),
        ("date", str(data.get("date", ""))),
        # # ("class_division", str(data.get("class_division", "")))
    ])
    return json.dumps(payload, separators=(',', ':'))

def generate_keccak256(text: str):
    hash_bytes = Web3.keccak(text=text)
    return Web3.to_hex(hash_bytes)

def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = min(len(doc), max_pages)
        all_text, all_images = [], []
        for i in range(num_pages):
            all_text.append(doc[i].get_text().strip())
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
            all_images.append(pix.tobytes("jpg"))
        doc.close()
        return all_images, "\n\n".join(all_text)
    except Exception as e:
        logger.error(f"PDF failed: {e}")
        return [], ""

def generate_with_cerebras(prompt: str):
    try:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama3.1-8b",
            "messages": [{"role": "system", "content": "Extract certificate data as JSON."}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30).json()
        content = resp["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group()) if match else json.loads(content)
    except Exception as e:
        logger.warning(f"Cerebras failed: {e}")
        return None

def generate_with_gemini(image_data, prompt: str):
    from google import genai
    from google.genai import types
    import time
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [prompt]
    images = image_data if isinstance(image_data, list) else [image_data]
    for img_bytes in images:
        contents.append(Image.open(io.BytesIO(img_bytes)))
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CertificateData,
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(2**(attempt+1))
                continue
            return None
    return None

def generate_structured_data(image_data, ocr_text: str):
    prompt = f"""
Extract certificate fields exactly as shown in the document:
1. **certificate_no**: Top right numeric sequence.
2. **no**: Reference number (e.g., 'No.XIII/210/2018').
3. **university**: Full university name from header.
4. **name**: Student's full name.
5. **degree**: Full degree title (e.g. 'Bachelor of Technology').
6. **ogpa**: Overall grade point average.
7. **year**: Academic session year (e.g. '2016-17').
8. **date**: Issue date at the bottom.

OCR TEXT:
{ocr_text}

Return ONLY valid JSON matching CertificateData schema.
"""
    if GEMINI_API_KEY:
        logger.info("Running Gemini Extraction...")
        res = generate_with_gemini(image_data, prompt)
        if res: return res

    if CEREBRAS_API_KEY:
        logger.info("Running Cerebras Fallback...")
        res = generate_with_cerebras(prompt)
        if res: return res

    raise ValueError("Extraction models failed.")

@app.post("/parse-certificate", response_model=CertificateData)
async def parse_certificate(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if file.content_type == "application/pdf":
            img_list, ocr_text = process_pdf_pages(file_bytes, max_pages=1)
            if not ocr_text: ocr_text = run_ocr_on_pdf(file_bytes)
        else:
            img_list = [file_bytes]
            ocr_text = run_ocr(file_bytes)

        structured_data = generate_structured_data(img_list, ocr_text)
        
        # Canonical Hash for Blockchain
        canonical_json = build_canonical_payload(structured_data)
        structured_data["merkle_hash"] = generate_keccak256(canonical_json)
        
        return CertificateData(**structured_data)
    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Certificate Verifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #fbbf24; --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); display: flex; align-items: center; justify-content: center; margin: 0; padding: 20px; min-height: 100vh; }
        .box { width: 100%; max-width: 800px; background: var(--card); border-radius: 24px; padding: 40px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); }
        .dropzone { border: 2px dashed #475569; border-radius: 16px; padding: 60px; text-align: center; cursor: pointer; transition: 0.3s; background: rgba(51,65,85,0.3); }
        .dropzone:hover { border-color: var(--primary); }
        .btn { background: var(--primary); color: #000; border: none; padding: 16px; border-radius: 12px; font-weight: 700; cursor: pointer; width: 100%; margin-top: 20px; }
        #results { margin-top: 40px; display: none; }
        .cert-card { background: #fff; color: #1e293b; border-radius: 12px; padding: 40px; border: 8px double #e2e8f0; }
        .univ-name { font-size: 1.5em; font-weight: 800; color: #1e3a8a; text-align: center; text-transform: uppercase; }
        .field { margin: 20px 0; }
        .label { font-size: 0.75em; text-transform: uppercase; color: #64748b; font-weight: 600; }
        .val { font-size: 1.3em; font-weight: 700; color: #0f172a; }
        .hash-box { background: #f8fafc; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.8em; margin-top: 20px; word-break: break-all; color: #64748b; }
        #jsonLogs { background: #000; color: #0f0; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.8em; margin-top: 20px; max-height: 300px; overflow: auto; }
    </style>
</head>
<body>
    <div class="box">
        <h1 style="text-align: center; color: var(--primary);">Certificate Intelligence</h1>
        <form id="pForm">
            <div class="dropzone" onclick="document.getElementById('f').click()">
                <span id="txt">Click or Drop Certificate (PDF/Image)</span>
                <input type="file" id="f" style="display:none" onchange="document.getElementById('txt').innerText=this.files[0].name">
            </div>
            <button type="submit" class="btn">Verify Certificate</button>
        </form>
        <div id="results">
            <div class="cert-card">
                <div id="univ" class="univ-name"></div>
                <div class="field"><div class="label">Degree Issued</div><div id="degree" class="val"></div></div>
                <div class="field"><div class="label">Student Name</div><div id="name" class="val"></div></div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div><div class="label">OGPA</div><div id="ogpa" class="val"></div></div>
                    <div><div class="label">Session</div><div id="year" class="val"></div></div>
                </div>
                <div class="hash-box"><strong>HASH:</strong> <span id="h"></span></div>
            </div>
            <div id="jsonLogs"></div>
        </div>
    </div>
    <script>
        document.getElementById('pForm').onsubmit = async (e) => {
            e.preventDefault();
            const file = document.getElementById('f').files[0];
            if(!file) return;
            const fd = new FormData(); fd.append('file', file);
            try {
                const res = await fetch('/parse-certificate', {method:'POST', body:fd});
                const d = await res.json();
                document.getElementById('jsonLogs').innerText = JSON.stringify(d, null, 2);
                if(res.ok) {
                    document.getElementById('univ').innerText = d.university || 'N/A';
                    document.getElementById('degree').innerText = d.degree;
                    document.getElementById('name').innerText = d.name;
                    document.getElementById('ogpa').innerText = d.ogpa || 'N/A';
                    document.getElementById('year').innerText = d.year || 'N/A';
                    document.getElementById('h').innerText = d.merkle_hash;
                    document.getElementById('results').style.display = 'block';
                }
            } catch(er) { alert('Failed: ' + er.msg); }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    force_free_port(8081)
    uvicorn.run(app, host="0.0.0.0", port=8081)