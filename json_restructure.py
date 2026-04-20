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

app = FastAPI(title="MarkSheet AI Parser (Reasoning Edition)")

# DATA MODELS
class Subject(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    code: str = Field(..., description="Course code")
    title: str = Field(..., description="Course title")
    credit_points: str = Field(..., description="Total Credit Points")
    # grade: str = Field(..., description="Grade Points (e.g., '8.8')")

class MarkSheetData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    registration_no: str = Field(..., description="Student Registration/Enrollment Number")
    name: str = Field(..., description="Student Name")
    gpa: Optional[str] = Field(None, description="Grade Point Average")
    subjects: List[Subject]
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
                        logger.warning(f"Graceful termination failed for PID {conn.pid}. Forcing...")
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
            return f"OCR Failed: {result.get('ErrorMessage')}"
        return result["ParsedResults"][0]["ParsedText"]
    except Exception as e:
        return f"OCR Error: {e}"

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# HASHING UTILITIES
def build_canonical_payload(data: dict) -> str:
    """
    Builds a canonical JSON string with STRICT key ordering:
      registration_no -> name -> gpa -> subjects
    Each subject maintains: code -> title -> credits -> grade
    
    Uses separators=(',', ':') to produce compact JSON 
    identical to JavaScript's JSON.stringify().
    """
    # Build subjects list preserving key order
    subjects = []
    for s in data.get("subjects", []):
        ordered_subject = OrderedDict([
            ("code", str(s.get("code", ""))),
            ("title", str(s.get("title", ""))),
            ("credit_points", str(s.get("credit_points", ""))),
            # ("grade", str(s.get("grade", "")))
        ])
        subjects.append(ordered_subject)

    # Build top-level payload with strict key order
    payload = OrderedDict([
        ("registration_no", str(data.get("registration_no", ""))),
        ("name", str(data.get("name", ""))),
        ("gpa", str(data.get("gpa", ""))),
        ("subjects", subjects)
    ])

    # Compact JSON: no spaces, matches JavaScript's JSON.stringify()
    return json.dumps(payload, separators=(',', ':'))

def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
    """Converts up to max_pages of a PDF to OCR-ready images and extracts text."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = min(len(doc), max_pages)
        if num_pages == 0:
            return [], ""
        
        all_text = []
        all_images = []
        
        for i in range(num_pages):
            page = doc[i]
            # Extract text
            all_text.append(page.get_text().strip())
            
            # Convert to high-res image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            all_images.append(pix.tobytes("jpg"))
            
        doc.close()
        return all_images, "\n\n".join(all_text)
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return [], ""

def generate_keccak256(text: str):
    """Generates an Ethereum-standard Keccak-256 hash using Web3.py."""
    hash_bytes = Web3.keccak(text=text)
    return Web3.to_hex(hash_bytes)

def generate_with_cerebras(prompt: str):
    """High-speed text-only extraction using Cerebras."""
    try:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # --- PASS 1: Initial Extraction ---
        payload_1 = {
            # "model": "llama3.1-8b",
            # "model": "qwen-3-235b-a22b-instruct-2507",
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
        resp_1 = requests.post(url, headers=headers, json=payload_1, timeout=30).json()
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
            # "model": "llama3.1-8b",
            "messages": [
                {"role": "system", "content": correction_system_prompt},
                {"role": "user", "content": correction_user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        resp_2 = requests.post(url, headers=headers, json=payload_2, timeout=30).json()
        
        if "choices" in resp_2:
            corrected_content = resp_2["choices"][0]["message"]["content"]
            logger.info("Self-Correction loop completed.")
            return json.loads(corrected_content)
        
        return json.loads(initial_json)
    except Exception as e:
        logger.warning(f"Cerebras extraction/correction failed: {e}")
        return None

def generate_with_gemini(image_data, prompt: str):
    """Fallback to Gemini for high-precision vision parsing. image_data can be bytes or List[bytes]."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        contents = [prompt]
        if isinstance(image_data, list):
            for img_bytes in image_data:
                contents.append(Image.open(io.BytesIO(img_bytes)))
        else:
            contents.append(Image.open(io.BytesIO(image_data)))
        
        response = client.models.generate_content(
            # model='gemini-1.5-flash-lite',
            model='gemini-3.1-flash-lite-preview',
            # model='gemini-1.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MarkSheetData,
                temperature=0.1
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.warning(f"Gemini extraction failed: {e}")
        return None

def generate_structured_data(image_data, ocr_text: str):
    # If it's a list (multi-page PDF), we use the first page as the primary for fallback models
    # but Gemini will get all pages.
    primary_image_bytes = image_data[0] if isinstance(image_data, list) else image_data
    base64_image = encode_image(primary_image_bytes)
    
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

    # Try Cerebras First (Fast text-only extraction)
    if CEREBRAS_API_KEY:
        logger.info("Attempting primary extraction with Cerebras")
        cerebras_result = generate_with_cerebras(prompt)
        if cerebras_result:
            return cerebras_result

    # Try Gemini First (Handles multiple images natively and has better literal precision)
    if GEMINI_API_KEY:
        logger.info("Attempting secondary extraction with Gemini")
        gemini_result = generate_with_gemini(image_data, prompt)
        if gemini_result:
            return gemini_result

    # Using Nvidia Vision-Language model as secondary (requested by user)
    models_to_try = [
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "google/gemma-4-31b-it:free",
        "liquid/lfm-2.5-1.2b-thinking:free",
        "qwen/qwen3-next-80b-a3b-instruct:free"
    ]
    
    for model_id in models_to_try:
        try:
            logger.info(f"Attempting reasoning-enabled extraction with {model_id}...")
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                "reasoning": {"enabled": True}
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
            res_json = response.json()
            
            if "choices" not in res_json:
                logger.warning(f"Model {model_id} error: {res_json.get('error')}")
                continue
                
            choice = res_json["choices"][0]
            if "reasoning_details" in choice["message"]:
                logger.debug(f"AI Reasoning: {choice['message']['reasoning_details']}")
                
            response_text = choice["message"]["content"]
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                logger.info(f"Successfully used {model_id}!")
                return json.loads(match.group())
                
        except Exception as e:
            logger.warning(f"Failed with {model_id}: {e}")
            continue
            
    raise ValueError("All reasoning-enabled models failed.")

# API ENDPOINTS
@app.post("/parse-marksheet", response_model=MarkSheetData)
async def parse_marksheet(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        ocr_text = ""
        processing_image = file_bytes
        
        # Handle PDF vs Image
        if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
            logger.info("Processing PDF input (extracting up to 3 pages)...")
            img_list, raw_text = process_pdf_pages(file_bytes)
            if img_list:
                processing_image = img_list # Now it's a list
                ocr_text = raw_text
        
        # Run OCR update if text is sparse (scanned documents)
        # Use primary image if it's a list
        ocr_source_image = processing_image[0] if isinstance(processing_image, list) else processing_image
        if len(ocr_text) < 50:
            logger.info("Running OCR.space on document...")
            ocr_text = run_ocr(ocr_source_image)
            
        try:
            print(f"\n--- OCR TEXT ---\n{ocr_text}\n---------------\n")
        except UnicodeEncodeError:
            logger.info("OCR Text contains non-ASCII characters; skipping direct print to terminal.")
        
        structured_data = generate_structured_data(processing_image, ocr_text)
        
        # Calculate Verification Hash (Using Canonical JSON payload)
        canonical_json = build_canonical_payload(structured_data)
        structured_data["merkle_hash"] = generate_keccak256(canonical_json)
        
        return MarkSheetData(**structured_data)
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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI MarkSheet Reasoning Parser</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #8b5cf6; --bg: #030712; --card: #111827; --text: #f9fafb; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; }
        .box { width: 90%; max-width: 900px; background: var(--card); border: 1px solid #1f2937; border-radius: 20px; padding: 30px; }
        .dropzone { border: 2px dashed #374151; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: 0.3s; }
        .dropzone:hover { border-color: var(--primary); background: #1f2937; }
        .btn { background: var(--primary); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 20px; width: 100%; transition: 0.2s; }
        .btn:hover { background: #7c3aed; }
        #results { margin-top: 30px; display: none; }
        .meta-card { background: rgba(255,255,255,0.03); border: 1px solid #1f2937; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        .label { font-size: 0.7em; text-transform: uppercase; color: #9ca3af; margin-bottom: 2px; }
        .val { font-size: 0.95em; font-weight: 600; color: #f3f4f6; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.85em; }
        th, td { padding: 10px; border-bottom: 1px solid #1f2937; text-align: left; }
        th { color: #a78bfa; font-weight: 600; }
        .hash-box { background: #064e3b; color: #34d399; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.8em; margin-top: 20px; border: 1px solid #065f46; word-break: break-all; }
        .log-box { background: #000; color: #00ff00; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.8em; margin-top: 20px; border: 1px solid #333; height: 150px; overflow-y: auto; white-space: pre-wrap; }
        .loader { display: none; text-align: center; margin: 20px 0; color: #a78bfa; }
    </style>
</head>
<body>
    <div class="box">
        <h1 style="margin-top:0;">Perfect MarkSheet Parser</h1>
        <p style="color:#9ca3af;">Deep Reasoning + Keccak-256 Verification</p>
        <form id="pForm">
            <div class="dropzone" onclick="document.getElementById('f').click()">
                <span id="label">Select MarkSheet Scan (Image/PDF)</span>
                <input type="file" id="f" style="display:none" accept="image/*,.pdf" onchange="document.getElementById('label').innerText=this.files[0].name">
            </div>
            <button type="submit" class="btn">Start Blockchain Analysis</button>
        </form>
        <div class="loader" id="l">⚡ AI is Reasoning through your document...</div>
        <div id="results">
            <div class="meta-card">
                <div id="n" style="color:var(--primary); font-weight:600; font-size:1.1em; margin-bottom:2px;"></div>
                <div id="r" style="font-size:0.9em; opacity:0.8; margin-bottom:15px;"></div>
                <div class="grid">
                    <div><div class="label">GPA</div><div id="g" class="val"></div></div>
                </div>
            </div>
            <table>
                <thead><tr><th>Code</th><th>Subject</th><th>Credit Points</th><th>Grade</th></tr></thead>
                <tbody id="b"></tbody>
            </table>
            <div class="hash-box">
                <div class="label" style="color:#6ee7b7; margin-bottom:5px;">KECCAK-256 VERIFICATION HASH</div>
                <span id="h"></span>
            </div>
            <div class="log-box">
                <div class="label" style="color:#4ade80; margin-bottom:5px; font-family: sans-serif;">RAW JSON LOG</div>
                <div id="json-log"></div>
            </div>
        </div>
    </div>
    <script>
        document.getElementById('pForm').onsubmit = async (e) => {
            e.preventDefault();
            const f = document.getElementById('f').files[0];
            if(!f) return;
            document.getElementById('l').style.display='block';
            document.getElementById('results').style.display='none';
            const fd = new FormData(); fd.append('file', f);
            try {
                const res = await fetch('/parse-marksheet', {method:'POST', body:fd});
                const d = await res.json();
                document.getElementById('n').innerText = d.name;
                document.getElementById('r').innerText = 'Registration/Enrollment: ' + d.registration_no;
                document.getElementById('g').innerText = d.gpa;
                document.getElementById('h').innerText = d.merkle_hash;
                
                // Exclude merkle_hash from visual log for clarity
                const { merkle_hash, ...logData } = d;
                document.getElementById('json-log').innerText = JSON.stringify(logData, null, 2);
                
                document.getElementById('b').innerHTML = d.subjects.map(s => `
                    <tr>
                        <td>${s.code || '-'}</td>
                        <td>${s.title || '-'}</td>
                        <td>${s.credit_points || '-'}</td>
                        <td>${s.grade || '-'}</td>
                    </tr>`).join('');
                document.getElementById('results').style.display='block';
                console.log("Extraction Success:", d);
            } catch(er) { 
                console.error("UI Error:", er);
                alert('Analysis Success but UI Rendering Failed: ' + er); 
            }
            finally { document.getElementById('l').style.display='none'; }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    TARGET_PORT = 8000
    force_free_port(TARGET_PORT)
    uvicorn.run(app, host="0.0.0.0", port=TARGET_PORT)