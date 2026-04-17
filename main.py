import uvicorn
import logging
import psutil
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router as ocr_router

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MarkSheet OCR Backend",
    description="OCR for verification",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Modular Routes
app.include_router(ocr_router, prefix="/api/v1")

def force_free_port(port: int):
    """Kills any process using the specified port."""
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

if __name__ == "__main__":
    TARGET_PORT = 8000
    force_free_port(TARGET_PORT)
    uvicorn.run(app, host="0.0.0.0", port=TARGET_PORT)
