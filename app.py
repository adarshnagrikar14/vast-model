import os
# import base64
import traceback
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from predict import (
    Predictor,
    process_local,
    process_replicate,
    process_gemini
)
from queue_manager import queue_manager
import torch

TEMP_DIR = "./tmp_job_files"
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except OSError as e:
    print(f"FATAL: Could not create temporary directory {TEMP_DIR}: {e}")


def occupy_gpu_memory(gb=30):
    """Occupy approximately `gb` GB of GPU memory."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 1 float32 = 4 bytes, so 1GB = 2^30 bytes / 4 = 268,435,456 floats
        num_floats = int((gb * (1024 ** 3)) / 4)
        try:
            _ = torch.empty(num_floats, dtype=torch.float32, device=device)
            print(f"Allocated ~{gb}GB on GPU {torch.cuda.current_device()}")
        except RuntimeError as e:
            print(f"Failed to allocate {gb}GB on GPU: {e}")
    else:
        print("CUDA not available, cannot occupy GPU memory.")


app = FastAPI(title="Manhwa AI Generation Service")
# occupy_gpu_memory(30)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()
predictor.setup()

queue_manager.start(
    local_processor=process_local,
    replicate_processor=process_replicate,
    gemini_processor=process_gemini
)

# If you have 2 OpenAI keys that can handle 9 requests each per minute
# Update this line in app.py or add it to your .env file
MAX_GEMINI_CONCURRENCY = int(os.environ.get("MAX_GEMINI_CONCURRENCY", "80"))


@app.get("/")
def index():
    return {
        "status": "online",
        "message": "Manhwa AI Generation Service",
        "endpoints": {
            "generate": {
                "path": "/generate",
                "method": "POST",
                "description": "Submit image generation job (form data: subject, mask). Returns result directly.",
                "response": "{'image_base64': '...' | 'error': '...'}"
            },
            "submit": {
                "path": "/submit",
                "method": "POST",
                "description": "Submit image generation job asynchronously (form data: subject, mask). Returns job ID.",
                "response": "{'job_id': '...' | 'error': '...'}"
            },
            "status": {
                "path": "/status/{job_id}",
                "method": "GET",
                "description": "Check status of an asynchronous job.",
                "response": "{'status': 'queued|processing|completed|failed', 'result': '...'|null, 'error': '...'|null}"
            }
        }
    }


@app.post("/generate")
async def generate(
    subject: UploadFile = File(...),
    mask: UploadFile = File(...),
    expression: str = Form("k-pop happy")
):
    """Synchronous endpoint for image generation (backward compatibility)"""
    try:
        subject_bytes = await subject.read()
        mask_bytes = await mask.read()

        # Submit to queue manager
        job_id = queue_manager.submit_job({
            "input_image_bytes": subject_bytes,
            "mask_image_bytes": mask_bytes,
            "expression": expression
        })

        # Wait for result (blocking)
        while True:
            status, result, error = queue_manager.get_job_status(job_id)
            if status in (None, "failed"):
                return JSONResponse(status_code=500, content={
                    'error': error or 'An internal error occurred during generation.'
                })
            elif status == "completed":
                return JSONResponse(content={'image_base64': result})

            # Wait a bit before checking again
            import asyncio
            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"Error in /generate endpoint: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={
            'error': 'An internal error occurred during generation.',
            'detail': str(e)
        })


@app.post("/submit")
async def submit_job(
    subject: UploadFile = File(...),
    mask: UploadFile = File(...),
    expression: str = Form("k-pop happy")
):
    """Asynchronous endpoint to submit a job"""
    try:
        subject_bytes = await subject.read()
        mask_bytes = await mask.read()

        job_id = queue_manager.submit_job({
            "input_image_bytes": subject_bytes,
            "mask_image_bytes": mask_bytes,
            "expression": expression
        })

        return {"job_id": job_id}

    except Exception as e:
        print(f"Error in /submit endpoint: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={
            'error': 'Failed to submit job',
            'detail': str(e)
        })


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a submitted job"""
    status, result, error = queue_manager.get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": status.value if hasattr(status, "value") else status,
        "result": result,
        "error": error
    }


@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown"""
    queue_manager.stop()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)
