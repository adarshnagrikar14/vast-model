import asyncio
import uuid
import os
import replicate
import aiohttp
import base64
import io
import tempfile
import traceback
from PIL import Image as PILImage
from predict import Predictor  # Assuming Predictor is accessible

# Configuration
MAX_LOCAL_CONCURRENCY = int(os.environ.get("MAX_LOCAL_CONCURRENCY", 1))
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
# Ensure you use the correct model identifier for your Replicate model
REPLICATE_MODEL_ID = "adarshnagrikar14/manhwa-ai:0ed8ac8e28cfb050730eb3e1fbcbc9c60d7001e3a53931cc4f3c44cf08bab659"


class QueueManager:
    def __init__(self, predictor: Predictor):
        if not isinstance(MAX_LOCAL_CONCURRENCY, int) or MAX_LOCAL_CONCURRENCY < 1:
            print(
                f"Warning: Invalid MAX_LOCAL_CONCURRENCY ({MAX_LOCAL_CONCURRENCY}), defaulting to 1.")
            self._max_local = 1
        else:
            self._max_local = MAX_LOCAL_CONCURRENCY
        self.predictor = predictor
        self.local_semaphore = asyncio.Semaphore(self._max_local)
        self.replicate_semaphore = asyncio.Semaphore(1)
        self.job_store = {}  # Stores {job_id: {"status": ..., "result": ...}}
        self.replicate_client = None
        if REPLICATE_API_TOKEN:
            try:
                self.replicate_client = replicate.Client(
                    api_token=REPLICATE_API_TOKEN)
                print("Replicate client configured.")
            except Exception as e:
                print(
                    f"Warning: Failed to initialize Replicate client: {e}. Replicate offloading disabled.")
                self.replicate_client = None
        else:
            print("Warning: REPLICATE_API_TOKEN not set. Replicate offloading disabled.")

    async def _cleanup_temp_files(self, job_id, *file_paths):
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    # print(f"Job {job_id}: Cleaned up temp file: {file_path}") # Optional: verbose logging
                except Exception as e:
                    print(
                        f"Job {job_id}: Warning - failed to delete temp file {file_path}: {e}")

    async def _download_and_encode(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    image_bytes = await resp.read()
                    return base64.b64encode(image_bytes).decode('utf-8')
            except aiohttp.ClientError as e:
                raise ConnectionError(
                    f"Failed to download image from Replicate URL {url}: {e}") from e
            except Exception as e:
                raise ValueError(
                    f"Failed processing image from Replicate URL {url}: {e}") from e

    async def _process_locally(self, job_id: str, job_data: dict):
        input_path = job_data["input_image_path"]
        mask_path = job_data["mask_image_path"]
        output_path = None
        try:
            async with self.local_semaphore:
                self.job_store[job_id]["status"] = "processing_local"
                print(
                    f"Job {job_id}: Acquired local semaphore. Starting local processing.")

                # Run synchronous predictor in a separate thread
                output_path = await asyncio.to_thread(
                    self.predictor.predict,
                    input_image=input_path,
                    inpainting_mask=mask_path,
                    expression=job_data["expression"],
                    seed=job_data["seed"],
                    height=job_data["height"],
                    width=job_data["width"],
                    subject_lora_scale=job_data["subject_lora_scale"],
                    inpainting_lora_scale=job_data["inpainting_lora_scale"]
                )

                print(
                    f"Job {job_id}: Local prediction finished. Output at {output_path}")

                # Read result and encode
                with open(output_path, 'rb') as f:
                    image_bytes = f.read()
                base64_result = base64.b64encode(image_bytes).decode('utf-8')

                self.job_store[job_id]["status"] = "completed"
                self.job_store[job_id]["result"] = base64_result
                print(f"Job {job_id}: Local processing completed successfully.")

        except Exception as e:
            print(
                f"Job {job_id}: Local processing failed: {e}\n{traceback.format_exc()}")
            self.job_store[job_id]["status"] = "failed"
            self.job_store[job_id]["result"] = f"Local processing error: {e}"
        finally:
            print(f"Job {job_id}: Releasing local semaphore.")
            # Cleanup input files and the temporary output file from predict
            await self._cleanup_temp_files(job_id, input_path, mask_path, output_path)

    async def _process_replicate(self, job_id: str, job_data: dict):
        input_path = job_data["input_image_path"]
        mask_path = job_data["mask_image_path"]
        if not self.replicate_client:
            # Should not happen if check is done before calling, but as safeguard
            self.job_store[job_id]["status"] = "failed"
            self.job_store[job_id]["result"] = "Replicate client not configured."
            print(f"Job {job_id}: Skipping Replicate - client not configured.")
            await self._cleanup_temp_files(job_id, input_path, mask_path)
            return

        try:
            async with self.replicate_semaphore:
                self.job_store[job_id]["status"] = "processing_replicate"
                print(
                    f"Job {job_id}: Acquired Replicate semaphore. Starting Replicate processing.")

                # Prepare input for Replicate, needs files opened
                # Use replicate library's file handling
                replicate_input = {
                    "input_image": open(input_path, "rb"),
                    "inpainting_mask": open(mask_path, "rb"),
                    "expression": job_data["expression"],
                    "seed": job_data["seed"],
                    "height": job_data["height"],
                    "width": job_data["width"],
                    "subject_lora_scale": job_data["subject_lora_scale"],
                    "inpainting_lora_scale": job_data["inpainting_lora_scale"]
                }

                try:
                    # Run prediction on Replicate - this is blocking, run in thread
                    output = await asyncio.to_thread(
                        self.replicate_client.run,
                        REPLICATE_MODEL_ID,
                        input=replicate_input
                    )
                finally:
                    # Ensure files opened for replicate are closed
                    replicate_input["input_image"].close()
                    replicate_input["inpainting_mask"].close()

                if isinstance(output, list) and len(output) > 0 and isinstance(output[0], str):
                    image_url = output[0]
                    print(
                        f"Job {job_id}: Replicate finished, downloading result from {image_url}")
                    base64_result = await self._download_and_encode(image_url)
                    self.job_store[job_id]["status"] = "completed"
                    self.job_store[job_id]["result"] = base64_result
                    print(
                        f"Job {job_id}: Replicate processing completed successfully.")
                else:
                    raise ValueError(
                        f"Unexpected output format from Replicate: {output}")

        except Exception as e:
            print(
                f"Job {job_id}: Replicate processing failed: {e}\n{traceback.format_exc()}")
            self.job_store[job_id]["status"] = "failed"
            self.job_store[job_id]["result"] = f"Replicate processing error: {e}"
        finally:
            print(f"Job {job_id}: Releasing Replicate semaphore.")
            # Cleanup input files
            await self._cleanup_temp_files(job_id, input_path, mask_path)

    async def submit_job(self, job_data: dict) -> str:
        job_id = str(uuid.uuid4())
        self.job_store[job_id] = {"status": "pending", "result": None}
        print(f"Job {job_id}: Submitted. Checking resources...")

        # Check if local processing slot is available *without* waiting
        can_process_locally = not self.local_semaphore.locked(
        ) or self._local_count() < self._max_local

        if can_process_locally:
            print(f"Job {job_id}: Assigning to local processor.")
            asyncio.create_task(self._process_locally(job_id, job_data))
        elif self.replicate_client and not self.replicate_semaphore.locked():
            # Local busy, try Replicate if configured and free
            print(f"Job {job_id}: Local busy, assigning to Replicate.")
            asyncio.create_task(self._process_replicate(job_id, job_data))
        else:
            # Both busy or Replicate unavailable, queue for local (will wait for semaphore)
            print(
                f"Job {job_id}: Local and Replicate busy/unavailable. Queued for local.")
            asyncio.create_task(self._process_locally(job_id, job_data))

        return job_id

    def _local_count(self):
        return self._max_local - self.local_semaphore._value

    async def get_job_status(self, job_id: str) -> dict | None:
        return self.job_store.get(job_id)
