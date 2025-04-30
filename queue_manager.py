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
TEMP_DIR = "./tmp_job_files"  # Ensure consistency if used here


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
                    # More verbose cleanup log
                    print(f"Job {job_id}: Cleaned up temp file: {file_path}")
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
        print(f"Job {job_id}: >>> ENTERING SIMPLIFIED _process_locally <<<")
        output_path_from_thread = None
        thread_exception = None

        try:
            async with self.local_semaphore:
                # Update status immediately to processing
                self.job_store[job_id]["status"] = "processing_local_simplified"
                print(
                    f"Job {job_id}: Acquired semaphore. Status -> processing_local_simplified.")

                # --- Call Predictor ---
                try:
                    print(f"Job {job_id}: Calling asyncio.to_thread...")
                    output_path_from_thread = await asyncio.to_thread(
                        self.predictor.predict,
                        # Pass necessary args
                        input_image_bytes=job_data["input_image_bytes"],
                        mask_image_bytes=job_data["mask_image_bytes"],
                        expression=job_data["expression"],
                        seed=job_data["seed"],
                        height=job_data["height"],
                        width=job_data["width"],
                        subject_lora_scale=job_data["subject_lora_scale"],
                        inpainting_lora_scale=job_data["inpainting_lora_scale"]
                    )
                    print(
                        f"Job {job_id}: >>> Thread finished. Raw result: {output_path_from_thread} <<<")

                except Exception as e:
                    print(
                        f"Job {job_id}: >>> Exception IN thread: {e}\n{traceback.format_exc()} <<<")
                    thread_exception = e

                # --- VERY Basic Post-Thread Logic ---
                # Check if we reach this point AT ALL
                print(
                    f"Job {job_id}: >>> Reached post-thread code block. Exception was: {thread_exception} <<<")

                # Attempt a simple status update based ONLY on whether the thread threw an error
                if thread_exception:
                    print(
                        f"Job {job_id}: Attempting status update -> failed (thread error)")
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id][
                        "result"] = f"Predictor thread error: {str(thread_exception)}"
                elif output_path_from_thread and isinstance(output_path_from_thread, str):
                    # If thread finished and returned a string path (don't check exists, don't read)
                    print(
                        f"Job {job_id}: Attempting status update -> thread_finished_ok")
                    # Use a distinct status to indicate this simplified success
                    self.job_store[job_id]["status"] = "thread_finished_ok"
                    # Store path as result
                    self.job_store[job_id][
                        "result"] = f"Predictor returned path: {output_path_from_thread}"
                else:
                    # Thread didn't error but didn't return a usable path
                    print(
                        f"Job {job_id}: Attempting status update -> failed (bad thread result)")
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id][
                        "result"] = f"Predictor thread returned invalid result: {output_path_from_thread}"

                print(
                    f"Job {job_id}: >>> Post-thread status update attempted. Final status should be '{self.job_store[job_id]['status']}' <<<")

        except Exception as outer_exc:
            # Catch errors in the semaphore logic or the post-thread block itself
            print(
                f"Job {job_id}: >>> Exception in OUTER try/except: {outer_exc}\n{traceback.format_exc()} <<<")
            # Ensure status reflects failure
            self.job_store[job_id]["status"] = "failed"
            self.job_store[job_id]["result"] = f"Queue manager error: {str(outer_exc)}"

        finally:
            print(
                f"Job {job_id}: >>> Releasing local semaphore (finally block). <<<")
            # We won't clean up the output file in this test version
            # await self._cleanup_temp_files(job_id, output_path_from_thread) # Keep cleanup commented out

    async def _process_replicate(self, job_id: str, job_data: dict):
        input_bytes = job_data["input_image_bytes"]
        mask_bytes = job_data["mask_image_bytes"]
        replicate_input_temp_path = None
        replicate_mask_temp_path = None

        if not self.replicate_client:
            self.job_store[job_id]["status"] = "failed"
            self.job_store[job_id]["result"] = "Replicate client not configured."
            print(f"Job {job_id}: Skipping Replicate - client not configured.")
            return

        try:
            async with self.replicate_semaphore:
                self.job_store[job_id]["status"] = "processing_replicate"
                print(
                    f"Job {job_id}: Acquired Replicate semaphore. Starting Replicate processing.")

                input_fd, replicate_input_temp_path = tempfile.mkstemp(
                    suffix=".png", dir=TEMP_DIR)
                with os.fdopen(input_fd, 'wb') as f:
                    f.write(input_bytes)
                print(
                    f"Job {job_id}: Wrote input bytes to temp file for Replicate: {replicate_input_temp_path}")

                mask_fd, replicate_mask_temp_path = tempfile.mkstemp(
                    suffix=".png", dir=TEMP_DIR)
                with os.fdopen(mask_fd, 'wb') as f:
                    f.write(mask_bytes)
                print(
                    f"Job {job_id}: Wrote mask bytes to temp file for Replicate: {replicate_mask_temp_path}")

                replicate_input_dict = {
                    "input_image": open(replicate_input_temp_path, "rb"),
                    "inpainting_mask": open(replicate_mask_temp_path, "rb"),
                    "expression": job_data["expression"], "seed": job_data["seed"],
                    "height": job_data["height"], "width": job_data["width"],
                    "subject_lora_scale": job_data["subject_lora_scale"],
                    "inpainting_lora_scale": job_data["inpainting_lora_scale"]
                }
                output = None
                try:
                    output = await asyncio.to_thread(self.replicate_client.run, REPLICATE_MODEL_ID, input=replicate_input_dict)
                finally:
                    replicate_input_dict["input_image"].close()
                    replicate_input_dict["inpainting_mask"].close()
                    print(
                        f"Job {job_id}: Closed temp file handles passed to Replicate.")

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
            await self._cleanup_temp_files(job_id, replicate_input_temp_path, replicate_mask_temp_path)

    async def submit_job(self, job_data: dict) -> str:
        job_id = str(uuid.uuid4())
        self.job_store[job_id] = {"status": "pending", "result": None}
        print(f"Job {job_id}: Submitted. Checking resources...")
        can_process_locally = not self.local_semaphore.locked(
        ) or self._local_count() < self._max_local
        if can_process_locally:
            print(f"Job {job_id}: Assigning to local processor.")
            asyncio.create_task(self._process_locally(job_id, job_data))
        elif self.replicate_client and not self.replicate_semaphore.locked():
            print(f"Job {job_id}: Local busy, assigning to Replicate.")
            asyncio.create_task(self._process_replicate(job_id, job_data))
        else:
            print(
                f"Job {job_id}: Local and Replicate busy/unavailable. Queued for local.")
            asyncio.create_task(self._process_locally(job_id, job_data))
        return job_id

    def _local_count(self):
        return self._max_local - self.local_semaphore._value

    async def get_job_status(self, job_id: str) -> dict | None:
        return self.job_store.get(job_id)
