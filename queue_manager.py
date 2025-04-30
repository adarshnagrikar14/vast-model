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
        output_path = None
        predictor_exception = None  # Variable to store exception from thread

        try:
            async with self.local_semaphore:
                self.job_store[job_id]["status"] = "processing_local"
                print(
                    f"Job {job_id}: Acquired local semaphore. Starting local processing.")

                # --- Call Predictor ---
                try:
                    output_path = await asyncio.to_thread(
                        self.predictor.predict,
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
                        f"Job {job_id}: Predictor thread finished. Returned path: {output_path}")
                except Exception as pred_exc:
                    # Catch exception *from the predictor thread*
                    predictor_exception = pred_exc
                    print(
                        f"Job {job_id}: Predictor thread raised an exception: {pred_exc}\n{traceback.format_exc()}")
                    # Don't re-raise yet, let the main block handle status update

                # --- Process Result and Update Status ---
                # This block runs *after* the thread finishes or raises an exception

                if predictor_exception:
                    # If the predictor itself failed, update status to failed now
                    print(
                        f"Job {job_id}: Setting status to failed due to predictor exception.")
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id]["result"] = f"Predictor error: {str(predictor_exception)}"

                elif not output_path or not os.path.exists(output_path):
                    # If predictor didn't fail but path is bad
                    print(
                        f"Job {job_id}: Predictor returned invalid/missing path: {output_path}. Setting status to failed.")
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id][
                        "result"] = f"Predictor returned invalid path: {output_path}"
                    # Set output_path to None to prevent cleanup error
                    output_path = None

                else:
                    # Predictor succeeded and path looks okay, try reading/encoding/updating
                    try:
                        print(
                            f"Job {job_id}: Attempting to read output file: {output_path}")
                        with open(output_path, 'rb') as f:
                            image_bytes = f.read()
                        print(
                            f"Job {job_id}: Successfully read {len(image_bytes)} bytes from output file.")

                        print(
                            f"Job {job_id}: Attempting to Base64 encode image bytes.")
                        base64_result = base64.b64encode(
                            image_bytes).decode('utf-8')
                        print(f"Job {job_id}: Base64 encoding successful.")

                        print(
                            f"Job {job_id}: Updating job store status to 'completed'.")
                        self.job_store[job_id]["status"] = "completed"
                        self.job_store[job_id]["result"] = base64_result
                        print(
                            f"Job {job_id}: Local processing completed successfully. Status updated.")

                    except Exception as post_proc_exc:
                        # Catch error during file read, encode, or status update itself
                        print(
                            f"Job {job_id}: Error during post-processing/status update: {post_proc_exc}\n{traceback.format_exc()}")
                        self.job_store[job_id]["status"] = "failed"
                        self.job_store[job_id][
                            "result"] = f"Post-processing error: {str(post_proc_exc)}"

        except Exception as main_exc:
            # Catch unexpected errors in the main async logic (e.g., semaphore issues)
            print(
                f"Job {job_id}: Uncaught exception in _process_locally main block: {main_exc}\n{traceback.format_exc()}")
            # Ensure status is marked as failed if not already set
            if job_id in self.job_store and self.job_store[job_id]["status"] not in ["completed", "failed"]:
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id]["result"] = f"Queue processing error: {str(main_exc)}"

        finally:
            print(f"Job {job_id}: Releasing local semaphore.")
            # Cleanup the temporary *output* file from predict if path is valid
            # output_path might be None if predictor failed early
            await self._cleanup_temp_files(job_id, output_path)

    async def _process_replicate(self, job_id: str, job_data: dict):
        input_bytes = job_data["input_image_bytes"]
        mask_bytes = job_data["mask_image_bytes"]
        replicate_input_temp_path = None
        replicate_mask_temp_path = None

        if not self.replicate_client:
            self.job_store[job_id]["status"] = "failed"
            self.job_store[job_id]["result"] = "Replicate client not configured."
            print(f"Job {job_id}: Skipping Replicate - client not configured.")
            # No files to clean up here as bytes were passed
            return

        try:
            async with self.replicate_semaphore:
                self.job_store[job_id]["status"] = "processing_replicate"
                print(
                    f"Job {job_id}: Acquired Replicate semaphore. Starting Replicate processing.")

                # --- Create temporary files from bytes for Replicate library ---
                # Assuming suffix might matter for replicate based on original filename if available
                # suffix = Path(job_data.get("input_image_filename", ".png")).suffix or '.png' # Example
                input_fd, replicate_input_temp_path = tempfile.mkstemp(
                    suffix=".png", dir=TEMP_DIR)  # Use appropriate suffix
                with os.fdopen(input_fd, 'wb') as f:
                    f.write(input_bytes)
                print(
                    f"Job {job_id}: Wrote input bytes to temp file for Replicate: {replicate_input_temp_path}")

                # suffix = Path(job_data.get("mask_image_filename", ".png")).suffix or '.png'
                mask_fd, replicate_mask_temp_path = tempfile.mkstemp(
                    suffix=".png", dir=TEMP_DIR)
                with os.fdopen(mask_fd, 'wb') as f:
                    f.write(mask_bytes)
                print(
                    f"Job {job_id}: Wrote mask bytes to temp file for Replicate: {replicate_mask_temp_path}")
                # --- End temp file creation ---

                replicate_input_dict = {
                    "input_image": open(replicate_input_temp_path, "rb"),
                    "inpainting_mask": open(replicate_mask_temp_path, "rb"),
                    "expression": job_data["expression"],
                    "seed": job_data["seed"],
                    "height": job_data["height"],
                    "width": job_data["width"],
                    "subject_lora_scale": job_data["subject_lora_scale"],
                    "inpainting_lora_scale": job_data["inpainting_lora_scale"]
                }

                output = None
                try:
                    output = await asyncio.to_thread(
                        self.replicate_client.run,
                        REPLICATE_MODEL_ID,
                        input=replicate_input_dict
                    )
                finally:
                    # Ensure files opened for replicate are closed
                    replicate_input_dict["input_image"].close()
                    replicate_input_dict["inpainting_mask"].close()
                    print(
                        f"Job {job_id}: Closed temp file handles passed to Replicate.")

                # ... (rest of replicate processing: download, encode, update status) ...
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
            # Cleanup the temporary files created *specifically* for Replicate
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
