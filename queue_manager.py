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
from concurrent.futures import ThreadPoolExecutor  # Import executor

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
        # Create an executor (can be shared)
        self.executor = ThreadPoolExecutor(
            max_workers=self._max_local + 2)  # +2 for some buffer

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
        print(
            f"Job {job_id}: >>> ENTERING _process_locally (using run_in_executor) <<<")
        output_path_from_thread = None
        task_exception = None

        try:
            async with self.local_semaphore:
                # New status
                self.job_store[job_id]["status"] = "processing_local_executor"
                print(
                    f"Job {job_id}: Acquired semaphore. Status -> processing_local_executor.")

                # --- Get the current event loop ---
                loop = asyncio.get_running_loop()

                # --- Call Predictor using run_in_executor ---
                try:
                    print(f"Job {job_id}: Calling loop.run_in_executor...")
                    loop = asyncio.get_running_loop()  # Ensure loop is defined here
                    output_path_from_thread = await loop.run_in_executor(
                        self.executor,
                        self.predictor.predict,
                        # --- CORRECTED ARGUMENT ORDER ---
                        job_data["input_image_bytes"],      # 1
                        job_data["mask_image_bytes"],       # 2
                        job_data["expression"],             # 3
                        job_data["height"],                 # 4 CORRECT
                        job_data["width"],                  # 5 CORRECT
                        job_data["seed"],                   # 6 CORRECT
                        job_data["subject_lora_scale"],     # 7
                        job_data["inpainting_lora_scale"]   # 8
                        # --- END CORRECTED ORDER ---
                    )
                    print(
                        f"Job {job_id}: >>> run_in_executor call completed. Raw result: {output_path_from_thread} <<<")

                except asyncio.CancelledError:
                    # Explicitly catch CancelledError
                    print(
                        f"Job {job_id}: >>> Task was CANCELLED while awaiting executor. <<<")
                    # Set status to failed on cancellation
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id]["result"] = "Task cancelled during prediction."
                    # Re-raise or handle as needed, here we just log and set status
                    raise  # Re-raising might be useful for higher level handling if needed

                except Exception as e:
                    print(
                        f"Job {job_id}: >>> Exception IN executor task: {e}\n{traceback.format_exc()} <<<")
                    task_exception = e

                # --- Basic Post-Thread Logic ---
                print(
                    f"Job {job_id}: >>> Reached post-executor code block. Exception was: {task_exception} <<<")

                # Update status based ONLY on whether the executor task threw an error
                # (Keep simplified logic for now)
                if task_exception:
                    print(
                        f"Job {job_id}: Attempting status update -> failed (executor error)")
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id][
                        "result"] = f"Predictor executor error: {str(task_exception)}"
                elif output_path_from_thread and isinstance(output_path_from_thread, str):
                    print(
                        f"Job {job_id}: Attempting status update -> executor_finished_ok")
                    # Distinct status
                    self.job_store[job_id]["status"] = "executor_finished_ok"
                    self.job_store[job_id][
                        "result"] = f"Predictor returned path: {output_path_from_thread}"
                else:
                    print(
                        f"Job {job_id}: Attempting status update -> failed (bad executor result)")
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id][
                        "result"] = f"Predictor executor returned invalid result: {output_path_from_thread}"

                print(
                    f"Job {job_id}: >>> Post-executor status update attempted. Final status should be '{self.job_store[job_id]['status']}' <<<")

        except asyncio.CancelledError:
            # Catch cancellation if it happens outside the executor await but within the semaphore
            print(f"Job {job_id}: >>> Task was CANCELLED (outer try block). <<<")
            if job_id in self.job_store:  # Ensure job exists before updating
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id]["result"] = "Task cancelled during processing."

        except Exception as outer_exc:
            print(
                f"Job {job_id}: >>> Exception in OUTER try/except: {outer_exc}\n{traceback.format_exc()} <<<")
            if job_id in self.job_store:
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id]["result"] = f"Queue manager error: {str(outer_exc)}"

        finally:
            print(
                f"Job {job_id}: >>> Releasing local semaphore (finally block). <<<")
            # Cleanup output file if path was valid (use output_path_from_thread)
            await self._cleanup_temp_files(job_id, output_path_from_thread if not task_exception else None)

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
                loop = asyncio.get_running_loop()  # Get loop for executor
                try:
                    # Use run_in_executor for replicate call too
                    output = await loop.run_in_executor(
                        self.executor,  # Reuse executor
                        self.replicate_client.run,
                        REPLICATE_MODEL_ID,  # Ensure REPLICATE_MODEL_ID is defined
                        replicate_input_dict  # input= dict is handled by run method
                    )
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

    # Add a cleanup method for the executor on shutdown if needed
    async def shutdown(self):
        print("Shutting down thread pool executor...")
        self.executor.shutdown(wait=True)
        print("Executor shut down.")

# Consider calling queue_manager.shutdown() during application exit if possible
