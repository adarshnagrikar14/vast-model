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
import threading
import queue
import time

# Configuration
MAX_LOCAL_CONCURRENCY = int(os.environ.get("MAX_LOCAL_CONCURRENCY", 1))
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
# Ensure you use the correct model identifier for your Replicate model
REPLICATE_MODEL_ID = "adarshnagrikar14/manhwa-ai:0ed8ac8e28cfb050730eb3e1fbcbc9c60d7001e3a53931cc4f3c44cf08bab659"
TEMP_DIR = "./tmp_job_files"  # Ensure consistency if used here

job_queue: queue.Queue[tuple[str, dict]] = queue.Queue()


def worker_loop():
    while True:
        job_id, job_data = job_queue.get()
        try:
            QueueManager._really_run_job(job_id, job_data)   # new helper
        finally:
            job_queue.task_done()


threading.Thread(target=worker_loop, daemon=True).start()


class QueueManager:
    def __init__(self, predictor: Predictor):
        if not isinstance(MAX_LOCAL_CONCURRENCY, int) or MAX_LOCAL_CONCURRENCY < 1:
            print(
                f"Warning: Invalid MAX_LOCAL_CONCURRENCY ({MAX_LOCAL_CONCURRENCY}), defaulting to 1.")
            self._max_local = 1
        else:
            self._max_local = MAX_LOCAL_CONCURRENCY
        self.predictor = predictor
        self.local_semaphore = threading.Semaphore(self._max_local)
        self.replicate_semaphore = threading.Semaphore(1)
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

        # --- Start Background Worker Thread ---
        # Pass 'self' (this instance) to the worker loop
        self.worker_thread = threading.Thread(
            target=self._worker_loop, args=(self,), daemon=True)
        self.worker_thread.start()
        print("Background worker thread started.")

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
            f"Job {job_id}: >>> ENTERING _process_locally (using polling) <<<")
        output_path = None
        task_exception = None
        future = None  # Variable to hold the future object

        try:
            async with self.local_semaphore:
                self.job_store[job_id]["status"] = "processing_local_polling"
                print(
                    f"Job {job_id}: Acquired semaphore. Status -> processing_local_polling.")

            # --- Submit predictor task to executor (don't await yet) ---
            print(f"Job {job_id}: Submitting predictor task to executor...")
            future = self.executor.submit(
                self.predictor.predict,
                # Corrected args:
                job_data["input_image_bytes"],
                job_data["mask_image_bytes"],
                job_data["expression"],
                job_data["height"],
                job_data["width"],
                job_data["seed"],
                job_data["subject_lora_scale"],
                job_data["inpainting_lora_scale"]
            )

            # --- Poll the future until it's done ---
            polling_interval = 0.5  # seconds
            print(
                f"Job {job_id}: Polling executor future every {polling_interval}s...")
            while not future.done():
                # Check for cancellation explicitly during polling
                try:
                    # await asyncio.wait_for(asyncio.shield(future), timeout=polling_interval) # Alternative check
                    await asyncio.sleep(polling_interval)  # Yield control
                except asyncio.CancelledError:
                    print(
                        f"Job {job_id}: >>> Task CANCELLED during polling sleep/wait. <<<")
                    # Attempt to cancel the background future if it's still running
                    if not future.done():
                        print(
                            f"Job {job_id}: Attempting to cancel background future...")
                        future.cancel()
                    # Set status and re-raise
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id]["result"] = "Task cancelled during prediction polling."
                    raise  # Propagate cancellation

            # --- Future is done, get the result (or exception) ---
            print(f"Job {job_id}: Executor future is done.")
            try:
                output_path = future.result()  # This will raise exception if thread failed
                print(
                    f"Job {job_id}: >>> future.result() succeeded. Raw result: {output_path} <<<")
            except Exception as e:
                # Exception came from the predictor thread itself
                print(
                    f"Job {job_id}: >>> Exception retrieved from future.result(): {e}\n{traceback.format_exc()} <<<")
                task_exception = e

            # --- Process Result and Update Status (Simplified Logic is Fine Here) ---
            print(
                f"Job {job_id}: >>> Reached post-polling code block. Exception was: {task_exception} <<<")

            if task_exception:
                print(
                    f"Job {job_id}: Attempting status update -> failed (predictor error)")
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id][
                    "result"] = f"Predictor error: {str(task_exception)}"
            elif output_path and isinstance(output_path, str) and os.path.exists(output_path):
                # <<< MODIFICATION: Use the original successful processing logic >>>
                print(
                    f"Job {job_id}: Predictor success. Proceeding with post-processing for path: {output_path}")
                try:
                    print(
                        f"Job {job_id}: Reading output file: {output_path}")
                    with open(output_path, 'rb') as f:
                        image_bytes = f.read()
                    print(f"Job {job_id}: Read {len(image_bytes)} bytes.")
                    print(f"Job {job_id}: Base64 encoding...")
                    base64_result = base64.b64encode(
                        image_bytes).decode('utf-8')
                    print(f"Job {job_id}: Base64 done.")
                    print(f"Job {job_id}: Updating status to 'completed'.")
                    self.job_store[job_id]["status"] = "completed"
                    self.job_store[job_id]["result"] = base64_result
                    print(f"Job {job_id}: Status updated to 'completed'.")
                except Exception as post_proc_exc:
                    print(
                        f"Job {job_id}: Exception during post-processing: {post_proc_exc}\n{traceback.format_exc()}")
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id][
                        "result"] = f"Post-processing error: {str(post_proc_exc)}"
                # <<< END MODIFICATION >>>
            else:
                # Predictor finished but path is bad
                result_str = str(output_path) if output_path else "None"
                print(
                    f"Job {job_id}: Attempting status update -> failed (bad predictor result: {result_str})")
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id][
                    "result"] = f"Predictor returned invalid result: {result_str}"
                output_path = None  # Prevent cleanup error

            print(
                f"Job {job_id}: >>> Post-polling status update finished. Status should be '{self.job_store[job_id]['status']}' <<<")

        except asyncio.CancelledError:
            print(
                f"Job {job_id}: >>> Task was CANCELLED (outer try block). <<<")
            if job_id in self.job_store and self.job_store[job_id]["status"] not in ["completed", "failed"]:
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id]["result"] = "Task cancelled during processing."

        except Exception as outer_exc:
            print(
                f"Job {job_id}: >>> Exception in OUTER try/except: {outer_exc}\n{traceback.format_exc()} <<<")
            if job_id in self.job_store and self.job_store[job_id]["status"] not in ["completed", "failed"]:
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id]["result"] = f"Queue manager error: {str(outer_exc)}"

        finally:
            print(
                f"Job {job_id}: >>> Releasing local semaphore (finally block). <<<")
            # Cleanup output file if path was set correctly and no exception happened in thread
            await self._cleanup_temp_files(job_id, output_path if not task_exception else None)

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
                    # Using executor for replicate too, can switch back to to_thread if preferred
                    output = self.executor.submit(
                        self.replicate_client.run,
                        REPLICATE_MODEL_ID,
                        replicate_input_dict
                    ).result()
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

    def submit_job(self, job_data):
        job_id = str(uuid.uuid4())
        self.job_store[job_id] = {"status": "queued", "result": None}
        job_queue.put((job_id, job_data))
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

    def _really_run_job(self, job_id: str, job_data: dict):
        # This method is now synchronous and does not use asyncio
        # Implement the logic to process the job
        pass

    # --- Worker loop function (now a method to access self easily) ---
    @staticmethod  # Make it static or pass self if needed, here static is fine
    def _worker_loop(manager_instance):  # Receive the QueueManager instance
        """Pulls jobs from queue and runs them."""
        print("Worker loop waiting for jobs...")
        while True:
            job_id, job_data = job_queue.get()  # Blocks until a job is available
            print(f"Worker picked up job {job_id}.")
            try:
                # Call the instance method _really_run_job using the passed instance
                # Decide here whether to run locally or replicate based on semaphores maybe?
                # Simple approach: prioritize local for now
                if manager_instance.local_semaphore.acquire(blocking=False):
                    manager_instance._really_run_job_locally(job_id, job_data)
                elif manager_instance.replicate_client and manager_instance.replicate_semaphore.acquire(blocking=False):
                    # Add a _really_run_job_replicate if implementing replicate offload
                    print(
                        f"Job {job_id}: Local busy, attempting Replicate (Not fully implemented in worker yet).")
                    # manager_instance._really_run_job_replicate(job_id, job_data) # TODO
                    manager_instance.replicate_semaphore.release()  # Release if not implemented
                    # For now, fail if local is busy and replicate not done
                    manager_instance.job_store[job_id]['status'] = 'failed'
                    manager_instance.job_store[job_id]['result'] = 'Local worker busy, Replicate worker not implemented.'
                    print(f"Job {job_id}: Failed - local worker busy.")
                else:
                    # Could re-queue or fail immediately
                    manager_instance.job_store[job_id]['status'] = 'failed'
                    manager_instance.job_store[job_id]['result'] = 'All workers busy.'
                    print(f"Job {job_id}: Failed - all workers busy.")

            except Exception as e:
                print(
                    f"!!! UNHANDLED EXCEPTION IN WORKER for job {job_id}: {e}\n{traceback.format_exc()}")
                try:  # Attempt to update status even if worker errored badly
                    manager_instance.job_store[job_id]['status'] = 'failed'
                    manager_instance.job_store[job_id][
                        'result'] = f'Worker loop error: {e}'
                except Exception:
                    pass  # Ignore errors during error reporting
            finally:
                job_queue.task_done()  # Signal queue that task is complete
                print(f"Worker finished processing job {job_id}.")

    # --- The actual blocking work (previously _process_locally without async/await) ---
    def _really_run_job_locally(self, job_id: str, job_data: dict):
        """Synchronous function to run prediction and update status."""
        print(f"Job {job_id}: >>> ENTERING _really_run_job_locally <<<")
        output_path = None
        task_exception = None

        try:
            # Update status now that it's actively processing
            self.job_store[job_id]["status"] = "processing_local"
            print(f"Job {job_id}: Status -> processing_local.")

            # --- Call Predictor Directly (Blocking) ---
            try:
                print(f"Job {job_id}: Calling predictor.predict directly...")
                output_path = self.predictor.predict(
                    # Corrected args:
                    input_image_bytes=job_data["input_image_bytes"],
                    mask_image_bytes=job_data["mask_image_bytes"],
                    expression=job_data["expression"],
                    height=job_data["height"],
                    width=job_data["width"],
                    seed=job_data["seed"],
                    subject_lora_scale=job_data["subject_lora_scale"],
                    inpainting_lora_scale=job_data["inpainting_lora_scale"]
                )
                print(
                    f"Job {job_id}: >>> Predictor finished. Raw result path: {output_path} <<<")

            except Exception as e:
                print(
                    f"Job {job_id}: >>> Exception IN predictor: {e}\n{traceback.format_exc()} <<<")
                task_exception = e

            # --- Process Result and Update Status ---
            print(
                f"Job {job_id}: >>> Reached post-predictor code block. Exception was: {task_exception} <<<")

            if task_exception:
                print(
                    f"Job {job_id}: Attempting status update -> failed (predictor error)")
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id]["result"] = f"Predictor error: {str(task_exception)}"
            elif output_path and isinstance(output_path, str) and os.path.exists(output_path):
                print(
                    f"Job {job_id}: Predictor success. Proceeding with post-processing for path: {output_path}")
                try:
                    print(f"Job {job_id}: Reading output file: {output_path}")
                    with open(output_path, 'rb') as f:
                        image_bytes = f.read()
                    print(f"Job {job_id}: Read {len(image_bytes)} bytes.")
                    print(f"Job {job_id}: Base64 encoding...")
                    base64_result = base64.b64encode(
                        image_bytes).decode('utf-8')
                    print(f"Job {job_id}: Base64 done.")
                    print(f"Job {job_id}: Updating status to 'completed'.")
                    # <<< CORRECT STATUS
                    self.job_store[job_id]["status"] = "completed"
                    self.job_store[job_id]["result"] = base64_result
                    print(f"Job {job_id}: Status updated to 'completed'.")
                except Exception as post_proc_exc:
                    print(
                        f"Job {job_id}: Exception during post-processing: {post_proc_exc}\n{traceback.format_exc()}")
                    self.job_store[job_id]["status"] = "failed"
                    self.job_store[job_id][
                        "result"] = f"Post-processing error: {str(post_proc_exc)}"
            else:
                # Predictor finished but path is bad
                result_str = str(output_path) if output_path else "None"
                print(
                    f"Job {job_id}: Attempting status update -> failed (bad predictor result: {result_str})")
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id][
                    "result"] = f"Predictor returned invalid result: {result_str}"
                output_path = None  # Prevent cleanup error

            print(
                f"Job {job_id}: >>> Post-predictor status update finished. Status should be '{self.job_store[job_id]['status']}' <<<")

        except Exception as outer_exc:
            # Catch errors in this function's logic
            print(
                f"Job {job_id}: >>> Exception in _really_run_job_locally: {outer_exc}\n{traceback.format_exc()} <<<")
            # Ensure status reflects failure if not already set
            if job_id in self.job_store and self.job_store[job_id]["status"] not in ["completed", "failed"]:
                self.job_store[job_id]["status"] = "failed"
                self.job_store[job_id][
                    "result"] = f"Queue manager worker error: {str(outer_exc)}"

        finally:
            print(
                f"Job {job_id}: >>> Releasing local semaphore (_really_run_job_locally finally block). <<<")
            self.local_semaphore.release()  # Release the semaphore
            # Cleanup output file
            self._cleanup_temp_files_sync(
                job_id, output_path if not task_exception else None)

    # --- Synchronous Cleanup Helper ---
    # (No async needed here as it's called from the worker thread)
    def _cleanup_temp_files_sync(self, job_id, *file_paths):
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    print(
                        f"Job {job_id}: Cleaned up temp file (sync): {file_path}")
                except Exception as e:
                    print(
                        f"Job {job_id}: Warning - failed to delete temp file {file_path} (sync): {e}")

# Consider calling queue_manager.shutdown() during application exit if possible
