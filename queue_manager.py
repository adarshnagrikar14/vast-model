import os
import time
import uuid
import queue
import threading
from enum import Enum
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple, Callable, List

load_dotenv()

MAX_LOCAL_CONCURRENCY = int(os.environ.get("MAX_LOCAL_CONCURRENCY", "0"))
MAX_REPLICATE_CONCURRENCY = int(
    os.environ.get("MAX_REPLICATE_CONCURRENCY", "0"))
MAX_GEMINI_CONCURRENCY = int(os.environ.get("MAX_GEMINI_CONCURRENCY", "80"))


class ProcessorType(Enum):
    LOCAL = "local"
    REPLICATE = "replicate"
    GEMINI = "gemini"


class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, data: Dict[str, Any]):
        self.job_id = job_id
        self.data = data
        self.status = JobStatus.QUEUED
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.processor_type = None


class QueueManager:
    def __init__(self):
        self.job_queue = queue.Queue()
        self.jobs = {}  # job_id -> Job

        # Resource tracking
        self.local_semaphore = threading.Semaphore(MAX_LOCAL_CONCURRENCY)
        self.replicate_semaphore = threading.Semaphore(
            MAX_REPLICATE_CONCURRENCY)
        self.gemini_semaphore = threading.Semaphore(MAX_GEMINI_CONCURRENCY)

        # Worker thread pools
        self.local_workers: List[threading.Thread] = []
        self.replicate_workers: List[threading.Thread] = []
        self.gemini_workers: List[threading.Thread] = []

        # State tracking
        self._stop_event = threading.Event()
        self._is_running = False

        # Processing callbacks
        self.local_processor = None
        self.replicate_processor = None
        self.gemini_processor = None

    def start(self, local_processor: Callable = None, replicate_processor: Callable = None, gemini_processor: Callable = None):
        """Start the queue manager with processor functions"""
        if self._is_running:
            return

        self.local_processor = local_processor
        self.replicate_processor = replicate_processor
        self.gemini_processor = gemini_processor

        self._is_running = True

        # Create worker thread pools based on concurrency settings
        for i in range(MAX_LOCAL_CONCURRENCY):
            if local_processor:
                worker = threading.Thread(
                    target=self._local_worker,
                    daemon=True,
                    name=f"local-worker-{i}")
                worker.start()
                self.local_workers.append(worker)

        for i in range(MAX_REPLICATE_CONCURRENCY):
            if replicate_processor:
                worker = threading.Thread(
                    target=self._replicate_worker,
                    daemon=True,
                    name=f"replicate-worker-{i}")
                worker.start()
                self.replicate_workers.append(worker)

        for i in range(MAX_GEMINI_CONCURRENCY):
            if gemini_processor:
                worker = threading.Thread(
                    target=self._gemini_worker,
                    daemon=True,
                    name=f"gemini-worker-{i}")
                worker.start()
                self.gemini_workers.append(worker)

        print(
            f"Queue manager started with {len(self.local_workers)} local workers, "
            f"{len(self.replicate_workers)} replicate worker(s), and "
            f"{len(self.gemini_workers)} Gemini worker(s)")

    def stop(self):
        """Stop the queue manager gracefully"""
        self._stop_event.set()
        self._is_running = False

        # Join all worker threads with timeout
        for worker in self.local_workers + self.replicate_workers + self.gemini_workers:
            worker.join(timeout=5)

        self.local_workers.clear()
        self.replicate_workers.clear()
        self.gemini_workers.clear()

        print("Queue manager stopped")

    def submit_job(self, data: Dict[str, Any]) -> str:
        """Submit a new job to the queue"""
        job_id = str(uuid.uuid4())
        job = Job(job_id, data)
        self.jobs[job_id] = job
        self.job_queue.put(job)
        print(
            f"Job {job_id} submitted to queue. Queue size: {self.job_queue.qsize()}")
        return job_id

    def get_job_status(self, job_id: str) -> Tuple[Optional[JobStatus], Optional[Any], Optional[str]]:
        """Get the status, result and error message for a job"""
        if job_id not in self.jobs:
            return None, None, "Job not found"

        job = self.jobs[job_id]
        return job.status, job.result, job.error

    def _process_job(self, job: Job, processor_type: ProcessorType):
        """Process a single job"""
        try:
            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            job.processor_type = processor_type

            print(f"Processing job {job.job_id} with {processor_type.value}")

            if processor_type == ProcessorType.LOCAL:
                result = self.local_processor(**job.data)
            elif processor_type == ProcessorType.REPLICATE:
                result = self.replicate_processor(**job.data)
            elif processor_type == ProcessorType.GEMINI:
                result = self.gemini_processor(**job.data)

            job.result = result
            job.status = JobStatus.COMPLETED
            print(f"Job {job.job_id} completed with {processor_type.value}")
        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            print(f"Job {job.job_id} failed with {processor_type.value}: {e}")
        finally:
            job.completed_at = time.time()

    def _worker_loop(self, semaphore, processor_type):
        """Generic worker loop used by all worker types"""
        while not self._stop_event.is_set():
            acquired = semaphore.acquire(blocking=False)
            if acquired:
                try:
                    # Try to get a job from the queue
                    try:
                        job = self.job_queue.get(block=False)
                        self._process_job(job, processor_type)
                        self.job_queue.task_done()
                    except queue.Empty:
                        pass  # No jobs available
                finally:
                    semaphore.release()
            time.sleep(0.1)  # Prevent tight loop

    def _local_worker(self):
        """Worker thread for local processing"""
        self._worker_loop(self.local_semaphore, ProcessorType.LOCAL)

    def _replicate_worker(self):
        """Worker thread for replicate processing"""
        self._worker_loop(self.replicate_semaphore, ProcessorType.REPLICATE)

    def _gemini_worker(self):
        """Worker thread for Gemini processing"""
        self._worker_loop(self.gemini_semaphore, ProcessorType.GEMINI)

    def clear_completed_jobs(self, max_age_seconds: int = 3600):
        """Clear completed jobs older than max_age_seconds"""
        current_time = time.time()
        to_remove = []

        for job_id, job in self.jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if job.completed_at and (current_time - job.completed_at) > max_age_seconds:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]

    def set_processor_state(self, local_enabled=True, replicate_enabled=True, gemini_enabled=True):
        """Enable or disable specific processor types without restarting the queue manager"""
        if not self._is_running:
            print("Queue manager not running, cannot change processor state")
            return

        # Handle LOCAL processor
        current_local_count = len(self.local_workers)
        target_local_count = MAX_LOCAL_CONCURRENCY if local_enabled else 0

        # Handle REPLICATE processor
        current_replicate_count = len(self.replicate_workers)
        target_replicate_count = MAX_REPLICATE_CONCURRENCY if replicate_enabled else 0

        # Handle GEMINI processor
        current_gemini_count = len(self.gemini_workers)
        target_gemini_count = MAX_GEMINI_CONCURRENCY if gemini_enabled else 0

        # Stop excess workers
        if current_local_count > target_local_count:
            for worker in self.local_workers[target_local_count:]:
                # We can't directly stop threads in Python, so we'll let them terminate naturally
                # by removing them from our tracking list
                pass
            self.local_workers = self.local_workers[:target_local_count]

        if current_replicate_count > target_replicate_count:
            self.replicate_workers = self.replicate_workers[:target_replicate_count]

        if current_gemini_count > target_gemini_count:
            self.gemini_workers = self.gemini_workers[:target_gemini_count]

        # Start new workers if needed
        for i in range(current_local_count, target_local_count):
            if self.local_processor:
                worker = threading.Thread(
                    target=self._local_worker,
                    daemon=True,
                    name=f"local-worker-{i}")
                worker.start()
                self.local_workers.append(worker)

        for i in range(current_replicate_count, target_replicate_count):
            if self.replicate_processor:
                worker = threading.Thread(
                    target=self._replicate_worker,
                    daemon=True,
                    name=f"replicate-worker-{i}")
                worker.start()
                self.replicate_workers.append(worker)

        for i in range(current_gemini_count, target_gemini_count):
            if self.gemini_processor:
                worker = threading.Thread(
                    target=self._gemini_worker,
                    daemon=True,
                    name=f"gemini-worker-{i}")
                worker.start()
                self.gemini_workers.append(worker)

        print(f"Updated processor state: LOCAL={local_enabled} ({len(self.local_workers)} workers), "
              f"REPLICATE={replicate_enabled} ({len(self.replicate_workers)} workers), "
              f"GEMINI={gemini_enabled} ({len(self.gemini_workers)} workers)")


# Global queue manager instance
queue_manager = QueueManager()

# Use this to enable only Gemini processing
queue_manager.set_processor_state(
    local_enabled=False, replicate_enabled=False, gemini_enabled=True)
