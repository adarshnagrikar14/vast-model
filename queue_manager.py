import os
import time
import uuid
import queue
import threading
from enum import Enum
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple, Callable

load_dotenv()

MAX_LOCAL_CONCURRENCY = int(os.environ.get("MAX_LOCAL_CONCURRENCY", "1"))
MAX_REPLICATE_CONCURRENCY = int(
    os.environ.get("MAX_REPLICATE_CONCURRENCY", "1"))
MAX_GEMINI_CONCURRENCY = int(os.environ.get("MAX_GEMINI_CONCURRENCY", "8"))


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

        # Worker threads
        self.local_worker_thread = threading.Thread(
            target=self._local_worker, daemon=True)
        self.replicate_worker_thread = threading.Thread(
            target=self._replicate_worker, daemon=True)
        self.gemini_worker_thread = threading.Thread(
            target=self._gemini_worker, daemon=True)

        # State tracking
        self._stop_event = threading.Event()
        self._is_running = False

        # Processing callbacks
        self.local_processor = None
        self.replicate_processor = None
        self.gemini_processor = None

    def start(self, local_processor: Callable, replicate_processor: Callable, gemini_processor: Callable = None):
        """Start the queue manager with processor functions"""
        if self._is_running:
            return

        self.local_processor = local_processor
        self.replicate_processor = replicate_processor
        self.gemini_processor = gemini_processor

        self._is_running = True
        self.local_worker_thread.start()
        self.replicate_worker_thread.start()
        if gemini_processor:
            self.gemini_worker_thread.start()

        print(
            f"Queue manager started with {MAX_LOCAL_CONCURRENCY} local workers, {MAX_REPLICATE_CONCURRENCY} replicate worker(s), and {MAX_GEMINI_CONCURRENCY} Gemini worker(s)")

    def stop(self):
        """Stop the queue manager gracefully"""
        self._stop_event.set()
        self._is_running = False
        self.local_worker_thread.join(timeout=5)
        self.replicate_worker_thread.join(timeout=5)
        self.gemini_worker_thread.join(timeout=5)
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

    def _try_next_job(self, processor_type: ProcessorType):
        """Try to get the next job from the queue"""
        # Use non-blocking get to avoid locking
        try:
            for _ in range(self.job_queue.qsize()):
                job = self.job_queue.get(block=False)
                self._process_job(job, processor_type)
                self.job_queue.task_done()
                return True
        except queue.Empty:
            return False

    def _local_worker(self):
        """Worker thread for local processing"""
        while not self._stop_event.is_set():
            if self.local_semaphore.acquire(blocking=False):
                try:
                    if self._try_next_job(ProcessorType.LOCAL):
                        continue
                finally:
                    self.local_semaphore.release()
            time.sleep(0.1)

    def _replicate_worker(self):
        """Worker thread for replicate processing"""
        while not self._stop_event.is_set():
            if self.replicate_semaphore.acquire(blocking=False):
                try:
                    if self._try_next_job(ProcessorType.REPLICATE):
                        continue
                finally:
                    self.replicate_semaphore.release()
            time.sleep(0.1)

    def _gemini_worker(self):
        """Worker thread for Gemini processing"""
        while not self._stop_event.is_set():
            if self.gemini_semaphore.acquire(blocking=False):
                try:
                    if self._try_next_job(ProcessorType.GEMINI):
                        continue
                finally:
                    self.gemini_semaphore.release()
            time.sleep(0.1)

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


# Global queue manager instance
queue_manager = QueueManager()
