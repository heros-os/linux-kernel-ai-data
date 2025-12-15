"""
Producer-Consumer multiprocessing pipeline for extracting Linux kernel history.

Implements a parallel extraction pipeline using Python's multiprocessing
module to efficiently process 1.3M+ commits.
"""

import logging
import multiprocessing as mp
from multiprocessing import Queue, Process, Event
from queue import Empty
from typing import Optional, Callable
from dataclasses import dataclass
import json
from datetime import datetime
import time

import pygit2
from tqdm import tqdm

from config.settings import settings
from src.extractor import DiffExtractor, CommitData, FileChange

logger = logging.getLogger(__name__)


@dataclass
class WorkBatch:
    """A batch of commit hashes to process."""
    batch_id: int
    commit_hashes: list[str]


@dataclass
class ResultBatch:
    """Results from processing a batch of commits."""
    batch_id: int
    commits: list[CommitData]
    file_changes: list[FileChange]
    errors: list[str]


def worker_process(
    repo_path: str,
    work_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
    worker_id: int
) -> None:
    """
    Worker process function for parallel extraction.
    
    Each worker maintains its own Repository instance (pygit2 handles
    are not thread/process safe) and processes batches of commits.
    
    Args:
        repo_path: Path to the Git repository.
        work_queue: Queue to receive WorkBatch from.
        result_queue: Queue to send ResultBatch to.
        stop_event: Event to signal shutdown.
        worker_id: Identifier for this worker.
    """
    # Initialize worker-local repository
    try:
        repo = pygit2.Repository(repo_path)
        extractor = DiffExtractor(repo)
    except Exception as e:
        logger.error(f"Worker {worker_id}: Failed to initialize repository: {e}")
        return
    
    logger.debug(f"Worker {worker_id}: Initialized and ready")
    
    while not stop_event.is_set():
        try:
            # Get next batch with timeout
            batch: WorkBatch = work_queue.get(timeout=1.0)
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id}: Queue error: {e}")
            break
        
        # Process the batch
        commits = []
        file_changes = []
        errors = []
        
        for commit_hash in batch.commit_hashes:
            try:
                oid = pygit2.Oid(hex=commit_hash)
                commit = repo.get(oid)
                
                if commit is None or not isinstance(commit, pygit2.Commit):
                    errors.append(f"Invalid commit: {commit_hash}")
                    continue
                
                commit_data, changes = extractor.extract_commit(commit)
                commits.append(commit_data)
                file_changes.extend(changes)
                
            except Exception as e:
                errors.append(f"{commit_hash}: {str(e)}")
        
        # Send results
        result = ResultBatch(
            batch_id=batch.batch_id,
            commits=commits,
            file_changes=file_changes,
            errors=errors
        )
        
        try:
            result_queue.put(result)
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to send results: {e}")
    
    logger.debug(f"Worker {worker_id}: Shutting down")


class ExtractionPipeline:
    """
    Orchestrates the parallel extraction of commit history.
    
    Uses a producer-consumer pattern with:
    - Planner: Generates batches of commit hashes
    - Workers: Process commits in parallel
    - Writer: Collects results and writes to database
    """
    
    def __init__(
        self,
        repo_path: str,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        on_batch_complete: Optional[Callable[[ResultBatch], None]] = None
    ):
        """
        Initialize the extraction pipeline.
        
        Args:
            repo_path: Path to the Git repository.
            num_workers: Number of worker processes (default: auto-detect).
            batch_size: Commits per batch (default: from settings).
            on_batch_complete: Callback for each completed batch.
        """
        self.repo_path = repo_path
        self.num_workers = num_workers or settings.extraction.effective_workers
        self.batch_size = batch_size or settings.extraction.batch_size
        self.on_batch_complete = on_batch_complete
        
        # Multiprocessing components
        self.work_queue: Optional[Queue] = None
        self.result_queue: Optional[Queue] = None
        self.stop_event: Optional[Event] = None
        self.workers: list[Process] = []
        
        # Statistics
        self.commits_processed = 0
        self.files_processed = 0
        self.errors_count = 0
    
    def _create_batches(self, commit_hashes: list[str]) -> list[WorkBatch]:
        """Split commit hashes into batches."""
        batches = []
        for i in range(0, len(commit_hashes), self.batch_size):
            batch = WorkBatch(
                batch_id=i // self.batch_size,
                commit_hashes=commit_hashes[i:i + self.batch_size]
            )
            batches.append(batch)
        return batches
    
    def _start_workers(self) -> None:
        """Start worker processes."""
        self.work_queue = mp.Queue(maxsize=settings.extraction.queue_size)
        self.result_queue = mp.Queue()
        self.stop_event = mp.Event()
        
        for i in range(self.num_workers):
            worker = Process(
                target=worker_process,
                args=(
                    self.repo_path,
                    self.work_queue,
                    self.result_queue,
                    self.stop_event,
                    i
                ),
                name=f"ExtractWorker-{i}"
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} worker processes")
    
    def _stop_workers(self) -> None:
        """Stop all worker processes."""
        if self.stop_event:
            self.stop_event.set()
        
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
        
        self.workers.clear()
        logger.info("All workers stopped")
    
    def run(
        self,
        commit_hashes: list[str],
        resume_from: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> tuple[int, int, int]:
        """
        Run the extraction pipeline.
        
        Args:
            commit_hashes: List of commit hashes to process.
            resume_from: Hash to resume from (skip earlier commits).
            progress_callback: Called with (processed, total) counts.
            
        Returns:
            Tuple of (commits_processed, files_processed, errors_count).
        """
        # Handle resume
        if resume_from:
            try:
                resume_idx = commit_hashes.index(resume_from)
                commit_hashes = commit_hashes[resume_idx:]
                logger.info(f"Resuming from commit {resume_from}, {len(commit_hashes)} remaining")
            except ValueError:
                logger.warning(f"Resume commit {resume_from} not found, starting from beginning")
        
        total_commits = len(commit_hashes)
        batches = self._create_batches(commit_hashes)
        total_batches = len(batches)
        
        logger.info(f"Processing {total_commits} commits in {total_batches} batches")
        logger.info(f"Batch size: {self.batch_size}, Workers: {self.num_workers}")
        
        # Start workers
        self._start_workers()
        
        try:
            # Track progress
            batches_sent = 0
            batches_received = 0
            
            # Start producer thread to keep queues flowing
            def producer():
                for batch_item in batches:
                    self.work_queue.put(batch_item)
                
            import threading
            producer_thread = threading.Thread(target=producer, daemon=True)
            producer_thread.start()
            
            # Consumer: Collect results
            with tqdm(total=total_commits, desc="Extracting commits") as pbar:
                while batches_received < total_batches:
                    try:
                        result: ResultBatch = self.result_queue.get(timeout=30.0)
                        batches_received += 1
                        
                        # Update statistics
                        self.commits_processed += len(result.commits)
                        self.files_processed += len(result.file_changes)
                        self.errors_count += len(result.errors)
                        
                        # Log errors
                        for error in result.errors:
                            logger.debug(f"Extraction error: {error}")
                        
                        # Callback for writing
                        if self.on_batch_complete:
                            self.on_batch_complete(result)
                        
                        # Update progress
                        pbar.update(len(result.commits))
                        
                        # Calculate speed and estimated completion
                        elapsed = time.time() - pbar.start_t
                        speed = self.commits_processed / elapsed if elapsed > 0 else 0
                        
                        pbar.set_postfix({
                            "files": self.files_processed,
                            "errors": self.errors_count,
                            "speed": f"{speed:.1f}/s"
                        })

                        # Write status file for external monitoring
                        status_data = {
                            "timestamp": datetime.now().isoformat(),
                            "commits_processed": self.commits_processed,
                            "total_commits": total_commits,
                            "progress_percent": round((self.commits_processed / total_commits) * 100, 2),
                            "files_processed": self.files_processed,
                            "errors_count": self.errors_count,
                            "speed_commits_per_sec": round(speed, 2),
                            "batches_processed": batches_received,
                            "total_batches": total_batches,
                            "status": "running"
                        }
                        
                        try:
                            with open("extraction_status.json", "w") as f:
                                json.dump(status_data, f, indent=2)
                        except Exception:
                            pass  # Ignore status write errors
                        
                        if progress_callback:
                            progress_callback(self.commits_processed, total_commits)
                            
                    except Empty:
                        logger.warning("Timeout waiting for results, checking workers...")
                        # Check if workers are alive
                        alive = sum(1 for w in self.workers if w.is_alive())
                        if alive == 0:
                            logger.error("All workers died!")
                            break
                            
            # Wait for producer (it should be done if we received all batches)
            producer_thread.join(timeout=1.0)
                
        finally:
            self._stop_workers()
        
        logger.info(
            f"Extraction complete: {self.commits_processed} commits, "
            f"{self.files_processed} file changes, {self.errors_count} errors"
        )
        
        return self.commits_processed, self.files_processed, self.errors_count
    
    def run_single_threaded(
        self,
        commit_hashes: list[str],
        on_commit: Optional[Callable[[CommitData, list[FileChange]], None]] = None
    ) -> tuple[int, int, int]:
        """
        Run extraction in single-threaded mode (useful for debugging).
        
        Args:
            commit_hashes: List of commit hashes to process.
            on_commit: Callback for each processed commit.
            
        Returns:
            Tuple of (commits_processed, files_processed, errors_count).
        """
        repo = pygit2.Repository(self.repo_path)
        extractor = DiffExtractor(repo)
        
        commits_processed = 0
        files_processed = 0
        errors_count = 0
        
        with tqdm(commit_hashes, desc="Extracting (single-threaded)") as pbar:
            for commit_hash in pbar:
                try:
                    oid = pygit2.Oid(hex=commit_hash)
                    commit = repo.get(oid)
                    
                    if commit is None:
                        errors_count += 1
                        continue
                    
                    commit_data, file_changes = extractor.extract_commit(commit)
                    
                    commits_processed += 1
                    files_processed += len(file_changes)
                    
                    if on_commit:
                        on_commit(commit_data, file_changes)
                        
                except Exception as e:
                    logger.debug(f"Error processing {commit_hash}: {e}")
                    errors_count += 1
        
        return commits_processed, files_processed, errors_count
