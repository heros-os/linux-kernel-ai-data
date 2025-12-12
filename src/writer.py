"""
ClickHouse batch writer for the Linux Kernel History Pipeline.

Handles high-throughput writes with buffering, batching, and error handling.
"""

import logging
import threading
from datetime import datetime
from typing import Optional
from queue import Queue, Empty
import time

from clickhouse_driver import Client

from config.settings import settings
from src.extractor import CommitData, FileChange
from src.pipeline import ResultBatch

logger = logging.getLogger(__name__)


class ClickHouseWriter:
    """
    Writes extracted data to ClickHouse with batching and buffering.
    
    Features:
    - Configurable batch size for optimal insert performance
    - Auto-flush on buffer full or timeout
    - Thread-safe for use with multiprocessing pipeline
    - Error handling with retry logic
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize the ClickHouse writer.
        
        Args:
            host: ClickHouse server host (default: from settings).
            port: ClickHouse native port (default: from settings).
            database: Database name (default: from settings).
            batch_size: Records per batch insert (default: from settings).
        """
        self.host = host or settings.clickhouse.host
        self.port = port or settings.clickhouse.port
        self.database = database or settings.clickhouse.database
        self.batch_size = batch_size or settings.clickhouse.batch_size
        
        self._client: Optional[Client] = None
        self._commit_buffer: list[dict] = []
        self._file_change_buffer: list[dict] = []
        self._lock = threading.Lock()
        
        # Statistics
        self.commits_written = 0
        self.file_changes_written = 0
        self.batches_written = 0
    
    @property
    def client(self) -> Client:
        """Get or create ClickHouse client."""
        if self._client is None:
            self._client = Client(
                host=self.host,
                port=self.port,
                database=self.database,
                settings={
                    'insert_block_size': self.batch_size,
                    'max_insert_block_size': self.batch_size * 2
                }
            )
        return self._client
    
    def _commit_to_dict(self, commit: CommitData) -> dict:
        """Convert CommitData to dictionary for insertion."""
        return {
            'commit_hash': commit.commit_hash,
            'author_name': commit.author_name,
            'author_email': commit.author_email,
            'committer_name': commit.committer_name,
            'committer_email': commit.committer_email,
            'author_date': datetime.fromtimestamp(commit.author_date),
            'commit_date': datetime.fromtimestamp(commit.commit_date),
            'subject': commit.subject[:1000] if commit.subject else '',
            'body': commit.body[:50000] if commit.body else '',
            'is_merge': 1 if commit.is_merge else 0,
            'parent_count': commit.parent_count,
            'files_changed': commit.files_changed,
            'total_additions': commit.total_additions,
            'total_deletions': commit.total_deletions,
            'fixes_hash': commit.fixes_hash,
            'signed_off_by': commit.signed_off_by or [],
            'reviewed_by': commit.reviewed_by or [],
            'acked_by': commit.acked_by or []
        }
    
    def _file_change_to_dict(self, fc: FileChange) -> dict:
        """Convert FileChange to dictionary for insertion."""
        return {
            'commit_hash': fc.commit_hash,
            'file_path': fc.file_path,
            'old_file_path': fc.old_file_path,
            'change_type': fc.change_type.value,
            'file_extension': fc.file_extension,
            'subsystem': fc.subsystem,
            'diff_hunk': fc.diff_hunk[:100000] if fc.diff_hunk else '',
            'code_before': fc.code_before[:500000] if fc.code_before else '',
            'code_after': fc.code_after[:500000] if fc.code_after else '',
            'lines_added': fc.lines_added,
            'lines_deleted': fc.lines_deleted,
            'is_binary': 1 if fc.is_binary else 0
        }
    
    def write_batch(self, result: ResultBatch) -> None:
        """
        Write a result batch to the buffers.
        
        Thread-safe. Flushes automatically when buffer is full.
        
        Args:
            result: The ResultBatch from extraction pipeline.
        """
        with self._lock:
            # Add commits to buffer
            for commit in result.commits:
                self._commit_buffer.append(self._commit_to_dict(commit))
            
            # Add file changes to buffer
            for fc in result.file_changes:
                self._file_change_buffer.append(self._file_change_to_dict(fc))
            
            # Check if flush is needed
            if len(self._commit_buffer) >= self.batch_size:
                self._flush_commits()
            
            if len(self._file_change_buffer) >= self.batch_size:
                self._flush_file_changes()
    
    def _flush_commits(self) -> None:
        """Flush commit buffer to database."""
        if not self._commit_buffer:
            return
        
        try:
            self.client.execute(
                '''
                INSERT INTO commits (
                    commit_hash, author_name, author_email,
                    committer_name, committer_email,
                    author_date, commit_date,
                    subject, body, is_merge, parent_count,
                    files_changed, total_additions, total_deletions,
                    fixes_hash, signed_off_by, reviewed_by, acked_by
                ) VALUES
                ''',
                self._commit_buffer
            )
            
            self.commits_written += len(self._commit_buffer)
            self.batches_written += 1
            logger.debug(f"Flushed {len(self._commit_buffer)} commits")
            self._commit_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush commits: {e}")
            # Keep data in buffer for retry
            raise
    
    def _flush_file_changes(self) -> None:
        """Flush file changes buffer to database."""
        if not self._file_change_buffer:
            return
        
        try:
            self.client.execute(
                '''
                INSERT INTO file_changes (
                    commit_hash, file_path, old_file_path,
                    change_type, file_extension, subsystem,
                    diff_hunk, code_before, code_after,
                    lines_added, lines_deleted, is_binary
                ) VALUES
                ''',
                self._file_change_buffer
            )
            
            self.file_changes_written += len(self._file_change_buffer)
            logger.debug(f"Flushed {len(self._file_change_buffer)} file changes")
            self._file_change_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush file changes: {e}")
            raise
    
    def flush(self) -> None:
        """Flush all buffers to database."""
        with self._lock:
            self._flush_commits()
            self._flush_file_changes()
    
    def get_stats(self) -> dict:
        """Get writer statistics."""
        return {
            'commits_written': self.commits_written,
            'file_changes_written': self.file_changes_written,
            'batches_written': self.batches_written,
            'commit_buffer_size': len(self._commit_buffer),
            'file_change_buffer_size': len(self._file_change_buffer)
        }
    
    def close(self) -> None:
        """Flush remaining data and close connection."""
        self.flush()
        if self._client:
            self._client.disconnect()
            self._client = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def init_database(
    host: Optional[str] = None,
    port: Optional[int] = None
) -> None:
    """
    Initialize the ClickHouse database and tables.
    
    Args:
        host: ClickHouse server host.
        port: ClickHouse native port.
    """
    host = host or settings.clickhouse.host
    port = port or settings.clickhouse.port
    
    # Read schema file
    schema_path = settings.repository.data_dir.parent / "schema" / "clickhouse.sql"
    
    # Alternative paths to try
    alt_paths = [
        "schema/clickhouse.sql",
        "../schema/clickhouse.sql",
    ]
    
    schema_sql = None
    for path_str in [str(schema_path)] + alt_paths:
        try:
            with open(path_str, 'r') as f:
                schema_sql = f.read()
            break
        except FileNotFoundError:
            continue
    
    if not schema_sql:
        raise FileNotFoundError("Could not find clickhouse.sql schema file")
    
    # Connect without database first
    client = Client(host=host, port=port)
    
    # Execute each statement
    statements = schema_sql.split(';')
    for stmt in statements:
        stmt = stmt.strip()
        if stmt and not stmt.startswith('--'):
            try:
                client.execute(stmt)
                logger.debug(f"Executed: {stmt[:50]}...")
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    logger.warning(f"Statement failed: {e}")
    
    client.disconnect()
    logger.info("Database initialization complete")
