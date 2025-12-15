"""
Diff extraction engine for the Linux Kernel History Pipeline.

Extracts file-level diffs and code context from Git commits using pygit2
for high-performance direct ODB (Object Database) access.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import chardet
import pygit2

from config.settings import settings
from src.quality_scorer import QualityScorer

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Type of file change in a commit."""
    ADDED = "A"
    MODIFIED = "M"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"


@dataclass
class FileChange:
    """Represents a single file change in a commit."""
    
    commit_hash: str
    file_path: str
    old_file_path: Optional[str]
    change_type: ChangeType
    file_extension: str
    subsystem: str
    diff_hunk: str
    code_before: str
    code_after: str
    lines_added: int
    lines_deleted: int
    is_binary: bool = False


@dataclass
class CommitData:
    """Represents extracted commit metadata."""
    
    commit_hash: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    author_date: int  # Unix timestamp
    commit_date: int  # Unix timestamp
    subject: str
    body: str
    is_merge: bool
    parent_count: int
    files_changed: int = 0
    total_additions: int = 0
    total_deletions: int = 0
    fixes_hash: Optional[str] = None
    signed_off_by: list[str] = field(default_factory=list)
    reviewed_by: list[str] = field(default_factory=list)
    acked_by: list[str] = field(default_factory=list)
    heuristic_score: float = 0.0
    quality_flags: int = 0


class DiffExtractor:
    """
    Extracts diffs and metadata from Git commits.
    
    Uses pygit2 for direct Object Database access, avoiding the overhead
    of shell commands and working directory operations.
    """
    
    # Regex patterns for parsing commit message trailers
    FIXES_PATTERN = re.compile(r'^Fixes:\s*([0-9a-f]{7,40})', re.MULTILINE | re.IGNORECASE)
    SIGNED_OFF_PATTERN = re.compile(r'^Signed-off-by:\s*(.+)$', re.MULTILINE)
    REVIEWED_PATTERN = re.compile(r'^Reviewed-by:\s*(.+)$', re.MULTILINE)
    ACKED_PATTERN = re.compile(r'^Acked-by:\s*(.+)$', re.MULTILINE)
    
    # Binary file signatures
    BINARY_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.bmp',
        '.bin', '.o', '.ko', '.so', '.a',
        '.fw', '.dts', '.dtb',
        '.pdf', '.doc', '.docx',
        '.tar', '.gz', '.bz2', '.xz', '.zip'
    }
    
    def __init__(self, repo: pygit2.Repository):
        """
        Initialize the diff extractor.
        
        Args:
            repo: The pygit2.Repository instance to extract from.
        """
        self.repo = repo
        self.include_binary = settings.extraction.include_binary
        self.max_diff_size = settings.extraction.max_diff_size
        self.include_extensions = set(settings.extraction.include_extensions)
        self.exclude_paths = settings.extraction.exclude_paths
        self.scorer = QualityScorer()
    
    def extract_commit(self, commit: pygit2.Commit) -> tuple[CommitData, list[FileChange]]:
        """
        Extract full data from a commit.
        
        Args:
            commit: The pygit2.Commit object.
            
        Returns:
            Tuple of (CommitData, list of FileChange objects).
        """
        commit_hash = str(commit.id)
        
        # Parse commit message
        message = commit.message or ""
        lines = message.split('\n', 1)
        subject = lines[0].strip() if lines else ""
        body = lines[1].strip() if len(lines) > 1 else ""
        
        # Extract trailer tags
        fixes_match = self.FIXES_PATTERN.search(message)
        fixes_hash = fixes_match.group(1) if fixes_match else None
        
        signed_off_by = self.SIGNED_OFF_PATTERN.findall(message)
        reviewed_by = self.REVIEWED_PATTERN.findall(message)
        acked_by = self.ACKED_PATTERN.findall(message)
        
        # Determine if merge commit
        is_merge = len(commit.parents) > 1
        
        # Create commit data
        commit_data = CommitData(
            commit_hash=commit_hash,
            author_name=commit.author.name or "Unknown",
            author_email=commit.author.email or "",
            committer_name=commit.committer.name or "Unknown",
            committer_email=commit.committer.email or "",
            author_date=commit.author.time,
            commit_date=commit.committer.time,
            subject=subject,
            body=body,
            is_merge=is_merge,
            parent_count=len(commit.parents),
            fixes_hash=fixes_hash,
            signed_off_by=signed_off_by,
            reviewed_by=reviewed_by,
            acked_by=acked_by
        )
        
        # Calculate heuristic score
        commit_data.heuristic_score = self.scorer.calculate_heuristic_score(
            subject=subject,
            body=body,
            files_changed=0,  # Will update after file extraction
            signed_off_by=signed_off_by,
            reviewed_by=reviewed_by,
            acked_by=acked_by,
            fixes_hash=fixes_hash
        )
        
        # Extract file changes
        file_changes = self._extract_file_changes(commit)
        
        # Update statistics
        commit_data.files_changed = len(file_changes)
        commit_data.total_additions = sum(fc.lines_added for fc in file_changes)
        commit_data.total_additions = sum(fc.lines_added for fc in file_changes)
        commit_data.total_deletions = sum(fc.lines_deleted for fc in file_changes)
        
        # Update score with file count
        commit_data.heuristic_score = self.scorer.calculate_heuristic_score(
            subject=subject,
            body=body,
            files_changed=commit_data.files_changed,
            signed_off_by=signed_off_by,
            reviewed_by=reviewed_by,
            acked_by=acked_by,
            fixes_hash=fixes_hash
        )
        
        return commit_data, file_changes
    
    def _extract_file_changes(self, commit: pygit2.Commit) -> list[FileChange]:
        """
        Extract per-file changes from a commit.
        
        Args:
            commit: The pygit2.Commit object.
            
        Returns:
            List of FileChange objects.
        """
        commit_hash = str(commit.id)
        file_changes = []
        
        # Get the parent tree (or empty tree for initial commit)
        if commit.parents:
            parent = commit.parents[0]
            parent_tree = parent.tree
        else:
            # Initial commit - diff against empty tree
            parent_tree = None
        
        # Generate diff with rename detection
        diff_flags = pygit2.GIT_DIFF_PATIENCE
        find_flags = pygit2.GIT_DIFF_FIND_RENAMES | pygit2.GIT_DIFF_FIND_COPIES
        
        try:
            if parent_tree:
                diff = self.repo.diff(parent_tree, commit.tree, flags=diff_flags)
            else:
                diff = commit.tree.diff_to_tree(flags=diff_flags)
            
            diff.find_similar(flags=find_flags)
        except Exception as e:
            logger.warning(f"Failed to generate diff for {commit_hash[:12]}: {e}")
            return []
        
        # Process each file delta
        for patch in diff:
            try:
                fc = self._process_patch(commit_hash, patch, commit, parent_tree)
                if fc:
                    file_changes.append(fc)
            except Exception as e:
                logger.debug(f"Failed to process patch in {commit_hash[:12]}: {e}")
        
        return file_changes
    
    def _process_patch(
        self,
        commit_hash: str,
        patch: pygit2.Patch,
        commit: pygit2.Commit,
        parent_tree: Optional[pygit2.Tree]
    ) -> Optional[FileChange]:
        """
        Process a single file patch.
        
        Args:
            commit_hash: The commit SHA-1.
            patch: The pygit2.Patch object.
            commit: The commit object.
            parent_tree: The parent commit's tree (or None).
            
        Returns:
            FileChange object or None if filtered.
        """
        delta = patch.delta
        
        # Determine file path and change type
        if delta.status == pygit2.GIT_DELTA_ADDED:
            file_path = delta.new_file.path
            old_file_path = None
            change_type = ChangeType.ADDED
        elif delta.status == pygit2.GIT_DELTA_DELETED:
            file_path = delta.old_file.path
            old_file_path = None
            change_type = ChangeType.DELETED
        elif delta.status == pygit2.GIT_DELTA_RENAMED:
            file_path = delta.new_file.path
            old_file_path = delta.old_file.path
            change_type = ChangeType.RENAMED
        elif delta.status == pygit2.GIT_DELTA_COPIED:
            file_path = delta.new_file.path
            old_file_path = delta.old_file.path
            change_type = ChangeType.COPIED
        else:
            file_path = delta.new_file.path
            old_file_path = None
            change_type = ChangeType.MODIFIED
        
        # Check if binary
        is_binary = delta.is_binary
        file_ext = self._get_extension(file_path)
        
        if file_ext.lower() in self.BINARY_EXTENSIONS:
            is_binary = True
        
        if is_binary and not self.include_binary:
            return None
        
        # Check exclude paths
        for exclude_path in self.exclude_paths:
            if file_path.startswith(exclude_path):
                return None
        
        # Get diff hunk text
        diff_hunk = ""
        lines_added = 0
        lines_deleted = 0
        
        if not is_binary:
            try:
                diff_hunk = patch.text or ""
                
                # Check diff size limit
                if len(diff_hunk) > self.max_diff_size:
                    logger.debug(f"Diff too large for {file_path}, truncating")
                    diff_hunk = diff_hunk[:self.max_diff_size] + "\n... [TRUNCATED]"
                
                # Count lines
                for line in diff_hunk.split('\n'):
                    if line.startswith('+') and not line.startswith('+++'):
                        lines_added += 1
                    elif line.startswith('-') and not line.startswith('---'):
                        lines_deleted += 1
                        
            except Exception as e:
                logger.debug(f"Failed to get diff text for {file_path}: {e}")
        
        # Get code before and after
        code_before = ""
        code_after = ""
        
        if not is_binary:
            code_before = self._get_blob_content(parent_tree, delta.old_file.path) if parent_tree else ""
            code_after = self._get_blob_content(commit.tree, file_path)
        
        # Extract subsystem from path
        subsystem = self._extract_subsystem(file_path)
        
        return FileChange(
            commit_hash=commit_hash,
            file_path=file_path,
            old_file_path=old_file_path,
            change_type=change_type,
            file_extension=file_ext,
            subsystem=subsystem,
            diff_hunk=diff_hunk,
            code_before=code_before,
            code_after=code_after,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            is_binary=is_binary
        )
    
    def _get_blob_content(self, tree: pygit2.Tree, path: str) -> str:
        """
        Get the content of a file blob from a tree.
        
        Args:
            tree: The tree object.
            path: The file path within the tree.
            
        Returns:
            The file content as a string, or empty string on error.
        """
        if not tree or not path:
            return ""
        
        try:
            entry = tree[path]
            blob = self.repo.get(entry.id)
            
            if not isinstance(blob, pygit2.Blob):
                return ""
            
            data = blob.data
            
            if len(data) > 100000:
                data = data[:100000]
            
            # Detect encoding
            if isinstance(data, bytes):
                # Check for binary content
                if b'\x00' in data[:8000]:
                    return ""
                
                # Try UTF-8 first
                try:
                    return data.decode('utf-8')
                except UnicodeDecodeError:
                    # Fallback to chardet
                    try:
                        detected = chardet.detect(data)
                        encoding = detected.get('encoding')
                        
                        if not encoding:
                            # If chardet fails to detect, try latin-1 as last resort
                            return data.decode('latin-1', errors='replace')
                            
                        return data.decode(encoding, errors='replace')
                    except Exception as e:
                        # If detection or decoding fails entirely, treat as binary
                        logger.debug(f"Encoding detection failed: {e}")
                        return ""
            
            return data if isinstance(data, str) else ""
            
        except KeyError:
            return ""  # File doesn't exist in this tree
        except Exception as e:
            logger.debug(f"Failed to get blob content for {path}: {e}")
            return ""
    
    def _get_extension(self, path: str) -> str:
        """Extract file extension from path."""
        if '.' in path:
            return '.' + path.rsplit('.', 1)[-1]
        return ""
    
    def _extract_subsystem(self, path: str) -> str:
        """
        Extract the subsystem from a file path.
        
        Examples:
            - kernel/sched/core.c -> kernel/sched
            - mm/slub.c -> mm
            - drivers/net/ethernet/intel/e1000/e1000.c -> drivers/net
            - arch/x86/kernel/cpu.c -> arch/x86
        """
        parts = path.split('/')
        
        if len(parts) == 1:
            return "root"
        
        # Special handling for common subsystems
        if parts[0] in {'kernel', 'mm', 'fs', 'net', 'block', 'ipc', 'security', 'crypto'}:
            if len(parts) > 2:
                return '/'.join(parts[:2])
            return parts[0]
        
        if parts[0] == 'drivers':
            # drivers/net/... -> drivers/net
            if len(parts) > 2:
                return '/'.join(parts[:2])
            return parts[0]
        
        if parts[0] == 'arch':
            # arch/x86/... -> arch/x86
            if len(parts) > 2:
                return '/'.join(parts[:2])
            return parts[0]
        
        if parts[0] in {'include', 'tools', 'scripts', 'Documentation'}:
            return parts[0]
        
        return '/'.join(parts[:2]) if len(parts) > 1 else parts[0]
