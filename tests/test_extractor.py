"""Tests for the extraction module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import DiffExtractor, ChangeType, FileChange, CommitData


class TestDiffExtractor:
    """Tests for DiffExtractor class."""
    
    def test_extract_subsystem_kernel(self):
        """Test subsystem extraction for kernel paths."""
        extractor = DiffExtractor.__new__(DiffExtractor)
        extractor.include_binary = False
        extractor.max_diff_size = 100000
        extractor.include_extensions = {'.c', '.h'}
        extractor.exclude_paths = []
        
        assert extractor._extract_subsystem("kernel/sched/core.c") == "kernel/sched"
        assert extractor._extract_subsystem("kernel/fork.c") == "kernel"
        assert extractor._extract_subsystem("mm/slub.c") == "mm"
        assert extractor._extract_subsystem("mm/page_alloc.c") == "mm"
    
    def test_extract_subsystem_drivers(self):
        """Test subsystem extraction for driver paths."""
        extractor = DiffExtractor.__new__(DiffExtractor)
        extractor.include_binary = False
        extractor.max_diff_size = 100000
        extractor.include_extensions = {'.c', '.h'}
        extractor.exclude_paths = []
        
        assert extractor._extract_subsystem("drivers/net/ethernet/intel/e1000/e1000.c") == "drivers/net"
        assert extractor._extract_subsystem("drivers/gpu/drm/i915/i915.c") == "drivers/gpu"
    
    def test_extract_subsystem_arch(self):
        """Test subsystem extraction for architecture paths."""
        extractor = DiffExtractor.__new__(DiffExtractor)
        extractor.include_binary = False
        extractor.max_diff_size = 100000
        extractor.include_extensions = {'.c', '.h'}
        extractor.exclude_paths = []
        
        assert extractor._extract_subsystem("arch/x86/kernel/cpu.c") == "arch/x86"
        assert extractor._extract_subsystem("arch/arm64/boot/dts/foo.dts") == "arch/arm64"
    
    def test_get_extension(self):
        """Test file extension extraction."""
        extractor = DiffExtractor.__new__(DiffExtractor)
        
        assert extractor._get_extension("kernel/fork.c") == ".c"
        assert extractor._get_extension("include/linux/sched.h") == ".h"
        assert extractor._get_extension("Makefile") == ""
        assert extractor._get_extension("arch/x86/entry/entry_64.S") == ".S"
    
    def test_binary_extension_detection(self):
        """Test binary file extension detection."""
        assert ".png" in DiffExtractor.BINARY_EXTENSIONS
        assert ".bin" in DiffExtractor.BINARY_EXTENSIONS
        assert ".fw" in DiffExtractor.BINARY_EXTENSIONS
        assert ".c" not in DiffExtractor.BINARY_EXTENSIONS


class TestChangeType:
    """Tests for ChangeType enum."""
    
    def test_change_type_values(self):
        """Test change type enum values."""
        assert ChangeType.ADDED.value == "A"
        assert ChangeType.MODIFIED.value == "M"
        assert ChangeType.DELETED.value == "D"
        assert ChangeType.RENAMED.value == "R"
        assert ChangeType.COPIED.value == "C"


class TestFileChange:
    """Tests for FileChange dataclass."""
    
    def test_file_change_creation(self):
        """Test FileChange creation."""
        fc = FileChange(
            commit_hash="abc123" + "0" * 34,
            file_path="kernel/fork.c",
            old_file_path=None,
            change_type=ChangeType.MODIFIED,
            file_extension=".c",
            subsystem="kernel",
            diff_hunk="--- a/kernel/fork.c\n+++ b/kernel/fork.c\n@@ -1 +1 @@\n-old\n+new",
            code_before="old",
            code_after="new",
            lines_added=1,
            lines_deleted=1,
            is_binary=False
        )
        
        assert fc.file_path == "kernel/fork.c"
        assert fc.change_type == ChangeType.MODIFIED
        assert fc.lines_added == 1


class TestCommitData:
    """Tests for CommitData dataclass."""
    
    def test_commit_data_creation(self):
        """Test CommitData creation."""
        commit = CommitData(
            commit_hash="abc123" + "0" * 34,
            author_name="Linus Torvalds",
            author_email="torvalds@linux-foundation.org",
            committer_name="Linus Torvalds",
            committer_email="torvalds@linux-foundation.org",
            author_date=1234567890,
            commit_date=1234567890,
            subject="kernel: Fix memory leak in fork()",
            body="This fixes a memory leak that occurs when...",
            is_merge=False,
            parent_count=1,
            fixes_hash="def456" + "0" * 34
        )
        
        assert commit.author_name == "Linus Torvalds"
        assert commit.is_merge is False
        assert commit.fixes_hash is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
