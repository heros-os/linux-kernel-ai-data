-- Linux Kernel Chronological Intelligence Engine
-- ClickHouse Schema Definition
-- Database: kernel_history
-- 
-- NOTE: Tables have no partitioning to avoid "Too many partitions per INSERT" errors
-- Partitioning can be added later if needed for data lifecycle management

-- Create database
CREATE DATABASE IF NOT EXISTS kernel_history;

-- Commits table: stores commit metadata
CREATE TABLE IF NOT EXISTS kernel_history.commits
(
    -- Primary identifier
    commit_hash FixedString(40) CODEC(ZSTD(1)),
    
    -- Author information (LowCardinality for repeating maintainer names)
    author_name LowCardinality(String) CODEC(ZSTD(1)),
    author_email LowCardinality(String) CODEC(ZSTD(1)),
    
    -- Committer information (may differ from author)
    committer_name LowCardinality(String) CODEC(ZSTD(1)),
    committer_email LowCardinality(String) CODEC(ZSTD(1)),
    
    -- Timestamps
    author_date DateTime CODEC(Delta, ZSTD(1)),
    commit_date DateTime CODEC(Delta, ZSTD(1)),
    
    -- Commit message
    subject String CODEC(ZSTD(1)),
    body String CODEC(ZSTD(3)),
    
    -- Metadata flags
    is_merge UInt8 DEFAULT 0,
    parent_count UInt8 DEFAULT 1,
    
    -- Statistics
    files_changed UInt32 DEFAULT 0,
    total_additions UInt32 DEFAULT 0,
    total_deletions UInt32 DEFAULT 0,
    
    -- Trailer tags extracted from commit message
    fixes_hash Nullable(FixedString(40)) CODEC(ZSTD(1)),
    signed_off_by Array(String) CODEC(ZSTD(1)),
    reviewed_by Array(String) CODEC(ZSTD(1)),
    acked_by Array(String) CODEC(ZSTD(1)),
    
    -- Insertion timestamp
    inserted_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (commit_hash)
SETTINGS index_granularity = 8192;

-- File changes table: stores per-file diffs (the training signal)
CREATE TABLE IF NOT EXISTS kernel_history.file_changes
(
    -- Foreign key to commits
    commit_hash FixedString(40) CODEC(ZSTD(1)),
    
    -- File identification
    file_path LowCardinality(String) CODEC(ZSTD(1)),
    old_file_path Nullable(String) CODEC(ZSTD(1)),
    
    -- Change type: 'A' (Add), 'M' (Modify), 'D' (Delete), 'R' (Rename), 'C' (Copy)
    change_type Enum8('A' = 1, 'M' = 2, 'D' = 3, 'R' = 4, 'C' = 5),
    
    -- File type classification
    file_extension LowCardinality(String) CODEC(ZSTD(1)),
    subsystem LowCardinality(String) CODEC(ZSTD(1)),
    
    -- The diff content
    diff_hunk String CODEC(ZSTD(3)),
    
    -- Code context for training
    code_before String CODEC(ZSTD(3)),
    code_after String CODEC(ZSTD(3)),
    
    -- Statistics
    lines_added UInt32 CODEC(T64, ZSTD(1)),
    lines_deleted UInt32 CODEC(T64, ZSTD(1)),
    
    -- Binary file flag
    is_binary UInt8 DEFAULT 0,
    
    -- Insertion timestamp
    inserted_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (commit_hash, file_path)
SETTINGS index_granularity = 8192;

-- Index for fast commit message search (optional, run after data import)
-- ALTER TABLE kernel_history.commits ADD INDEX idx_subject_search subject TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4;

-- Index for file path filtering (optional, run after data import)
-- ALTER TABLE kernel_history.file_changes ADD INDEX idx_file_path file_path TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4;

