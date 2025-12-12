# Linux Kernel Chronological Intelligence Engine

A high-performance data engineering pipeline to extract, store, and structure 30+ years of Linux kernel commit history for LLM instruction tuning.

## Overview

This project implements a comprehensive framework for:

- **Extracting** 1.3M+ commits from the Linux kernel repository (including pre-Git history)
- **Storing** file-level diffs and metadata in ClickHouse for analytical querying
- **Exporting** structured training data in JSONL format for LLM fine-tuning

## Features

- 🚀 **High-Performance Extraction**: Uses `pygit2` (libgit2 bindings) for direct ODB access
- ⚡ **Parallel Processing**: Producer-consumer multiprocessing pipeline
- 📊 **Optimized Storage**: ClickHouse columnar database with ZSTD compression
- 🎯 **File-Level Granularity**: Per-file diffs with code context (before/after)
- 📝 **Instruction Tuning Ready**: CommitPack/OctoPack compatible JSONL export

## Requirements

- Python 3.10+
- ClickHouse (via Docker or native installation)
- Git
- 64GB+ RAM (recommended for full kernel extraction)
- 1TB+ NVMe SSD storage

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/linux-kernel-ai-data.git
cd linux-kernel-ai-data
pip install -r requirements.txt

# Start ClickHouse
docker-compose up -d

# Initialize database
python scripts/init_db.py

# Run extraction (starts with kernel clone)
python scripts/run_extraction.py

# Export training data
python scripts/export_training_data.py --output training_data.jsonl
```

## Project Structure

```
├── config/settings.py      # Configuration management
├── schema/clickhouse.sql   # Database DDL
├── src/
│   ├── repository.py       # Git repository management
│   ├── extractor.py        # Commit/diff extraction
│   ├── pipeline.py         # Multiprocessing orchestration
│   ├── writer.py           # ClickHouse batch writer
│   └── exporter.py         # JSONL training data export
├── scripts/                # Runner scripts
└── tests/                  # Unit tests
```

## License

MIT License
