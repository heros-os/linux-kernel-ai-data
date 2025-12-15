# Linux Kernel AI Training Data Pipeline

Extract, filter, and prepare the entire Linux kernel git history for LLM fine-tuning.

## Features

- **Full History Extraction**: 1.4M+ commits with diffs and metadata
- **Quality Scoring**: Heuristic + AI-based filtering
- **SOTA Enhancements**: Commit chains, author expertise, revert detection
- **Multiple Export Modes**: Review, BugFix, Security specialized datasets

## Quick Start

```powershell
# 1. Start ClickHouse
docker-compose up -d

# 2. Clone kernel and extract (auto-resumes on failure)
python scripts/run_extraction.py --clone --batch-size 50 --workers 8 --auto-resume

# 3. Post-process
python scripts/classify_commits.py
python scripts/build_commit_chains.py
python scripts/build_author_expertise.py

# 4. Export training data
python scripts/export_training_data.py -o exports/kernel_training.jsonl --min-quality 50

# 5. Validate
python scripts/validate_data.py exports/kernel_training.jsonl
```

## Project Structure

```
├── config/             # Settings (ClickHouse, extraction params)
├── schema/             # ClickHouse SQL schema
├── src/
│   ├── extractor.py    # Commit/diff extraction
│   ├── pipeline.py     # Parallel processing
│   ├── writer.py       # Database writer
│   ├── exporter.py     # JSONL export
│   └── quality_scorer.py
├── scripts/
│   ├── run_extraction.py
│   ├── classify_commits.py
│   ├── build_commit_chains.py
│   ├── build_author_expertise.py
│   ├── export_specialized.py
│   ├── score_commits.py
│   └── validate_data.py
├── configs/
│   └── axolotl_kernel.yaml  # Ready-to-use training config
└── tests/
```

## Export Modes

| Mode     | Command                                 | Use Case                |
| -------- | --------------------------------------- | ----------------------- |
| Standard | `export_training_data.py`               | General fine-tuning     |
| Security | `export_specialized.py --mode security` | CVE/vulnerability fixes |
| BugFix   | `export_specialized.py --mode bugfix`   | Bug→Fix pairs           |
| Review   | `export_specialized.py --mode review`   | Code review training    |

## Filtering Options

```powershell
# High quality only
python scripts/export_training_data.py --min-quality 60 --min-ai-score 4

# Security commits only
python scripts/export_training_data.py --tags security

# Exclude duplicates (backports)
python scripts/export_training_data.py --exclude-tags backport

# Specific subsystem
python scripts/export_training_data.py --subsystems mm kernel/sched
```

## Training with Axolotl

```bash
# In WSL with GPU
accelerate launch -m axolotl.cli.train configs/axolotl_kernel.yaml
```

## Requirements

- Python 3.10+
- ClickHouse (via Docker)
- ~100GB storage for full extraction
- 16GB+ RAM recommended

## License

MIT
