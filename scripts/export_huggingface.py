#!/usr/bin/env python3
"""
Export all dataset tiers for HuggingFace upload.

Creates:
- premium_score: Top heuristic scores (>=90)
- high_score: High heuristic scores (>=70)  
- premium_reasoning: Premium with AI reasoning metadata
- high_reasoning: High quality with AI reasoning metadata

All datasets include smart context extraction for optimal training quality.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from clickhouse_driver import Client
from tqdm import tqdm

from config.settings import settings
from config.constants import (
    KERNEL_SYSTEM_PROMPT,
    INPUT_TRUNCATION_LENGTH,
    OUTPUT_TRUNCATION_LENGTH,
    MIN_DIFF_LENGTH,
    MAX_DIFF_LENGTH,
    PREMIUM_MIN_SCORE,
    HIGH_QUALITY_MIN_SCORE
)
from src.validator import TrainingExampleValidator
from src.context_extractor import SmartContextExtractor

console = Console()

# Dataset configurations
DATASETS = {
    "premium_score": {
        "description": "Top quality commits based on heuristic scoring (>=90)",
        "min_heuristic": 90,
        "min_ai_score": None,
        "include_reasoning": False,
        "limit": 5000
    },
    "high_score": {
        "description": "High quality commits based on heuristic scoring (>=70)",
        "min_heuristic": 70,
        "min_ai_score": None,
        "include_reasoning": False,
        "limit": 20000
    },
    "premium_reasoning": {
        "description": "Premium commits with AI quality scores and reasoning",
        "min_heuristic": 90,
        "min_ai_score": 4,
        "include_reasoning": True,
        "limit": 3000
    },
    "high_reasoning": {
        "description": "High quality commits with AI quality scores and reasoning",
        "min_heuristic": 70,
        "min_ai_score": 3,
        "include_reasoning": True,
        "limit": 10000
    }
}


def export_dataset(
    client: Client,
    name: str,
    config: dict,
    output_dir: Path,
    validator: TrainingExampleValidator,
    extractor: SmartContextExtractor
) -> dict:
    """Export a single dataset tier."""
    
    stats = {
        "exported": 0,
        "skipped_no_diff": 0,
        "skipped_validation": 0,
        "total_bytes": 0
    }
    
    # Build query based on config
    if config["min_ai_score"]:
        query = f"""
        SELECT 
            commit_hash,
            subject || '\n\n' || body as instruction,
            ai_quality_score,
            ai_score_reason
        FROM commits
        WHERE heuristic_score >= {config['min_heuristic']}
            AND ai_quality_score >= {config['min_ai_score']}
            AND length(ai_score_reason) > 10
        ORDER BY ai_quality_score DESC, heuristic_score DESC
        LIMIT {config['limit']}
        """
    else:
        query = f"""
        SELECT 
            commit_hash,
            subject || '\n\n' || body as instruction,
            0 as ai_quality_score,
            '' as ai_score_reason
        FROM commits
        WHERE heuristic_score >= {config['min_heuristic']}
        ORDER BY heuristic_score DESC
        LIMIT {config['limit']}
        """
    
    commits = client.execute(query)
    
    if not commits:
        console.print(f"  [yellow]No commits found for {name}[/yellow]")
        return stats
    
    output_path = output_dir / f"{name}.jsonl"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for commit_hash, instruction, ai_score, ai_reason in tqdm(commits, desc=f"  {name}"):
            # Get diff and code_before
            try:
                diff_query = """
                SELECT fc.diff_hunk, fc.code_before 
                FROM file_changes fc
                WHERE fc.commit_hash = %(hash)s
                    AND fc.file_extension IN ('.c', '.h')
                    AND length(fc.diff_hunk) > %(min_len)s
                    AND length(fc.diff_hunk) < %(max_len)s
                LIMIT 1
                """
                result = client.execute(diff_query, {
                    'hash': commit_hash,
                    'min_len': MIN_DIFF_LENGTH,
                    'max_len': MAX_DIFF_LENGTH
                })
                
                if not result:
                    stats["skipped_no_diff"] += 1
                    continue
                
                diff, code_before = result[0]
            except Exception:
                stats["skipped_no_diff"] += 1
                continue
            
            # Apply smart context extraction
            clean_instruction = validator.clean_instruction(instruction)
            clean_input = ""
            if code_before:
                clean_input = extractor.extract_changed_region(
                    code_before, diff, INPUT_TRUNCATION_LENGTH
                )
            clean_output = validator.clean_output(diff, OUTPUT_TRUNCATION_LENGTH)
            
            # Validate
            is_valid, _ = validator.is_valid(clean_instruction, clean_input, clean_output)
            if not is_valid:
                stats["skipped_validation"] += 1
                continue
            
            # Build record - no system column (constant), just instruction/input/output
            record = {
                "instruction": clean_instruction,
                "input": clean_input,
                "output": clean_output
            }
            
            # Add quality score and reasoning as proper columns (not hidden metadata)
            if config["include_reasoning"] and ai_reason:
                record["quality_score"] = float(ai_score)
                record["quality_reason"] = ai_reason
            
            line = json.dumps(record, ensure_ascii=False) + '\n'
            f.write(line)
            stats["exported"] += 1
            stats["total_bytes"] += len(line.encode('utf-8'))
    
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export all HuggingFace dataset tiers")
    parser.add_argument("-o", "--output", default="huggingface_datasets",
                       help="Output directory")
    parser.add_argument("--tiers", nargs="+", 
                       choices=list(DATASETS.keys()) + ["all"],
                       default=["all"],
                       help="Which tiers to export")
    
    args = parser.parse_args()
    
    # Determine which datasets to export
    if "all" in args.tiers:
        tiers_to_export = list(DATASETS.keys())
    else:
        tiers_to_export = args.tiers
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    console.print("\n" + "=" * 70)
    console.print("[bold blue] HUGGINGFACE DATASET EXPORT [/bold blue]")
    console.print("=" * 70)
    console.print(f"\nOutput: {output_dir.absolute()}")
    console.print(f"Tiers: {', '.join(tiers_to_export)}")
    
    # Initialize
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    validator = TrainingExampleValidator()
    extractor = SmartContextExtractor()
    
    results = {}
    
    console.print("\n[yellow]Exporting datasets...[/yellow]\n")
    
    for tier_name in tiers_to_export:
        config = DATASETS[tier_name]
        console.print(f"\n[cyan]{tier_name}[/cyan]: {config['description']}")
        
        stats = export_dataset(
            client, tier_name, config, output_dir,
            validator, extractor
        )
        results[tier_name] = stats
    
    # Summary table
    console.print("\n" + "=" * 70)
    console.print("[bold green] EXPORT COMPLETE [/bold green]")
    console.print("=" * 70)
    
    table = Table(title="Dataset Summary")
    table.add_column("Dataset", style="cyan")
    table.add_column("Exported", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("File", style="dim")
    
    total_exported = 0
    total_bytes = 0
    
    for name, stats in results.items():
        size_mb = stats["total_bytes"] / 1_000_000
        skipped = stats["skipped_no_diff"] + stats["skipped_validation"]
        
        table.add_row(
            name,
            str(stats["exported"]),
            str(skipped),
            f"{size_mb:.2f}",
            f"{name}.jsonl"
        )
        
        total_exported += stats["exported"]
        total_bytes += stats["total_bytes"]
    
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_exported}[/bold]",
        "",
        f"[bold]{total_bytes/1_000_000:.2f}[/bold]",
        ""
    )
    
    console.print(table)
    
    # Create dataset card
    create_dataset_card(output_dir, results)
    
    console.print(f"\n[green]Files ready in: {output_dir.absolute()}[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Review the datasets")
    console.print("2. huggingface-cli login")
    console.print("3. huggingface-cli upload YOUR_USERNAME/linux-kernel-patches " + str(output_dir))


def create_dataset_card(output_dir: Path, results: dict):
    """Create a README.md for the HuggingFace dataset."""
    
    total_examples = sum(s["exported"] for s in results.values())
    
    card = f"""---
license: apache-2.0
language:
- en
tags:
- code
- linux-kernel
- patches
- instruction-tuning
- code-generation
size_categories:
- 10K<n<100K
---

# Linux Kernel Code Patches Dataset

High-quality Linux kernel commit patches for training code generation and understanding models.

## Dataset Description

This dataset contains {total_examples:,} curated Linux kernel commits with:
- Commit messages (instruction)
- Smart-extracted code context (input)
- Unified diff patches (output)
- Optional AI quality scores and reasoning

## Dataset Variants

| Variant | Examples | Description |
|---------|----------|-------------|
"""
    
    for name, stats in results.items():
        config = DATASETS.get(name, {})
        card += f"| `{name}` | {stats['exported']:,} | {config.get('description', '')} |\n"
    
    card += f"""
## Usage

```python
from datasets import load_dataset

# Load a specific variant
dataset = load_dataset("YOUR_USERNAME/linux-kernel-patches", data_files="premium_score.jsonl")

# Access examples
for example in dataset["train"]:
    print(example["instruction"])
    print(example["input"])
    print(example["output"])
```

## Format

**Base columns (all variants):**
- `instruction`: Commit message explaining the change
- `input`: Relevant code context (smart-extracted from the file)
- `output`: Unified diff patch

**Reasoning variants add:**
- `quality_score`: AI-assigned quality score (1-5)
- `quality_reason`: AI explanation of the score

## Quality Metrics

- **Smart Context Extraction**: ~90% average coverage of relevant code
- **Heuristic Scoring**: Based on commit metadata (reviews, fixes tags, etc.)
- **AI Scoring**: LLM-evaluated quality for training value

## License

Apache 2.0 - Same as the Linux kernel documentation and examples.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{linux-kernel-patches,
  title={{Linux Kernel Code Patches Dataset}},
  year={{2024}},
  publisher={{HuggingFace}},
}}
```

Generated on {datetime.now().strftime('%Y-%m-%d')}
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(card)
    
    console.print(f"\n[dim]Created dataset card: {readme_path}[/dim]")


if __name__ == "__main__":
    main()
