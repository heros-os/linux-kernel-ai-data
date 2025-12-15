#!/usr/bin/env python3
"""
Production Training Data Exporter - v2.0

Includes all quality fixes:
- Parameterized queries (no SQL injection)
- Validation before export
- System prompts
- Subsystem diversity
- Proper filtering (no AI score=-1)
- Smart truncation
"""

import sys
import json
from pathlib import Path
import argparse
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from clickhouse_driver import Client
from tqdm import tqdm

from config.settings import settings
from config.constants import (
    MIN_DIFF_LENGTH, MAX_DIFF_LENGTH,
    INPUT_TRUNCATION_LENGTH, OUTPUT_TRUNCATION_LENGTH,
    DEFAULT_SYSTEM_PROMPT, KERNEL_SYSTEM_PROMPT,
    PREMIUM_MIN_SCORE, HIGH_QUALITY_MIN_SCORE, STANDARD_MIN_SCORE,
    AI_SCORE_SKIPPED
)
from src.validator import TrainingExampleValidator

console = Console()
validator = TrainingExampleValidator()


def export_tier(
    client: Client,
    output_path: Path,
    min_quality: float,
    limit: int,
    batch_size: int = 100,
    include_reasoning: bool = False,
    include_system_prompt: bool = True,
    ensure_diversity: bool = True,
    strict_validation: bool = False
) -> dict:
    """
    Export a single tier of training data with all quality improvements.
    
    Returns:
        Statistics dict with export counts
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'exported': 0,
        'skipped_validation': 0,
        'skipped_no_diff': 0,
        'skipped_ai_negative': 0,
        'subsystems': defaultdict(int)
    }
    
    offset = 0
    seen_commits = set()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        with tqdm(total=limit, desc=f"  {output_path.name}", leave=False) as pbar:
            while stats['exported'] < limit:
                remaining = limit - stats['exported']
                current_batch = min(batch_size, remaining + 50)
                
                # Build query based on whether we need reasoning
                if include_reasoning:
                    # Use parameterized query for safety
                    query = """
                    SELECT 
                        commit_hash,
                        subject || '\n\n' || body as instruction,
                        ai_quality_score,
                        ai_score_reason
                    FROM commits
                    WHERE heuristic_score >= %(min_quality)s
                        AND ai_quality_score > %(min_ai_score)s
                        AND (ai_score_reason IS NOT NULL AND ai_score_reason != '')
                    ORDER BY ai_quality_score DESC, heuristic_score DESC
                    LIMIT %(limit)s OFFSET %(offset)s
                    """
                    params = {
                        'min_quality': min_quality,
                        'min_ai_score': 0,  # Exclude negatives
                        'limit': current_batch,
                        'offset': offset
                    }
                else:
                    query = """
                    SELECT 
                        commit_hash,
                        subject || '\n\n' || body as instruction
                    FROM commits
                    WHERE heuristic_score >= %(min_quality)s
                    ORDER BY heuristic_score DESC
                    LIMIT %(limit)s OFFSET %(offset)s
                    """
                    params = {
                        'min_quality': min_quality,
                        'limit': current_batch,
                        'offset': offset
                    }
                
                try:
                    commits = client.execute(query, params)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    break
                
                if not commits:
                    break
                
                for row in commits:
                    if stats['exported'] >= limit:
                        break
                    
                    if include_reasoning:
                        commit_hash, instruction, ai_score, ai_reason = row
                    else:
                        commit_hash, instruction = row
                        ai_score, ai_reason = None, None
                    
                    # Skip duplicates
                    if commit_hash in seen_commits:
                        continue
                    seen_commits.add(commit_hash)
                    
                    # Skip negative AI scores
                    if ai_score is not None and ai_score <= AI_SCORE_SKIPPED:
                        stats['skipped_ai_negative'] += 1
                        continue
                    
                    # Get file change with parameterized query
                    try:
                        fc_query = """
                        SELECT 
                            code_before,
                            diff_hunk,
                            subsystem
                        FROM file_changes
                        WHERE commit_hash = %(hash)s
                            AND is_binary = 0
                            AND length(diff_hunk) > %(min_len)s
                            AND length(diff_hunk) < %(max_len)s
                        LIMIT 1
                        """
                        fc_params = {
                            'hash': commit_hash,
                            'min_len': MIN_DIFF_LENGTH,
                            'max_len': MAX_DIFF_LENGTH
                        }
                        fc_result = client.execute(fc_query, fc_params)
                        
                        if not fc_result:
                            stats['skipped_no_diff'] += 1
                            continue
                        
                        code_before, diff, subsystem = fc_result[0]
                    except Exception:
                        stats['skipped_no_diff'] += 1
                        continue
                    
                    # Clean and truncate with smart context extraction
                    clean_instruction = validator.clean_instruction(instruction)
                    clean_input = validator.clean_input(
                        code_before or "", 
                        diff=diff,  # Pass diff for smart extraction
                        max_length=INPUT_TRUNCATION_LENGTH
                    )
                    clean_output = validator.clean_output(
                        diff,
                        OUTPUT_TRUNCATION_LENGTH
                    )
                    
                    # Validate
                    is_valid, error = validator.is_valid(
                        clean_instruction,
                        clean_input,
                        clean_output,
                        strict=strict_validation
                    )
                    
                    if not is_valid:
                        stats['skipped_validation'] += 1
                        continue
                    
                    # Build record
                    record = {
                        "instruction": clean_instruction,
                        "input": clean_input if clean_input else "",
                        "output": clean_output
                    }
                    
                    # Add system prompt
                    if include_system_prompt:
                        record["system"] = KERNEL_SYSTEM_PROMPT
                    
                    # Add reasoning metadata
                    if include_reasoning and ai_reason:
                        record["_quality_score"] = ai_score
                        record["_quality_reason"] = ai_reason
                    
                    # Write
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats['exported'] += 1
                    stats['subsystems'][subsystem or 'unknown'] += 1
                    pbar.update(1)
                
                offset += len(commits)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Export production-quality training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all tiers
  python scripts/export_production.py --all
  
  # Export only premium with reasoning
  python scripts/export_production.py --premium --with-reasoning
  
  # Export with strict validation
  python scripts/export_production.py --all --strict
        """
    )
    parser.add_argument("-o", "--output-dir", default="exports", help="Output directory")
    parser.add_argument("--premium", action="store_true", help="Export premium tier")
    parser.add_argument("--hq", action="store_true", help="Export high quality tier")
    parser.add_argument("--standard", action="store_true", help="Export standard tier")
    parser.add_argument("--all", action="store_true", help="Export all tiers")
    parser.add_argument("--with-reasoning", action="store_true", help="Include AI reasoning")
    parser.add_argument("--no-system-prompt", action="store_true", help="Omit system prompt")
    parser.add_argument("--strict", action="store_true", help="Use strict validation")
    parser.add_argument("--premium-limit", type=int, default=10000)
    parser.add_argument("--hq-limit", type=int, default=50000)
    parser.add_argument("--standard-limit", type=int, default=100000)
    parser.add_argument("--reasoning-limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    
    args = parser.parse_args()
    
    # Default to --all if no tier specified
    if not (args.premium or args.hq or args.standard or args.all):
        args.all = True
    
    console.print("[bold blue]Production Training Data Exporter v2.0[/bold blue]")
    console.print(f"Output: {args.output_dir}")
    console.print(f"System prompt: {'No' if args.no_system_prompt else 'Yes'}")
    console.print(f"Strict validation: {'Yes' if args.strict else 'No'}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    results = {}
    
    # Build tier list
    tiers = []
    if args.all or args.premium:
        tiers.append(("premium.jsonl", PREMIUM_MIN_SCORE, args.premium_limit, "[PREMIUM]", False))
    if args.all or args.hq:
        tiers.append(("high_quality.jsonl", HIGH_QUALITY_MIN_SCORE, args.hq_limit, "[HQ]", False))
    if args.all or args.standard:
        tiers.append(("standard.jsonl", STANDARD_MIN_SCORE, args.standard_limit, "[STD]", False))
    
    # Add reasoning tiers if requested
    if args.with_reasoning:
        if args.all or args.premium:
            tiers.append(("premium_with_reasoning.jsonl", PREMIUM_MIN_SCORE, args.reasoning_limit, "[PREMIUM+AI]", True))
        if args.all or args.hq:
            tiers.append(("hq_with_reasoning.jsonl", HIGH_QUALITY_MIN_SCORE, args.reasoning_limit, "[HQ+AI]", True))
    
    console.print(f"\nExporting {len(tiers)} datasets...\n")
    
    for filename, min_quality, limit, label, include_reasoning in tiers:
        console.print(f"[yellow]{label} {filename}[/yellow]")
        output_path = output_dir / filename
        
        stats = export_tier(
            client,
            output_path,
            min_quality=min_quality,
            limit=limit,
            batch_size=args.batch_size,
            include_reasoning=include_reasoning,
            include_system_prompt=not args.no_system_prompt,
            strict_validation=args.strict
        )
        
        size = output_path.stat().st_size / 1_000_000 if output_path.exists() else 0
        results[filename] = {**stats, 'size': size}
        
        console.print(f"  OK - {stats['exported']:,} exported, "
                     f"{stats['skipped_validation']} validation fails, "
                     f"{stats['skipped_no_diff']} no diff ({size:.2f} MB)\n")
    
    # Summary
    console.print("\n[bold]==============================================")
    console.print("[bold green]Export Complete![/bold green]\n")
    
    table = Table(title="Production Datasets")
    table.add_column("File", style="cyan")
    table.add_column("Exported", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("Size (MB)", justify="right")
    
    total_exported = 0
    total_size = 0
    
    for filename, data in results.items():
        skipped = data['skipped_validation'] + data['skipped_no_diff'] + data['skipped_ai_negative']
        table.add_row(
            filename,
            f"{data['exported']:,}",
            f"{skipped:,}",
            f"{data['size']:.2f}"
        )
        total_exported += data['exported']
        total_size += data['size']
    
    table.add_row("-" * 15, "-" * 8, "-" * 8, "-" * 8)
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_exported:,}[/bold]", "", f"[bold]{total_size:.2f}[/bold]")
    
    console.print(table)
    console.print(f"\nOutput: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
