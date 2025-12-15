#!/usr/bin/env python3
"""
Training data export script.

Exports data from ClickHouse to JSONL format for LLM fine-tuning.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.exporter import TrainingDataExporter

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export Linux kernel training data to JSONL"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="training_data.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--mode",
        choices=["diff", "full"],
        default="diff",
        help="Output mode: 'diff' for patches, 'full' for complete code"
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum lines changed to include"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500,
        help="Maximum lines changed to include"
    )
    parser.add_argument(
        "--subsystems",
        type=str,
        nargs="+",
        default=None,
        help="Subsystems to include (e.g., mm kernel/sched)"
    )
    parser.add_argument(
        "--exclude-subsystems",
        type=str,
        nargs="+",
        default=None,
        help="Subsystems to exclude"
    )
    parser.add_argument(
        "--include-merges",
        action="store_true",
        help="Include merge commits"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum examples to export"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Random sampling ratio (0.0-1.0)"
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include commit hash and subsystem in output"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only, don't export"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum heuristic quality score (0-100)"
    )
    parser.add_argument(
        "--min-ai-score",
        type=float,
        default=0.0,
        help="Minimum AI quality score (1-5)"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Include ONLY commits with these tags (e.g. security performance)"
    )
    parser.add_argument(
        "--exclude-tags",
        type=str,
        nargs="+",
        default=None,
        help="Exclude commits with these tags"
    )
    
    args = parser.parse_args()
    
    console.print("[bold blue]Linux Kernel Training Data Exporter[/bold blue]\n")
    
    exporter = TrainingDataExporter()
    
    try:
        if args.stats:
            # Show statistics
            console.print("[yellow]Fetching statistics...[/yellow]\n")
            stats = exporter.get_statistics()
            
            console.print(f"[green]Total commits (non-merge):[/green] {stats['total_commits']:,}")
            console.print(f"[green]Total file changes:[/green] {stats['total_file_changes']:,}\n")
            
            # Subsystem table
            table = Table(title="Top Subsystems")
            table.add_column("Subsystem", style="cyan")
            table.add_column("File Changes", justify="right")
            
            for subsys, count in stats['top_subsystems'].items():
                table.add_row(subsys, f"{count:,}")
            
            console.print(table)
            
            # Lines changed distribution
            console.print("\n[bold]Lines Changed Distribution:[/bold]")
            for bucket, count in stats['lines_changed_distribution'].items():
                console.print(f"  {bucket}: {count:,}")
            
            return
        
        # Export data
        output_path = Path(args.output)
        
        console.print(f"[yellow]Exporting to {output_path}...[/yellow]")
        console.print(f"[dim]Mode: {args.mode}, Lines: {args.min_lines}-{args.max_lines}[/dim]")
        
        if args.subsystems:
            console.print(f"[dim]Subsystems: {', '.join(args.subsystems)}[/dim]")
        
        count = exporter.export_jsonl(
            output_path=output_path,
            output_mode=args.mode,
            min_lines=args.min_lines,
            max_lines=args.max_lines,
            exclude_merges=not args.include_merges,
            subsystems=args.subsystems,
            exclude_subsystems=args.exclude_subsystems,
            limit=args.limit,
            sample_ratio=args.sample,
            include_metadata=args.include_metadata,
            min_heuristic_score=args.min_quality,
            min_ai_score=args.min_ai_score,
            tags=args.tags,
            exclude_tags=args.exclude_tags
        )
        
        console.print(f"\n[bold green]Export complete![/bold green]")
        console.print(f"  Examples: {count:,}")
        console.print(f"  Output: {output_path}")
        
        # Show file size
        file_size = output_path.stat().st_size
        if file_size > 1_000_000_000:
            size_str = f"{file_size / 1_000_000_000:.2f} GB"
        elif file_size > 1_000_000:
            size_str = f"{file_size / 1_000_000:.2f} MB"
        else:
            size_str = f"{file_size / 1000:.2f} KB"
        console.print(f"  Size: {size_str}")
        
    finally:
        exporter.close()


if __name__ == "__main__":
    main()
