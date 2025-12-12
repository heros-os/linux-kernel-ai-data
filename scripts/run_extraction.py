#!/usr/bin/env python3
"""
Main extraction runner script.

Orchestrates the full extraction pipeline:
1. Clone/validate kernel repository
2. Initialize database
3. Run parallel extraction
4. Write to ClickHouse
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings
from src.repository import RepositoryManager
from src.pipeline import ExtractionPipeline
from src.writer import ClickHouseWriter, init_database

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Linux kernel commit history for LLM training"
    )
    parser.add_argument(
        "--clone",
        action="store_true",
        help="Clone the kernel repository (first run)"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the ClickHouse database"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = auto)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Commits per batch"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of commits to process"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Commit hash to resume from"
    )
    parser.add_argument(
        "--single-threaded",
        action="store_true",
        help="Run in single-threaded mode (for debugging)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to database, just test extraction"
    )
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold blue]Linux Kernel Chronological Intelligence Engine[/bold blue]\n"
        "[dim]Extracting commit history for LLM training[/dim]",
        border_style="blue"
    ))
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Repository management
    repo_manager = RepositoryManager()
    
    if args.clone:
        console.print("\n[yellow]Cloning kernel repository...[/yellow]")
        console.print("[dim]This may take 30+ minutes on first run[/dim]")
        repo_manager.clone_kernel()
    
    # Verify repository exists
    if not repo_manager.kernel_path.exists():
        console.print(
            "[red]Kernel repository not found![/red]\n"
            "Run with --clone to download it first."
        )
        sys.exit(1)
    
    # Database initialization
    if args.init_db:
        console.print("\n[yellow]Initializing database...[/yellow]")
        try:
            init_database()
            console.print("[green]Database initialized![/green]")
        except Exception as e:
            console.print(f"[red]Database init failed: {e}[/red]")
            console.print("[dim]Make sure ClickHouse is running (docker-compose up -d)[/dim]")
            sys.exit(1)
    
    # Get commit list
    console.print("\n[yellow]Loading commit history...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Walking commit graph...", total=None)
        commit_hashes = repo_manager.get_all_commit_hashes()
        progress.update(task, completed=True)
    
    total_commits = len(commit_hashes)
    console.print(f"[green]Found {total_commits:,} commits[/green]")
    
    if args.limit:
        commit_hashes = commit_hashes[:args.limit]
        console.print(f"[yellow]Limited to {args.limit:,} commits[/yellow]")
    
    # Setup writer
    writer = None if args.dry_run else ClickHouseWriter()
    
    def on_batch_complete(result):
        if writer:
            writer.write_batch(result)
    
    # Run extraction
    console.print(f"\n[yellow]Starting extraction...[/yellow]")
    console.print(f"[dim]Workers: {args.workers or 'auto'}, Batch size: {args.batch_size}[/dim]")
    
    pipeline = ExtractionPipeline(
        repo_path=str(repo_manager.kernel_path),
        num_workers=args.workers or None,
        batch_size=args.batch_size,
        on_batch_complete=on_batch_complete if not args.single_threaded else None
    )
    
    try:
        if args.single_threaded:
            commits, files, errors = pipeline.run_single_threaded(
                commit_hashes,
                on_commit=lambda c, f: writer.write_batch(
                    type('Result', (), {'commits': [c], 'file_changes': f, 'errors': []})()
                ) if writer else None
            )
        else:
            commits, files, errors = pipeline.run(
                commit_hashes,
                resume_from=args.resume
            )
        
        # Final flush
        if writer:
            writer.flush()
            stats = writer.get_stats()
            console.print(f"\n[green]Database writes:[/green]")
            console.print(f"  Commits: {stats['commits_written']:,}")
            console.print(f"  File changes: {stats['file_changes_written']:,}")
            console.print(f"  Batches: {stats['batches_written']:,}")
        
        console.print(f"\n[bold green]Extraction complete![/bold green]")
        console.print(f"  Commits processed: {commits:,}")
        console.print(f"  File changes: {files:,}")
        console.print(f"  Errors: {errors:,}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted! Flushing data...[/yellow]")
        if writer:
            writer.flush()
    finally:
        if writer:
            writer.close()
        repo_manager.close()


if __name__ == "__main__":
    main()
