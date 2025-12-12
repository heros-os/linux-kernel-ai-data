#!/usr/bin/env python3
"""
Batch AI scoring script.

Reads commits from ClickHouse that have high heuristic scores but no AI score,
and sends them to a local LLM for evaluation.
"""

import sys
from pathlib import Path
import argparse
import logging
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from clickhouse_driver import Client

from config.settings import settings
from src.quality_scorer import QualityScorer

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run AI quality scoring on kernel commits")
    parser.add_argument("--limit", type=int, default=1000, help="Max commits to score")
    parser.add_argument("--min-heuristic", type=float, default=None, help="Min heuristic score threshold")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    
    args = parser.parse_args()
    
    # Defaults from settings
    min_heuristic = args.min_heuristic if args.min_heuristic is not None else settings.quality.min_heuristic_score
    batch_size = args.batch_size or settings.quality.batch_size
    
    console.print(f"[bold blue]Linux Kernel AI Quality Scorer[/bold blue]")
    console.print(f"Model: [cyan]{settings.quality.model_name}[/cyan] via {settings.quality.ollama_url}")
    console.print(f"Thresholds: Heuristic > {min_heuristic}")
    
    scorer = QualityScorer()
    
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    # 1. Fetch candidates
    # We join with file_changes to get the diff, picking the first/largest C file change
    query = f"""
    SELECT 
        c.commit_hash,
        c.subject || '\n\n' || c.body as instruction,
        fc.diff_hunk
    FROM commits c
    INNER JOIN file_changes fc ON c.commit_hash = fc.commit_hash
    WHERE 
        c.heuristic_score >= {min_heuristic}
        AND c.ai_quality_score IS NULL
        AND fc.file_extension IN ('.c', '.h')
        AND length(fc.diff_hunk) < 10000
    ORDER BY c.heuristic_score DESC
    LIMIT {args.limit}
    """
    
    console.print("\n[yellow]Fetching candidates from database...[/yellow]")
    candidates = client.execute(query)
    
    if not candidates:
        console.print("[yellow]No candidates found matching criteria.[/yellow]")
        return

    console.print(f"[green]Found {len(candidates)} commits to score.[/green]")
    
    # 2. Score loop
    updates = []
    
    with Progress(
        SpinnerColumn(),
        TextMessageColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Scoring commits...", total=len(candidates))
        
        for commit_hash, instruction, diff in candidates:
            score = scorer.get_ai_score(instruction, diff)
            
            if score is not None:
                updates.append((score, commit_hash))
            else:
                # Mark as processed but score 0 (failed/refused) so we don't loop forever
                updates.append((0.0, commit_hash))
            
            progress.advance(task)
            
            # Batch update
            if len(updates) >= batch_size:
                _update_db(client, updates)
                updates = []
                
        # Final flush
        if updates:
            _update_db(client, updates)

    console.print("\n[bold green]Scoring complete![/bold green]")

def _update_db(client: Client, updates: List[Tuple[float, str]]):
    """Update ClickHouse with scores."""
    client.execute(
        "ALTER TABLE commits UPDATE ai_quality_score = %(score)f WHERE commit_hash = %(hash)s",
        [{'score': s, 'hash': h} for s, h in updates]
    )

class TextMessageColumn(TextColumn):
    """Renders text."""

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
