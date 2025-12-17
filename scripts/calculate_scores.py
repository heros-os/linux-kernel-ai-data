#!/usr/bin/env python3
"""
Calculate heuristic scores for commits that don't have them.
Runs retroactively after extraction to populate quality scores.
"""

import sys
import multiprocessing
import math
from pathlib import Path
import argparse
import logging
from typing import List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from clickhouse_driver import Client

from config.settings import settings
from src.quality_scorer import QualityScorer

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scorer for worker processes
scorer = None

def init_worker():
    """Initialize global scorer in worker process."""
    global scorer
    scorer = QualityScorer()

def calculate_score_wrapper(args):
    """Wrapper for parallel execution."""
    commit_hash, subject, body, files_changed, signed_off, reviewed, acked, fixes = args
    if scorer is None:
        init_worker()
    
    score = scorer.calculate_heuristic_score(
        subject=subject,
        body=body,
        files_changed=files_changed,
        signed_off_by=signed_off if signed_off else [],
        reviewed_by=reviewed if reviewed else [],
        acked_by=acked if acked else [],
        fixes_hash=fixes if fixes else None
    )
    return commit_hash, score

def main():
    parser = argparse.ArgumentParser(description="Calculate heuristic scores for commits")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for fetching")
    parser.add_argument("--update-chunk", type=int, default=1000, help="Chunk size for DB updates")
    parser.add_argument("--limit", type=int, default=None, help="Max commits to process")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of worker processes")
    
    args = parser.parse_args()
    
    console.print("[bold blue]Heuristic Score Calculator (Optimized)[/bold blue]")
    console.print(f"Workers: {args.workers}")
    console.print(f"Batch Size: {args.batch_size}")
    
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    # Count commits to process
    console.print("Counting commits to process...")
    total_found = client.execute("SELECT count() FROM commits WHERE heuristic_score = 0")[0][0]
    console.print(f"[yellow]Found {total_found:,} commits with score = 0[/yellow]")
    
    total_to_process = total_found
    if args.limit:
        total_to_process = min(total_found, args.limit)
        console.print(f"[yellow]Processing limit set to {total_to_process:,}[/yellow]")
    
    if total_to_process == 0:
        return

    processed_count = 0
    last_commit_hash = ''
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Scoring commits...", total=total_to_process)
        
        pool = multiprocessing.Pool(processes=args.workers, initializer=init_worker)
        
        try:
            while processed_count < total_to_process:
                # Calculate current batch size
                current_batch_limit = min(args.batch_size, total_to_process - processed_count)
                
                # Fetch batch using keyset pagination (stable against updates)
                query = f"""
                SELECT 
                    commit_hash, subject, body, files_changed,
                    signed_off_by, reviewed_by, acked_by, fixes_hash
                FROM commits
                WHERE heuristic_score = 0
                AND commit_hash > '{last_commit_hash}'
                ORDER BY commit_hash
                LIMIT {current_batch_limit}
                """
                
                try:
                    rows = client.execute(query)
                except Exception as e:
                    console.print(f"[red]Error fetching batch: {e}[/red]")
                    break
                
                if not rows:
                    break
                
                # Update last_commit_hash for next iteration
                last_commit_hash = rows[-1][0]
                
                # Parallelize scoring
                # rows is list of tuples, matches args for wrapper
                results = pool.map(calculate_score_wrapper, rows)
                
                # Batch update database
                # Split results into chunks for update
                updates_made = 0
                for i in range(0, len(results), args.update_chunk):
                    chunk = results[i:i + args.update_chunk]
                    
                    hashes = [r[0] for r in chunk]
                    scores = [r[1] for r in chunk]
                    
                    try:
                        update_query = """
                        ALTER TABLE commits UPDATE heuristic_score = transform(commit_hash, %(hashes)s, %(scores)s, heuristic_score)
                        WHERE commit_hash IN %(hashes)s
                        """
                        client.execute(update_query, {'hashes': hashes, 'scores': scores})
                        updates_made += len(chunk)
                    except Exception as e:
                        logger.error(f"Update failed for chunk starting with {hashes[0]}: {e}")
                
                processed_count += len(rows)
                progress.update(task, advance=len(rows))
                
                # No offset increment needed as we use keyset > last_hash
                
        finally:
            pool.close()
            pool.join()
            
    console.print(f"\n[bold green]Complete! Scored {processed_count:,} commits.[/bold green]")
    console.print("[dim]Note: ClickHouse updates are asynchronous and may take a moment to reflect.[/dim]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
