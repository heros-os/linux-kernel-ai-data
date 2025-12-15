#!/usr/bin/env python3
"""
Calculate heuristic scores for commits that don't have them.
Runs retroactively after extraction to populate quality scores.
"""

import sys
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from clickhouse_driver import Client

from config.settings import settings
from src.quality_scorer import QualityScorer

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Calculate heuristic scores for commits")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Max commits to process")
    
    args = parser.parse_args()
    
    console.print("[bold blue]Heuristic Score Calculator[/bold blue]")
    
    scorer = QualityScorer()
    
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    # Count commits to process
    total = client.execute("SELECT count() FROM commits WHERE heuristic_score = 0")[0][0]
    console.print(f"[yellow]Found {total:,} commits with score = 0[/yellow]")
    
    if args.limit:
        total = min(total, args.limit)
        console.print(f"[yellow]Processing first {total:,}[/yellow]")
    
    processed = 0
    offset = 0
    
    while processed < total:
        # Fetch batch
        query = f"""
        SELECT 
            commit_hash, subject, body, files_changed,
            signed_off_by, reviewed_by, acked_by, fixes_hash
        FROM commits
        WHERE heuristic_score = 0
        ORDER BY commit_hash
        LIMIT {args.batch_size} OFFSET {offset}
        """
        
        try:
            rows = client.execute(query)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            break
        
        if not rows:
            break
        
        console.print(f"[cyan]Processing batch at offset {offset}...[/cyan]")
        
        for row in rows:
            commit_hash, subject, body, files_changed, signed_off, reviewed, acked, fixes = row
            
            # Calculate score
            score = scorer.calculate_heuristic_score(
                subject=subject,
                body=body,
                files_changed=files_changed,
                signed_off_by=signed_off if signed_off else [],
                reviewed_by=reviewed if reviewed else [],
                acked_by=acked if acked else [],
                fixes_hash=fixes if fixes else None
            )
            
            # Update using parameterized query
            try:
                update_query = """
                ALTER TABLE commits UPDATE heuristic_score = %(score)s 
                WHERE commit_hash = %(hash)s
                """
                client.execute(update_query, {'score': score, 'hash': commit_hash})
            except Exception as e:
                logger.warning(f"Failed to update {commit_hash}: {e}")
            
            processed += 1
            
            if processed % 100 == 0:
                console.print(f"  Processed {processed:,}/{total:,}")
        
        offset += args.batch_size
    
    console.print(f"\n[bold green]Complete! Scored {processed:,} commits.[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
