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
from clickhouse_driver import Client

from config.settings import settings
from config.constants import (
    MIN_DIFF_LENGTH, MAX_DIFF_LENGTH,
    AI_SCORE_SKIPPED, KERNEL_EXTENSIONS
)
from src.quality_scorer import QualityScorer

console = Console()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run AI quality scoring on kernel commits")
    parser.add_argument("--limit", type=int, default=1000, help="Max commits to score")
    parser.add_argument("--min-heuristic", type=float, default=None, help="Min heuristic score threshold")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for fetching")
    
    args = parser.parse_args()
    
    # Defaults from settings
    min_heuristic = args.min_heuristic if args.min_heuristic is not None else settings.quality.min_heuristic_score
    batch_size = args.batch_size
    
    console.print(f"[bold blue]Linux Kernel AI Quality Scorer[/bold blue]")
    console.print(f"Model: [cyan]{settings.quality.model_name}[/cyan] via {settings.quality.ollama_url}")
    console.print(f"Thresholds: Heuristic > {min_heuristic}")
    console.print(f"Batch size: {batch_size}")
    
    scorer = QualityScorer()
    
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    total_scored = 0
    total_skipped = 0
    offset = 0
    consecutive_empty = 0
    
    console.print("\n[yellow]Starting batch scoring...[/yellow]")
    
    while total_scored < args.limit:
        # Fetch batch with LIMIT and OFFSET to avoid re-processing
        remaining = args.limit - total_scored
        current_batch = min(batch_size, remaining + 50)  # Fetch extra in case some fail
        
        # Get commits that haven't been scored yet (no reason = not processed)
        commit_query = f"""
        SELECT 
            commit_hash,
            subject || '\n\n' || body as instruction,
            ai_quality_score,
            ai_score_reason
        FROM commits
        WHERE 
            heuristic_score >= {min_heuristic}
            AND (ai_score_reason IS NULL OR ai_score_reason = '')
        ORDER BY heuristic_score DESC
        LIMIT {current_batch} OFFSET {offset}
        """
        
        try:
            commits = client.execute(commit_query)
        except Exception as e:
            console.print(f"[red]Database error: {e}[/red]")
            break
        
        if not commits:
            console.print("[yellow]No more candidates found.[/yellow]")
            break
        
        console.print(f"[dim]Fetched {len(commits)} commits (offset={offset})...[/dim]")
        
        batch_scored = 0
        batch_skipped = 0
        updates = []
        
        for commit_hash, instruction, existing_score, existing_reason in commits:
            if total_scored >= args.limit:
                break
            
            # Skip if already has a reason (already properly evaluated)
            if existing_reason and len(existing_reason) > 5:
                batch_skipped += 1
                continue
                
            # Get diff AND code context (parameterized query)
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
                diff_params = {
                    'hash': commit_hash,
                    'min_len': MIN_DIFF_LENGTH,
                    'max_len': MAX_DIFF_LENGTH
                }
                diff_result = client.execute(diff_query, diff_params)
                
                if not diff_result:
                    # Mark as skipped (score = -1) so we don't process again
                    updates.append((-1.0, "no valid diff", commit_hash))
                    batch_skipped += 1
                    total_skipped += 1
                    continue
                    
                diff, code_before = diff_result[0]
            except Exception:
                batch_skipped += 1
                total_skipped += 1
                continue
            
            # Extract smart context
            code_context = ""
            if code_before:
                from src.context_extractor import SmartContextExtractor
                from config.constants import INPUT_TRUNCATION_LENGTH
                extractor = SmartContextExtractor()
                code_context = extractor.extract_changed_region(
                    code_before, diff, max_chars=INPUT_TRUNCATION_LENGTH
                )
            
            # Get AI score with context
            score, reason = scorer.get_ai_score(instruction, diff, code_context)
            
            if score is not None:
                updates.append((score, reason or "", commit_hash))
                batch_scored += 1
                total_scored += 1
            else:
                updates.append((0.0, "scoring failed", commit_hash))
                batch_skipped += 1
                total_skipped += 1
        
        # Update database
        if updates:
            _update_db(client, updates)
        
        # Progress
        console.print(f"  [green]✓ Scored: {batch_scored}[/green], [yellow]Skipped: {batch_skipped}[/yellow] | Total: {total_scored}/{args.limit}")
        
        # Move offset forward
        offset += len(commits)
        
        # Safety: stop if we've gone through too many without scoring any
        if batch_scored == 0:
            consecutive_empty += 1
            if consecutive_empty >= 5:
                console.print("[yellow]5 consecutive batches with no valid commits. Stopping.[/yellow]")
                break
        else:
            consecutive_empty = 0
    
    console.print(f"\n[bold green]Scoring complete![/bold green]")
    console.print(f"  Scored: {total_scored}")
    console.print(f"  Skipped: {total_skipped}")

def _update_db(client: Client, updates: List[Tuple[float, str, str]]):
    """Update ClickHouse with scores and reasoning."""
    for score, reason, commit_hash in updates:
        try:
            safe_reason = reason.replace("'", "''")[:500] if reason else ""
            client.execute(
                f"ALTER TABLE commits UPDATE ai_quality_score = {score}, ai_score_reason = '{safe_reason}' WHERE commit_hash = '{commit_hash}'"
            )
        except Exception as e:
            pass  # Silently skip failed updates

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
