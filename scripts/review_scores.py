#!/usr/bin/env python3
"""
Review AI-scored commits to evaluate scoring quality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from clickhouse_driver import Client

from config.settings import settings

console = Console()

def main():
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    # Get AI-scored commits
    query = """
    SELECT 
        commit_hash,
        subject,
        heuristic_score,
        ai_quality_score
    FROM commits
    WHERE ai_quality_score > 0
    ORDER BY ai_quality_score DESC
    LIMIT 20
    """
    
    rows = client.execute(query)
    
    if not rows:
        console.print("[yellow]No AI-scored commits found.[/yellow]")
        return
    
    console.print(f"[bold cyan]Found {len(rows)} AI-scored commits[/bold cyan]\n")
    
    # Score distribution
    scores = [r[3] for r in rows]
    console.print(f"[bold]Score Distribution:[/bold]")
    console.print(f"  Min: {min(scores):.1f}, Max: {max(scores):.1f}, Avg: {sum(scores)/len(scores):.1f}")
    
    # Table
    table = Table(title="Top AI-Scored Commits")
    table.add_column("Hash", style="dim", width=12)
    table.add_column("Subject", width=50)
    table.add_column("Heuristic", justify="right")
    table.add_column("AI Score", justify="right")
    table.add_column("Match?", justify="center")
    
    for commit_hash, subject, heuristic, ai_score in rows:
        # Normalize heuristic (0-100) to 1-5 scale for comparison
        heuristic_normalized = (heuristic / 100) * 5
        diff = abs(ai_score - heuristic_normalized)
        
        match = "✅" if diff < 1.0 else "⚠️" if diff < 2.0 else "❌"
        
        table.add_row(
            commit_hash[:12],
            subject[:50],
            f"{heuristic:.0f}",
            f"{ai_score:.1f}",
            match
        )
    
    console.print(table)
    
    # Get a few samples with full details
    console.print("\n[bold]Sample Commit Details:[/bold]")
    for i, (commit_hash, subject, heuristic, ai_score) in enumerate(rows[:3]):
        console.print(f"\n[cyan]#{i+1}[/cyan] {commit_hash[:12]}")
        console.print(f"  Subject: {subject[:80]}")
        console.print(f"  Heuristic: {heuristic:.0f}/100 → AI: {ai_score:.1f}/5")
        
        # Get diff sample
        diff_query = f"""
        SELECT diff_hunk FROM file_changes 
        WHERE commit_hash = '{commit_hash}' 
        AND length(diff_hunk) > 50 LIMIT 1
        """
        diff_result = client.execute(diff_query)
        if diff_result:
            diff = diff_result[0][0][:300]
            console.print(f"  Diff preview: {diff[:200]}...")

if __name__ == "__main__":
    main()
