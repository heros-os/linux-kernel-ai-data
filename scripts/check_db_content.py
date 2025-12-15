#!/usr/bin/env python3
"""
Check what's actually in the database for code_before
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings
from rich.console import Console
from rich.panel import Panel

console = Console()

client = Client(
    host=settings.clickhouse.host,
    port=settings.clickhouse.port,
    database=settings.clickhouse.database
)

# Get a specific commit that we know has issues
query = """
SELECT 
    c.commit_hash,
    c.subject,
    fc.file_path,
    fc.diff_hunk,
    length(fc.code_before) as code_before_len,
    substring(fc.code_before, 1, 500) as code_before_start,
    substring(fc.code_before, length(fc.code_before) - 500, 500) as code_before_end
FROM commits c
JOIN file_changes fc ON c.commit_hash = fc.commit_hash
WHERE c.heuristic_score >= 8.0
    AND fc.is_binary = 0
    AND length(fc.diff_hunk) > 100
    AND length(fc.diff_hunk) < 5000
LIMIT 5
"""

results = client.execute(query)

for i, row in enumerate(results, 1):
    commit_hash, subject, file_path, diff, code_len, code_start, code_end = row
    
    console.print(f"\n[bold cyan]Entry #{i}[/bold cyan]")
    console.print(f"Commit: {commit_hash[:12]}")
    console.print(f"Subject: {subject[:80]}")
    console.print(f"File: {file_path}")
    console.print(f"Code before length: {code_len} chars\n")
    
    console.print(Panel(
        code_start,
        title="[blue]First 500 chars of code_before[/blue]",
        border_style="blue"
    ))
    
    console.print(Panel(
        diff[:800],
        title="[green]Diff (first 800 chars)[/green]",
        border_style="green"
    ))
