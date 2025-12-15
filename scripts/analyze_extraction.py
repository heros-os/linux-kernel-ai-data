#!/usr/bin/env python3
"""
Analyze a specific exported entry to understand the smart extraction issue
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings
from rich.console import Console
from rich.panel import Panel
from src.context_extractor import SmartContextExtractor

console = Console()

# Read first entry from exported file
file_path = Path("huggingface_dataset/premium_train.jsonl")

with open(file_path, 'r', encoding='utf-8') as f:
    entry = json.loads(f.readline())

commit_hash = None

# Try to find the commit hash by querying with the instruction
client = Client(
    host=settings.clickhouse.host,
    port=settings.clickhouse.port,
    database=settings.clickhouse.database
)

# Extract first line of instruction as subject
subject = entry['instruction'].split('\n')[0]

console.print(f"[cyan]Looking for commit with subject:[/cyan] {subject[:80]}")

query = """
SELECT 
    c.commit_hash,
    fc.file_path,
    fc.diff_hunk,
    length(fc.code_before) as code_len
FROM commits c
JOIN file_changes fc ON c.commit_hash = fc.commit_hash
WHERE c.subject = %(subject)s
LIMIT 1
"""

result = client.execute(query, {'subject': subject})

if result:
    commit_hash, file_path, diff, code_len = result[0]
    
    console.print(f"\n[green]Found commit:[/green] {commit_hash[:12]}")
    console.print(f"[green]File:[/green] {file_path}")
    console.print(f"[green]Code before length:[/green] {code_len:,} chars")
    console.print(f"[green]Diff length:[/green] {len(diff):,} chars\n")
    
    # Now get just the first 10KB of code_before to analyze
    query2 = """
    SELECT substring(code_before, 1, 10000) as code_sample
    FROM file_changes
    WHERE commit_hash = %(hash)s AND file_path = %(path)s
    """
    
    result2 = client.execute(query2, {'hash': commit_hash, 'path': file_path})
    
    if result2:
        code_sample = result2[0][0]
        
        console.print(Panel(
            code_sample[:1000],
            title="[blue]First 1000 chars of code_before from DB[/blue]",
            border_style="blue"
        ))
        
        console.print(Panel(
            entry['input'][:1000],
            title="[yellow]First 1000 chars of exported input[/yellow]",
            border_style="yellow"
        ))
        
        console.print(Panel(
            diff[:800],
            title="[green]Diff (first 800 chars)[/green]",
            border_style="green"
        ))
        
        # Test smart extraction
        console.print("\n[bold cyan]Testing Smart Context Extractor:[/bold cyan]\n")
        extractor = SmartContextExtractor()
        extracted = extractor.extract_changed_region(code_sample, diff, max_chars=4000)
        
        console.print(f"Extracted length: {len(extracted)} chars")
        console.print(Panel(
            extracted[:1000],
            title="[magenta]Smart Extracted Context (first 1000 chars)[/magenta]",
            border_style="magenta"
        ))
        
        # Compare
        if extracted[:500] == entry['input'][:500]:
            console.print("\n[green]✓ Smart extraction matches exported input![/green]")
        else:
            console.print("\n[red]✗ Smart extraction DIFFERS from exported input[/red]")
            console.print("[yellow]This suggests the export is using the wrong code_before[/yellow]")
else:
    console.print("[red]Commit not found![/red]")
