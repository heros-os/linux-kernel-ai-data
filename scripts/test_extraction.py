#!/usr/bin/env python3
"""
Test smart context extraction on specific commits to debug issues.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from src.context_extractor import SmartContextExtractor
import re

console = Console()

def parse_diff_line_numbers(diff: str) -> list:
    """Extract line numbers from diff hunks."""
    hunk_pattern = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'
    hunks = []
    
    for match in re.finditer(hunk_pattern, diff):
        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        hunks.append({
            'old_start': old_start,
            'old_count': old_count,
            'old_end': old_start + old_count - 1
        })
    
    return hunks


def test_single_commit(client: Client, commit_hash: str):
    """Test extraction for a single commit."""
    
    console.print(f"\n[bold cyan]Testing commit: {commit_hash[:12]}[/bold cyan]\n")
    
    # Get commit info
    query = """
    SELECT 
        c.subject,
        fc.file_path,
        fc.diff_hunk,
        substring(fc.code_before, 1, 50000) as code_sample
    FROM commits c
    JOIN file_changes fc ON c.commit_hash = fc.commit_hash
    WHERE c.commit_hash = %(hash)s
        AND fc.is_binary = 0
        AND length(fc.diff_hunk) > 100
    LIMIT 1
    """
    
    try:
        result = client.execute(query, {'hash': commit_hash})
        
        if not result:
            console.print("[red]No suitable file changes found[/red]")
            return False
        
        subject, file_path, diff, code_sample = result[0]
        
        console.print(f"[yellow]Subject:[/yellow] {subject}")
        console.print(f"[yellow]File:[/yellow] {file_path}")
        console.print(f"[yellow]Code sample length:[/yellow] {len(code_sample):,} chars")
        console.print(f"[yellow]Diff length:[/yellow] {len(diff):,} chars\n")
        
        # Parse diff to see what lines are being changed
        hunks = parse_diff_line_numbers(diff)
        console.print(f"[cyan]Diff hunks:[/cyan]")
        for i, hunk in enumerate(hunks, 1):
            console.print(f"  Hunk {i}: lines {hunk['old_start']}-{hunk['old_end']} ({hunk['old_count']} lines)")
        
        # Show what's at those line numbers in code_before
        console.print(f"\n[cyan]Code at those line numbers:[/cyan]")
        lines = code_sample.split('\n')
        
        if hunks:
            first_hunk = hunks[0]
            start = max(0, first_hunk['old_start'] - 1)  # 0-indexed
            end = min(len(lines), first_hunk['old_end'] + 5)
            
            if start < len(lines):
                context_lines = lines[start:end]
                console.print(Panel(
                    '\n'.join(f"{start + i + 1:4d}: {line}" for i, line in enumerate(context_lines)),
                    title=f"[blue]Lines {start + 1}-{end} from code_before[/blue]",
                    border_style="blue"
                ))
            else:
                console.print(f"[red]ERROR: Line {start + 1} exceeds code_sample length ({len(lines)} lines)[/red]")
                console.print(f"[yellow]This suggests code_before is truncated or doesn't match the diff![/yellow]")
        
        # Test smart extraction
        console.print(f"\n[cyan]Testing SmartContextExtractor:[/cyan]")
        extractor = SmartContextExtractor()
        extracted = extractor.extract_changed_region(code_sample, diff, max_chars=4000)
        
        console.print(f"Extracted length: {len(extracted):,} chars\n")
        console.print(Panel(
            extracted[:1500] if len(extracted) > 1500 else extracted,
            title="[green]Extracted Context (first 1500 chars)[/green]",
            border_style="green"
        ))
        
        # Show the diff for comparison
        console.print(Panel(
            diff[:1000] if len(diff) > 1000 else diff,
            title="[magenta]Diff (first 1000 chars)[/magenta]",
            border_style="magenta"
        ))
        
        # Check if extracted content contains the changed lines
        changed_line_samples = []
        for line in diff.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                changed_line_samples.append(line[1:].strip())
            elif line.startswith('-') and not line.startswith('---'):
                changed_line_samples.append(line[1:].strip())
        
        found_count = 0
        for sample in changed_line_samples[:10]:  # Check first 10
            if sample and sample in extracted:
                found_count += 1
        
        coverage = (found_count / min(10, len(changed_line_samples)) * 100) if changed_line_samples else 0
        
        console.print(f"\n[bold]Coverage Check:[/bold]")
        console.print(f"  Changed lines in extracted: {found_count}/{min(10, len(changed_line_samples))}")
        console.print(f"  Coverage: {coverage:.1f}%")
        
        if coverage >= 80:
            console.print(f"  [green]PASS[/green]")
            return True
        elif coverage >= 50:
            console.print(f"  [yellow]WARN[/yellow]")
            return False
        else:
            console.print(f"  [red]FAIL[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test smart context extraction")
    parser.add_argument('--commits', nargs='+', help='Commit hashes to test')
    parser.add_argument('--auto', action='store_true', help='Auto-select diverse commits')
    parser.add_argument('-n', '--num', type=int, default=5, help='Number of commits to test in auto mode')
    
    args = parser.parse_args()
    
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    if args.auto:
        # Get diverse commits
        console.print(f"[bold blue]Auto-selecting {args.num} diverse commits...[/bold blue]")
        
        query = """
        SELECT DISTINCT commit_hash
        FROM commits
        WHERE heuristic_score >= 8.0
        ORDER BY rand()
        LIMIT %(limit)s
        """
        
        results = client.execute(query, {'limit': args.num})
        commit_hashes = [row[0] for row in results]
        
        console.print(f"Selected {len(commit_hashes)} commits\n")
    else:
        commit_hashes = args.commits or []
    
    if not commit_hashes:
        console.print("[red]No commits specified. Use --commits or --auto[/red]")
        return
    
    # Test each commit
    passed = 0
    failed = 0
    
    for commit_hash in commit_hashes:
        result = test_single_commit(client, commit_hash)
        if result:
            passed += 1
        else:
            failed += 1
        
        console.print("\n" + "="*80 + "\n")
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Passed: {passed}/{len(commit_hashes)}")
    console.print(f"  Failed: {failed}/{len(commit_hashes)}")
    
    if passed == len(commit_hashes):
        console.print(f"\n[bold green]All tests passed![/bold green]")
    elif passed >= len(commit_hashes) * 0.8:
        console.print(f"\n[bold yellow]Most tests passed, but some issues remain[/bold yellow]")
    else:
        console.print(f"\n[bold red]Many tests failed - extraction needs fixes[/bold red]")


if __name__ == '__main__':
    main()
