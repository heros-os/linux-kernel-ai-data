#!/usr/bin/env python3
"""
Show detailed view of a specific training entry
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='JSONL file')
    parser.add_argument('index', type=int, help='Entry index (1-based)')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i == args.index:
                entry = json.loads(line)
                
                console.print(f"\n[bold cyan]Entry #{i}[/bold cyan]\n")
                
                console.print(Panel(
                    entry.get('instruction', ''),
                    title="[yellow]Instruction (Commit Message)[/yellow]",
                    border_style="yellow"
                ))
                
                console.print(Panel(
                    entry.get('input', '')[:2000] + ('...' if len(entry.get('input', '')) > 2000 else ''),
                    title="[blue]Input (Code Before)[/blue]",
                    border_style="blue"
                ))
                
                console.print(Panel(
                    entry.get('output', ''),
                    title="[green]Output (Diff)[/green]",
                    border_style="green"
                ))
                
                if '_quality_score' in entry:
                    console.print(f"\n[bold]Quality Score:[/bold] {entry['_quality_score']}")
                if '_quality_reason' in entry:
                    console.print(f"[bold]Reason:[/bold] {entry['_quality_reason']}")
                
                return
    
    console.print(f"[red]Entry {args.index} not found[/red]")


if __name__ == '__main__':
    main()
