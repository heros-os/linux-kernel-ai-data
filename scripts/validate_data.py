#!/usr/bin/env python3
"""
Data validation script for exported JSONL training data.

Validates structure, calculates statistics, and checks for common issues.
"""

import json
import argparse
import logging
from pathlib import Path
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.progress import track

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


def validate_jsonl(file_path: Path) -> dict:
    """Validate a JSONL file and return statistics."""
    
    stats = {
        'total_lines': 0,
        'valid_records': 0,
        'invalid_records': 0,
        'empty_instructions': 0,
        'empty_outputs': 0,
        'instruction_lengths': [],
        'output_lengths': [],
        'has_system': 0,
        'has_input': 0,
        'subsystems': Counter(),
        'errors': []
    }
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return stats
    
    file_size = file_path.stat().st_size
    console.print(f"[cyan]Validating: {file_path}[/cyan]")
    console.print(f"[dim]File size: {file_size / 1_000_000:.2f} MB[/dim]")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(track(f, description="Scanning"), 1):
            stats['total_lines'] += 1
            
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                stats['valid_records'] += 1
                
                # Check required fields
                instruction = record.get('instruction', '')
                output = record.get('output', '')
                
                if not instruction:
                    stats['empty_instructions'] += 1
                else:
                    stats['instruction_lengths'].append(len(instruction))
                
                if not output:
                    stats['empty_outputs'] += 1
                else:
                    stats['output_lengths'].append(len(output))
                
                # Optional fields
                if 'system' in record:
                    stats['has_system'] += 1
                if record.get('input'):
                    stats['has_input'] += 1
                
                # Track subsystems if present
                if '_subsystem' in record:
                    stats['subsystems'][record['_subsystem']] += 1
                    
            except json.JSONDecodeError as e:
                stats['invalid_records'] += 1
                if len(stats['errors']) < 10:  # Limit error collection
                    stats['errors'].append(f"Line {line_num}: {str(e)[:100]}")
    
    return stats


def print_stats(stats: dict):
    """Print validation statistics in a nice format."""
    
    console.print("\n[bold green]Validation Results[/bold green]\n")
    
    # Summary table
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Total Lines", f"{stats['total_lines']:,}")
    table.add_row("Valid Records", f"{stats['valid_records']:,}")
    table.add_row("Invalid Records", f"{stats['invalid_records']:,}")
    table.add_row("Empty Instructions", f"{stats['empty_instructions']:,}")
    table.add_row("Empty Outputs", f"{stats['empty_outputs']:,}")
    table.add_row("Has System Prompt", f"{stats['has_system']:,}")
    table.add_row("Has Input Field", f"{stats['has_input']:,}")
    
    console.print(table)
    
    # Length statistics
    if stats['instruction_lengths']:
        inst_lengths = stats['instruction_lengths']
        out_lengths = stats['output_lengths']
        
        console.print("\n[bold]Length Statistics:[/bold]")
        console.print(f"  Instruction: min={min(inst_lengths)}, max={max(inst_lengths)}, avg={sum(inst_lengths)/len(inst_lengths):.0f}")
        console.print(f"  Output: min={min(out_lengths)}, max={max(out_lengths)}, avg={sum(out_lengths)/len(out_lengths):.0f}")
        
        # Estimate token count (rough: 4 chars per token)
        total_chars = sum(inst_lengths) + sum(out_lengths)
        est_tokens = total_chars / 4
        console.print(f"\n[bold]Estimated Tokens:[/bold] {est_tokens:,.0f} (~{est_tokens/1_000_000:.1f}M)")
    
    # Top subsystems
    if stats['subsystems']:
        console.print("\n[bold]Top Subsystems:[/bold]")
        for subsys, count in stats['subsystems'].most_common(10):
            console.print(f"  {subsys}: {count:,}")
    
    # Errors
    if stats['errors']:
        console.print("\n[bold red]Parse Errors:[/bold red]")
        for error in stats['errors']:
            console.print(f"  {error}")
    
    # Quality assessment
    console.print("\n[bold]Quality Assessment:[/bold]")
    
    valid_pct = (stats['valid_records'] / stats['total_lines'] * 100) if stats['total_lines'] > 0 else 0
    if valid_pct >= 99:
        console.print(f"  ✅ Validity: {valid_pct:.2f}% (Excellent)")
    elif valid_pct >= 95:
        console.print(f"  ⚠️ Validity: {valid_pct:.2f}% (Good)")
    else:
        console.print(f"  ❌ Validity: {valid_pct:.2f}% (Needs attention)")
    
    empty_pct = ((stats['empty_instructions'] + stats['empty_outputs']) / (stats['valid_records'] * 2) * 100) if stats['valid_records'] > 0 else 0
    if empty_pct <= 1:
        console.print(f"  ✅ Completeness: {100-empty_pct:.2f}% (Excellent)")
    elif empty_pct <= 5:
        console.print(f"  ⚠️ Completeness: {100-empty_pct:.2f}% (Good)")
    else:
        console.print(f"  ❌ Completeness: {100-empty_pct:.2f}% (Needs attention)")


def main():
    parser = argparse.ArgumentParser(description="Validate JSONL training data")
    parser.add_argument("file", type=str, help="Path to JSONL file to validate")
    parser.add_argument("--sample", type=int, default=None, help="Only sample N records")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    stats = validate_jsonl(file_path)
    print_stats(stats)
    
    # Return exit code based on validity
    if stats['invalid_records'] > 0 or stats['valid_records'] == 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
