#!/usr/bin/env python3
"""
Improved validation that checks if extracted context contains the CONTEXT
needed to generate the diff, not the exact diff lines themselves.
"""

import json
import sys
import re
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def extract_context_lines_from_diff(diff: str) -> set:
    """
    Extract the context lines (unchanged lines) and deleted lines from diff.
    These SHOULD exist in code_before.
    Added lines won't exist, so we don't check for them.
    """
    context_lines = set()
    
    for line in diff.split('\n'):
        # Skip diff headers
        if line.startswith('diff ') or line.startswith('index ') or \
           line.startswith('---') or line.startswith('+++') or \
           line.startswith('@@'):
            continue
        
        # Context lines (no prefix or space prefix)
        if line and not line[0] in ['+', '-']:
            context_lines.add(line.strip())
        # Deleted lines (these existed in code_before)
        elif line.startswith('-'):
            context_lines.add(line[1:].strip())
        # Skip added lines - they won't be in code_before
    
    # Remove empty lines
    context_lines.discard('')
    
    return context_lines


def calculate_smart_coverage(input_code: str, diff: str) -> dict:
    """
    Calculate coverage based on whether the input contains the CONTEXT
    needed to understand and generate the diff.
    
    Checks:
    1. Are the unchanged context lines present?
    2. Are the deleted lines present?
    3. Is the function/block structure present?
    """
    context_lines = extract_context_lines_from_diff(diff)
    
    if not context_lines:
        return {
            'total_context_lines': 0,
            'found_lines': 0,
            'coverage_percent': 0,
            'has_input': len(input_code.strip()) > 0
        }
    
    # Normalize input for matching
    input_normalized = set()
    for line in input_code.split('\n'):
        stripped = line.strip()
        if stripped:
            input_normalized.add(stripped)
    
    # Count how many context lines are found
    found_count = 0
    missing_lines = []
    
    for context_line in context_lines:
        # Try exact match first
        if context_line in input_normalized:
            found_count += 1
        else:
            # Try fuzzy match (for whitespace differences)
            normalized_context = ' '.join(context_line.split())
            found = False
            for input_line in input_normalized:
                normalized_input = ' '.join(input_line.split())
                if normalized_context in normalized_input or normalized_input in normalized_context:
                    found = True
                    found_count += 1
                    break
            
            if not found:
                missing_lines.append(context_line)
    
    coverage_pct = (found_count / len(context_lines) * 100) if context_lines else 0
    
    return {
        'total_context_lines': len(context_lines),
        'found_lines': found_count,
        'missing_lines': missing_lines[:5],  # First 5 missing
        'coverage_percent': coverage_pct,
        'has_input': len(input_code.strip()) > 0
    }


def validate_entry(entry: dict, index: int) -> dict:
    """Validate a single training entry with improved coverage calculation."""
    instruction = entry.get('instruction', '')
    input_code = entry.get('input', '')
    output_diff = entry.get('output', '')
    
    # Basic checks
    has_instruction = len(instruction.strip()) > 0
    has_input = len(input_code.strip()) > 0
    has_output = len(output_diff.strip()) > 0
    
    # Smart coverage check
    coverage = calculate_smart_coverage(input_code, output_diff)
    
    return {
        'index': index,
        'has_instruction': has_instruction,
        'has_input': has_input,
        'has_output': has_output,
        'instruction_len': len(instruction),
        'input_len': len(input_code),
        'output_len': len(output_diff),
        **coverage
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate smart export with improved metric")
    parser.add_argument('file', help='JSONL file to validate')
    parser.add_argument('-n', '--num-entries', type=int, default=10,
                       help='Number of entries to validate')
    parser.add_argument('--show-failures', action='store_true',
                       help='Show entries with low coverage')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        sys.exit(1)
    
    console.print(f"[bold blue]Validating Smart Export (Improved Metric)[/bold blue]")
    console.print(f"File: {file_path.name}")
    console.print(f"Checking {args.num_entries} entries...\n")
    
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.num_entries:
                break
            
            try:
                entry = json.loads(line)
                result = validate_entry(entry, i + 1)
                results.append(result)
                
                if args.verbose:
                    console.print(f"\n[cyan]Entry #{i+1}[/cyan]")
                    console.print(f"  Coverage: {result['coverage_percent']:.1f}%")
                    console.print(f"  Context lines: {result['found_lines']}/{result['total_context_lines']}")
                    
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing line {i+1}: {e}[/red]")
                continue
    
    # Summary table
    console.print("\n[bold]Validation Results[/bold]\n")
    
    table = Table(title="Coverage Analysis")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Input Len", justify="right")
    table.add_column("Context Lines", justify="right")
    table.add_column("Found", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Status", justify="center")
    
    total_coverage = 0
    high_coverage = 0
    medium_coverage = 0
    low_coverage = 0
    
    for r in results:
        status = "PASS" if r['coverage_percent'] >= 90 else "GOOD" if r['coverage_percent'] >= 75 else "WARN" if r['coverage_percent'] >= 50 else "FAIL"
        status_color = "green" if r['coverage_percent'] >= 90 else "blue" if r['coverage_percent'] >= 75 else "yellow" if r['coverage_percent'] >= 50 else "red"
        
        table.add_row(
            str(r['index']),
            f"{r['input_len']}",
            f"{r['total_context_lines']}",
            f"{r['found_lines']}",
            f"{r['coverage_percent']:.1f}%",
            f"[{status_color}]{status}[/{status_color}]"
        )
        
        total_coverage += r['coverage_percent']
        
        if r['coverage_percent'] >= 90:
            high_coverage += 1
        elif r['coverage_percent'] >= 75:
            medium_coverage += 1
        elif r['coverage_percent'] >= 50:
            low_coverage += 1
    
    console.print(table)
    
    # Overall metrics
    avg_coverage = total_coverage / len(results) if results else 0
    
    console.print(f"\n[bold]Overall Metrics:[/bold]")
    console.print(f"  Total entries validated: {len(results)}")
    console.print(f"  Average coverage: {avg_coverage:.1f}%")
    console.print(f"  High coverage (>=90%): {high_coverage}")
    console.print(f"  Good coverage (>=75%): {medium_coverage}")
    console.print(f"  Medium coverage (>=50%): {low_coverage}")
    console.print(f"  Low coverage (<50%): {len(results) - high_coverage - medium_coverage - low_coverage}")
    
    # Show failures if requested
    if args.show_failures:
        low_cov = [r for r in results if r['coverage_percent'] < 75]
        if low_cov:
            console.print(f"\n[yellow]Entries with <75% coverage:[/yellow]\n")
            for r in low_cov[:5]:
                console.print(f"[cyan]Entry #{r['index']}[/cyan] - Coverage: {r['coverage_percent']:.1f}%")
                if r['missing_lines']:
                    console.print("  Missing context:")
                    for line in r['missing_lines']:
                        console.print(f"    - {line[:80]}...")
                console.print()
    
    # Final verdict
    console.print()
    if avg_coverage >= 90:
        console.print("[bold green]EXCELLENT: Smart export is working very well![/bold green]")
    elif avg_coverage >= 75:
        console.print("[bold blue]GOOD: Smart export is working well[/bold blue]")
    elif avg_coverage >= 60:
        console.print("[bold yellow]ACCEPTABLE: Smart export is working but could be improved[/bold yellow]")
    else:
        console.print("[bold red]NEEDS WORK: Smart export needs improvements[/bold red]")


if __name__ == '__main__':
    main()
