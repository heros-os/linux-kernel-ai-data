#!/usr/bin/env python3
"""
Validate Smart Export Quality

Checks if the exported training data contains sufficient context
in the input field to generate the output diffs.
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def extract_changed_lines_from_diff(diff: str) -> set:
    """Extract line numbers and content that were changed in the diff."""
    changed_lines = set()
    for line in diff.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            # Addition
            changed_lines.add(line[1:].strip())
        elif line.startswith('-') and not line.startswith('---'):
            # Deletion
            changed_lines.add(line[1:].strip())
    return changed_lines


def check_context_coverage(input_code: str, diff: str) -> dict:
    """
    Check if the input code contains the necessary context for the diff.
    
    Returns:
        dict with coverage metrics
    """
    # Extract changed lines from diff
    changed_lines = extract_changed_lines_from_diff(diff)
    
    # Get context from input
    input_lines = set(line.strip() for line in input_code.split('\n') if line.strip())
    
    # Check how many changed lines have context in input
    lines_with_context = 0
    lines_without_context = []
    
    for changed_line in changed_lines:
        if not changed_line:  # Skip empty lines
            continue
            
        # Check if this line or similar context exists in input
        found = False
        for input_line in input_lines:
            # Exact match or contains the key part
            if changed_line in input_line or input_line in changed_line:
                found = True
                break
        
        if found:
            lines_with_context += 1
        else:
            lines_without_context.append(changed_line)
    
    total_changed = len([l for l in changed_lines if l])
    coverage_pct = (lines_with_context / total_changed * 100) if total_changed > 0 else 0
    
    return {
        'total_changed_lines': total_changed,
        'lines_with_context': lines_with_context,
        'lines_without_context': lines_without_context,
        'coverage_percent': coverage_pct,
        'has_input': len(input_code.strip()) > 0
    }


def extract_function_from_diff(diff: str) -> str:
    """Extract the function name being modified from the diff."""
    for line in diff.split('\n'):
        if line.startswith('@@'):
            # Try to extract function name from context
            if '@@' in line[2:]:
                context = line.split('@@')[1].strip()
                return context
    return "unknown"


def validate_entry(entry: dict, index: int) -> dict:
    """Validate a single training entry."""
    instruction = entry.get('instruction', '')
    input_code = entry.get('input', '')
    output_diff = entry.get('output', '')
    
    # Basic checks
    has_instruction = len(instruction.strip()) > 0
    has_input = len(input_code.strip()) > 0
    has_output = len(output_diff.strip()) > 0
    
    # Context coverage check
    coverage = check_context_coverage(input_code, output_diff)
    
    # Extract metadata
    function_name = extract_function_from_diff(output_diff)
    
    return {
        'index': index,
        'has_instruction': has_instruction,
        'has_input': has_input,
        'has_output': has_output,
        'instruction_len': len(instruction),
        'input_len': len(input_code),
        'output_len': len(output_diff),
        'function': function_name,
        **coverage
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate smart export quality")
    parser.add_argument('file', help='JSONL file to validate')
    parser.add_argument('-n', '--num-entries', type=int, default=10,
                       help='Number of entries to validate')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output for each entry')
    parser.add_argument('--show-failures', action='store_true',
                       help='Show entries with low coverage')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        sys.exit(1)
    
    console.print(f"[bold blue]Validating Smart Export Quality[/bold blue]")
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
                    console.print(f"  Instruction: {result['instruction_len']} chars")
                    console.print(f"  Input: {result['input_len']} chars")
                    console.print(f"  Output: {result['output_len']} chars")
                    console.print(f"  Function: {result['function']}")
                    console.print(f"  Coverage: {result['coverage_percent']:.1f}%")
                    console.print(f"  Changed lines: {result['total_changed_lines']}")
                    console.print(f"  Lines with context: {result['lines_with_context']}")
                    
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing line {i+1}: {e}[/red]")
                continue
    
    # Summary table
    console.print("\n[bold]Summary Statistics[/bold]\n")
    
    table = Table(title="Validation Results")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Instruction", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Changed", justify="right")
    table.add_column("Status", justify="center")
    
    total_coverage = 0
    entries_with_input = 0
    low_coverage_entries = []
    
    for r in results:
        status = "OK" if r['coverage_percent'] >= 80 else "WARN" if r['coverage_percent'] >= 50 else "FAIL"
        status_color = "green" if r['coverage_percent'] >= 80 else "yellow" if r['coverage_percent'] >= 50 else "red"
        
        table.add_row(
            str(r['index']),
            f"{r['instruction_len']}",
            f"{r['input_len']}",
            f"{r['output_len']}",
            f"{r['coverage_percent']:.1f}%",
            str(r['total_changed_lines']),
            f"[{status_color}]{status}[/{status_color}]"
        )
        
        total_coverage += r['coverage_percent']
        if r['has_input']:
            entries_with_input += 1
        
        if r['coverage_percent'] < 80:
            low_coverage_entries.append(r)
    
    console.print(table)
    
    # Overall metrics
    avg_coverage = total_coverage / len(results) if results else 0
    
    console.print(f"\n[bold]Overall Metrics:[/bold]")
    console.print(f"  Total entries validated: {len(results)}")
    console.print(f"  Entries with input: {entries_with_input}/{len(results)}")
    console.print(f"  Average coverage: {avg_coverage:.1f}%")
    console.print(f"  Entries with >=80% coverage: {sum(1 for r in results if r['coverage_percent'] >= 80)}")
    console.print(f"  Entries with <80% coverage: {len(low_coverage_entries)}")
    
    # Show low coverage entries if requested
    if args.show_failures and low_coverage_entries:
        console.print(f"\n[yellow]Entries with <80% coverage:[/yellow]\n")
        for r in low_coverage_entries[:3]:  # Show first 3
            console.print(f"[cyan]Entry #{r['index']}[/cyan] - Coverage: {r['coverage_percent']:.1f}%")
            if r['lines_without_context']:
                console.print("  Missing context for:")
                for line in r['lines_without_context'][:5]:  # Show first 5
                    console.print(f"    - {line[:80]}...")
            console.print()
    
    # Final verdict
    console.print()
    if avg_coverage >= 80:
        console.print("[bold green]PASS: Smart export is working well![/bold green]")
        console.print("  Input fields contain sufficient context for generating diffs.")
    elif avg_coverage >= 60:
        console.print("[bold yellow]WARNING: Smart export needs improvement[/bold yellow]")
        console.print("  Some entries lack sufficient context.")
    else:
        console.print("[bold red]FAIL: Smart export has issues[/bold red]")
        console.print("  Many entries lack necessary context.")


if __name__ == '__main__':
    main()
