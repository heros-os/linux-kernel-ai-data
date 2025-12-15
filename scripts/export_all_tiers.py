#!/usr/bin/env python3
"""
Export all training dataset tiers in one run.
Generates Premium, High Quality, Standard, and Specialized datasets.
"""

import sys
import json
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from clickhouse_driver import Client
from tqdm import tqdm

from config.settings import settings

console = Console()

def export_tier(client, output_path: Path, min_quality: float, limit: int, 
                 batch_size: int = 100, include_reasoning: bool = False):
    """Export a single tier of training data."""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    exported = 0
    offset = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        with tqdm(total=limit, desc=f"  {output_path.name}", leave=False) as pbar:
            while exported < limit:
                remaining = limit - exported
                current_batch = min(batch_size, remaining)
                
                # Get commits - include AI reason if requested
                if include_reasoning:
                    query = f"""
                    SELECT 
                        commit_hash,
                        subject || '\n\n' || body as instruction,
                        ai_quality_score,
                        ai_score_reason
                    FROM commits
                    WHERE heuristic_score >= {min_quality}
                        AND ai_quality_score > 0
                    ORDER BY ai_quality_score DESC, heuristic_score DESC
                    LIMIT {current_batch} OFFSET {offset}
                    """
                else:
                    query = f"""
                    SELECT 
                        commit_hash,
                        subject || '\n\n' || body as instruction
                    FROM commits
                    WHERE heuristic_score >= {min_quality}
                    ORDER BY heuristic_score DESC
                    LIMIT {current_batch} OFFSET {offset}
                    """
                
                try:
                    commits = client.execute(query)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    break
                
                if not commits:
                    break
                
                for row in commits:
                    if include_reasoning:
                        commit_hash, instruction, ai_score, ai_reason = row
                    else:
                        commit_hash, instruction = row
                        ai_score, ai_reason = None, None
                    
                    try:
                        fc_query = f"""
                        SELECT 
                            code_before,
                            diff_hunk
                        FROM file_changes
                        WHERE commit_hash = '{commit_hash}'
                            AND is_binary = 0
                            AND length(diff_hunk) > 50
                            AND length(diff_hunk) < 5000
                        LIMIT 1
                        """
                        
                        fc_result = client.execute(fc_query)
                        
                        if fc_result:
                            code_before, diff = fc_result[0]
                            
                            if include_reasoning and ai_reason:
                                # Include reasoning in the instruction
                                record = {
                                    "instruction": instruction.strip(),
                                    "input": code_before[:2000] if code_before else "",
                                    "output": diff,
                                    "_quality_score": ai_score,
                                    "_quality_reason": ai_reason
                                }
                            else:
                                record = {
                                    "instruction": instruction.strip(),
                                    "input": code_before[:2000] if code_before else "",
                                    "output": diff
                                }
                            
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            exported += 1
                            pbar.update(1)
                            
                    except Exception:
                        continue
                
                offset += current_batch
    
    return exported

def main():
    parser = argparse.ArgumentParser(description="Export all training data tiers")
    parser.add_argument("--output-dir", "-o", default="exports", help="Output directory")
    parser.add_argument("--premium-limit", type=int, default=10000, help="Premium tier limit")
    parser.add_argument("--hq-limit", type=int, default=50000, help="High quality tier limit")
    parser.add_argument("--standard-limit", type=int, default=100000, help="Standard tier limit")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--with-reasoning", action="store_true", help="Also export versions with AI reasoning")
    parser.add_argument("--reasoning-limit", type=int, default=5000, help="Limit for reasoning datasets (requires AI-scored commits)")
    
    args = parser.parse_args()
    
    console.print("[bold blue]============================================[/bold blue]")
    console.print("[bold blue]    Linux Kernel Training Data Exporter     [/bold blue]")
    console.print("[bold blue]           All Tiers Generator              [/bold blue]")
    console.print("[bold blue]============================================[/bold blue]")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    results = {}
    
    # Standard tier definitions
    tiers = [
        ("premium.jsonl", 90, args.premium_limit, "[PREMIUM] (Heuristic >=90)", False),
        ("high_quality.jsonl", 70, args.hq_limit, "[HQ] High Quality (Heuristic >=70)", False),
        ("standard.jsonl", 50, args.standard_limit, "[STD] Standard (Heuristic >=50)", False),
    ]
    
    # Add reasoning tiers if requested
    if args.with_reasoning:
        tiers.extend([
            ("premium_with_reasoning.jsonl", 90, args.reasoning_limit, "[PREMIUM+AI] Premium + AI Reasoning", True),
            ("high_quality_with_reasoning.jsonl", 70, args.reasoning_limit, "[HQ+AI] High Quality + AI Reasoning", True),
        ])
    
    console.print("\n[bold cyan]Exporting Tiered Datasets...[/bold cyan]\n")
    
    for filename, min_quality, limit, description, include_reasoning in tiers:
        console.print(f"[yellow]{description}[/yellow]")
        output_path = output_dir / filename
        
        count = export_tier(
            client, 
            output_path, 
            min_quality=min_quality, 
            limit=limit,
            batch_size=args.batch_size,
            include_reasoning=include_reasoning
        )
        
        size = output_path.stat().st_size / 1_000_000 if output_path.exists() and output_path.stat().st_size > 0 else 0
        results[filename] = {"count": count, "size": size}
        console.print(f"  [green]OK - Exported {count:,} examples ({size:.2f} MB)[/green]\n")
    
    # Summary table
    console.print("\n[bold]==============================================[/bold]")
    console.print("[bold green]Export Complete![/bold green]\n")
    
    table = Table(title="Generated Datasets")
    table.add_column("File", style="cyan")
    table.add_column("Examples", justify="right")
    table.add_column("Size (MB)", justify="right")
    
    total_examples = 0
    total_size = 0
    
    for filename, data in results.items():
        table.add_row(filename, f"{data['count']:,}", f"{data['size']:.2f}")
        total_examples += data['count']
        total_size += data['size']
    
    table.add_row("-" * 15, "-" * 8, "-" * 8)
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_examples:,}[/bold]", f"[bold]{total_size:.2f}[/bold]")
    
    console.print(table)
    
    console.print(f"\n[dim]Output directory: {output_dir.absolute()}[/dim]")
    console.print("\n[bold cyan]Next: Run validation on each file:[/bold cyan]")
    console.print(f"  python scripts/validate_data.py {output_dir}/premium.jsonl")

if __name__ == "__main__":
    main()
