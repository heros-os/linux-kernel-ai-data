#!/usr/bin/env python3
"""
Lightweight batch exporter that avoids memory issues.
Exports commits in small batches to avoid ClickHouse memory limits.
"""

import sys
import json
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from clickhouse_driver import Client
from tqdm import tqdm

from config.settings import settings

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Export training data in batches")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=10000, help="Max examples to export")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--min-quality", type=float, default=0, help="Min heuristic score")
    
    args = parser.parse_args()
    
    console.print("[bold blue]Batch Training Data Exporter[/bold blue]")
    console.print(f"Output: {args.output}")
    console.print(f"Limit: {args.limit}, Batch: {args.batch_size}")
    
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    exported = 0
    offset = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        with tqdm(total=args.limit, desc="Exporting") as pbar:
            while exported < args.limit:
                remaining = args.limit - exported
                current_batch = min(args.batch_size, remaining)
                
                # Simple query: just get commits first
                query = f"""
                SELECT 
                    commit_hash,
                    subject || '\n\n' || body as instruction
                FROM commits
                WHERE heuristic_score >= {args.min_quality}
                ORDER BY heuristic_score DESC
                LIMIT {current_batch} OFFSET {offset}
                """
                
                try:
                    commits = client.execute(query)
                except Exception as e:
                    console.print(f"[red]Error fetching commits: {e}[/red]")
                    break
                
                if not commits:
                    console.print("[yellow]No more commits found.[/yellow]")
                    break
                
                # For each commit, get ONE file change
                for commit_hash, instruction in commits:
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
                            
                            record = {
                                "instruction": instruction.strip(),
                                "input": code_before[:2000] if code_before else "",
                                "output": diff
                            }
                            
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            exported += 1
                            pbar.update(1)
                            
                    except Exception as e:
                        continue
                
                offset += current_batch
    
    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"Exported: {exported:,} examples")
    console.print(f"Output: {output_path}")
    
    # File size
    size = output_path.stat().st_size
    if size > 1_000_000:
        console.print(f"Size: {size / 1_000_000:.2f} MB")
    else:
        console.print(f"Size: {size / 1000:.2f} KB")

if __name__ == "__main__":
    main()
