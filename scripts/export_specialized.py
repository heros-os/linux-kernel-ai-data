#!/usr/bin/env python3
"""
Export training data with specialized prompt templates for different training objectives.

Supports multiple modes:
- review: Code review training
- optimize: Performance optimization training
- security: Security vulnerability detection training
- bugfix: Bug fixing training (uses commit chains)
"""

import json
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from rich.console import Console
from tqdm import tqdm

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Prompt templates for different training objectives
TEMPLATES = {
    'review': {
        'system': "You are an expert Linux kernel code reviewer. Analyze patches for bugs, style issues, and potential problems.",
        'instruction_template': "Review this Linux kernel patch and identify any issues:\n\n{diff}",
        'output_template': "Analysis of commit '{subject}':\n\n{body}"
    },
    'optimize': {
        'system': "You are a Linux kernel performance optimization expert. Identify and implement performance improvements.",
        'instruction_template': "Optimize this kernel code for better performance:\n\n{code_before}",
        'output_template': "{code_after}"
    },
    'security': {
        'system': "You are a Linux kernel security expert. Identify and fix security vulnerabilities.",
        'instruction_template': "Analyze this kernel code for security vulnerabilities:\n\n{code_before}",
        'output_template': "Security fix applied:\n\n{diff}\n\nExplanation: {body}"
    },
    'bugfix': {
        'system': "You are a Linux kernel debugging expert. Given buggy code, produce the correct fix.",
        'instruction_template': "This kernel code has a bug. Fix it:\n\n{buggy_code}",
        'output_template': "{fixed_code}"
    },
    'explain': {
        'system': "You are a Linux kernel educator. Explain kernel code changes clearly.",
        'instruction_template': "Explain what this kernel patch does and why:\n\n{diff}",
        'output_template': "{subject}\n\n{body}"
    }
}

def get_client():
    return Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )

def export_review_mode(client, output_path: Path, limit: int = None, min_quality: float = 50):
    """Export data for code review training."""
    console.print("[bold cyan]Exporting Review Mode dataset...[/bold cyan]")
    
    query = f"""
        SELECT 
            c.commit_hash,
            c.subject,
            c.body,
            fc.diff_hunk
        FROM file_changes fc
        INNER JOIN commits c ON fc.commit_hash = c.commit_hash
        LEFT JOIN commit_tags ct ON c.commit_hash = ct.commit_hash
        WHERE fc.is_binary = 0
            AND length(fc.diff_hunk) > 50
            AND c.heuristic_score >= {min_quality}
            AND NOT has(ct.tags, 'backport')
            AND NOT has(ct.tags, 'documentation')
        ORDER BY c.heuristic_score DESC
        {"LIMIT " + str(limit) if limit else ""}
    """
    
    template = TEMPLATES['review']
    count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in tqdm(client.execute_iter(query), desc="Exporting"):
            commit_hash, subject, body, diff = row
            
            record = {
                "system": template['system'],
                "instruction": template['instruction_template'].format(diff=diff[:4000]),
                "output": template['output_template'].format(subject=subject, body=body),
                "_commit_hash": commit_hash
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    
    return count

def export_bugfix_mode(client, output_path: Path, limit: int = None):
    """Export bug-fix pairs using commit chains."""
    console.print("[bold cyan]Exporting Bug Fix Mode dataset...[/bold cyan]")
    
    # Check if commit_chains table exists
    try:
        client.execute("SELECT 1 FROM commit_chains LIMIT 1")
    except Exception:
        console.print("[red]Error: commit_chains table not found. Run build_commit_chains.py first.[/red]")
        return 0
    
    query = f"""
        SELECT 
            cc.buggy_commit,
            cc.fix_commit,
            cc.fix_subject,
            cc.fix_body,
            buggy.code_after as buggy_code,
            fixed.code_after as fixed_code
        FROM commit_chains cc
        INNER JOIN file_changes buggy ON cc.buggy_commit = buggy.commit_hash
        INNER JOIN file_changes fixed ON cc.fix_commit = fixed.commit_hash
            AND buggy.file_path = fixed.file_path
        WHERE cc.chain_type = 'fixes'
            AND length(buggy.code_after) > 50
            AND length(fixed.code_after) > 50
        {"LIMIT " + str(limit) if limit else ""}
    """
    
    template = TEMPLATES['bugfix']
    count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in tqdm(client.execute_iter(query), desc="Exporting"):
            buggy_hash, fix_hash, subject, body, buggy_code, fixed_code = row
            
            record = {
                "system": template['system'],
                "instruction": template['instruction_template'].format(buggy_code=buggy_code[:4000]),
                "output": template['output_template'].format(fixed_code=fixed_code[:4000]),
                "_buggy_commit": buggy_hash,
                "_fix_commit": fix_hash,
                "_explanation": f"{subject}\n{body}"
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    
    return count

def export_security_mode(client, output_path: Path, limit: int = None):
    """Export security-focused training data."""
    console.print("[bold cyan]Exporting Security Mode dataset...[/bold cyan]")
    
    query = f"""
        SELECT 
            c.commit_hash,
            c.subject,
            c.body,
            fc.code_before,
            fc.diff_hunk
        FROM file_changes fc
        INNER JOIN commits c ON fc.commit_hash = c.commit_hash
        INNER JOIN commit_tags ct ON c.commit_hash = ct.commit_hash
        WHERE has(ct.tags, 'security')
            AND fc.is_binary = 0
            AND length(fc.code_before) > 50
        {"LIMIT " + str(limit) if limit else ""}
    """
    
    template = TEMPLATES['security']
    count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in tqdm(client.execute_iter(query), desc="Exporting"):
            commit_hash, subject, body, code_before, diff = row
            
            record = {
                "system": template['system'],
                "instruction": template['instruction_template'].format(code_before=code_before[:4000]),
                "output": template['output_template'].format(diff=diff[:4000], body=body),
                "_commit_hash": commit_hash
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    
    return count

def main():
    parser = argparse.ArgumentParser(description="Export specialized training datasets")
    parser.add_argument("--mode", choices=['review', 'bugfix', 'security', 'all'], default='all',
                        help="Export mode")
    parser.add_argument("--output-dir", "-o", default="exports/specialized",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max examples per dataset")
    parser.add_argument("--min-quality", type=float, default=50, help="Minimum heuristic score")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = get_client()
    results = {}
    
    if args.mode in ['review', 'all']:
        count = export_review_mode(
            client, 
            output_dir / "review_training.jsonl",
            limit=args.limit,
            min_quality=args.min_quality
        )
        results['review'] = count
        console.print(f"[green]Review dataset: {count:,} examples[/green]")
    
    if args.mode in ['bugfix', 'all']:
        count = export_bugfix_mode(
            client,
            output_dir / "bugfix_training.jsonl",
            limit=args.limit
        )
        results['bugfix'] = count
        console.print(f"[green]Bug fix dataset: {count:,} examples[/green]")
    
    if args.mode in ['security', 'all']:
        count = export_security_mode(
            client,
            output_dir / "security_training.jsonl",
            limit=args.limit
        )
        results['security'] = count
        console.print(f"[green]Security dataset: {count:,} examples[/green]")
    
    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"Output directory: {output_dir}")
    
    total = sum(results.values())
    console.print(f"Total examples: {total:,}")

if __name__ == "__main__":
    main()
