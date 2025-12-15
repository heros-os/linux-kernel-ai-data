#!/usr/bin/env python3
"""
Build commit chains by linking bug-introducing commits to their fixes.
Uses the 'Fixes:' tag in commit messages.

Creates a 'commit_chains' table for generating paired training examples.
"""

import logging
import re
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from rich.console import Console
from rich.progress import track

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Regex to extract Fixes: tag
FIXES_PATTERN = re.compile(r'Fixes:\s*([0-9a-fA-F]{12,40})', re.MULTILINE)

def get_client():
    return Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )

def build_commit_chains(batch_size: int = 10000):
    client = get_client()
    
    # Create chains table
    console.print("[yellow]Creating commit_chains table...[/yellow]")
    client.execute("DROP TABLE IF EXISTS commit_chains")
    client.execute("""
        CREATE TABLE commit_chains (
            fix_commit FixedString(40),
            buggy_commit FixedString(40),
            fix_subject String,
            fix_body String,
            chain_type Enum8('fixes' = 1, 'revert' = 2)
        ) ENGINE = MergeTree() ORDER BY (fix_commit, buggy_commit)
    """)
    
    # Get all commits with Fixes: tag
    total = client.execute("SELECT count() FROM commits WHERE fixes_hash != ''")[0][0]
    console.print(f"[bold cyan]Processing {total:,} commits with Fixes: tags...[/bold cyan]")
    
    offset = 0
    chains_found = 0
    
    while offset < total:
        rows = client.execute(f"""
            SELECT commit_hash, subject, body, fixes_hash
            FROM commits
            WHERE fixes_hash != ''
            ORDER BY commit_hash
            LIMIT {batch_size} OFFSET {offset}
        """)
        
        if not rows:
            break
            
        insert_data = []
        
        for fix_hash, subject, body, buggy_hash in rows:
            # Verify the buggy commit exists in our database
            exists = client.execute(
                f"SELECT 1 FROM commits WHERE commit_hash = '{buggy_hash}' LIMIT 1"
            )
            
            if exists:
                insert_data.append({
                    'fix_commit': fix_hash,
                    'buggy_commit': buggy_hash,
                    'fix_subject': subject,
                    'fix_body': body,
                    'chain_type': 'fixes'
                })
        
        if insert_data:
            client.execute(
                'INSERT INTO commit_chains (fix_commit, buggy_commit, fix_subject, fix_body, chain_type) VALUES',
                insert_data
            )
            chains_found += len(insert_data)
        
        offset += batch_size
        console.print(f"Processed {offset:,}/{total:,} (Found {len(insert_data):,} valid chains)")
    
    # Also find revert chains
    console.print("\n[yellow]Finding revert chains...[/yellow]")
    revert_count = client.execute("SELECT count() FROM commits WHERE subject LIKE 'Revert \"%'")[0][0]
    console.print(f"[bold cyan]Processing {revert_count:,} revert commits...[/bold cyan]")
    
    # For reverts, we need to extract the original commit from the message
    # Revert messages typically contain the original commit hash
    reverts = client.execute(f"""
        SELECT commit_hash, subject, body
        FROM commits
        WHERE subject LIKE 'Revert "%'
        LIMIT 100000
    """)
    
    revert_pattern = re.compile(r'This reverts commit ([0-9a-fA-F]{12,40})')
    revert_data = []
    
    for fix_hash, subject, body in reverts:
        match = revert_pattern.search(body)
        if match:
            buggy_hash = match.group(1)
            if len(buggy_hash) < 40:
                # Try to find full hash
                full_hash = client.execute(
                    f"SELECT commit_hash FROM commits WHERE commit_hash LIKE '{buggy_hash}%' LIMIT 1"
                )
                if full_hash:
                    buggy_hash = full_hash[0][0]
                else:
                    continue
            
            revert_data.append({
                'fix_commit': fix_hash,
                'buggy_commit': buggy_hash,
                'fix_subject': subject,
                'fix_body': body,
                'chain_type': 'revert'
            })
    
    if revert_data:
        client.execute(
            'INSERT INTO commit_chains (fix_commit, buggy_commit, fix_subject, fix_body, chain_type) VALUES',
            revert_data
        )
        chains_found += len(revert_data)
    
    console.print(f"\n[green]Commit chain building complete![/green]")
    console.print(f"[bold]Total chains: {chains_found:,}[/bold]")
    console.print(f"  - Bug fixes: {chains_found - len(revert_data):,}")
    console.print(f"  - Reverts: {len(revert_data):,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()
    
    build_commit_chains(args.batch_size)
