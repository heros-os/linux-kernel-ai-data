#!/usr/bin/env python3
"""
Post-processing script to classify commits by type (Security, Performance, etc.).
Updates the 'quality_flags' column in ClickHouse.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Hardware/Driver Patterns
DRIVER_PATTERNS = [
    r'drivers/', r'arch/', r'soc/', r'firmware/', r'platform/',
    r'usb:', r'net:', r'scsi:', r'drm:', r'media:', r'sound:'
]

# Classification Patterns
PATTERNS = {
    'security': re.compile(
        r'\b(CVE-\d{4}-\d+|overflow|use-after-free|uaf|double-free|bounds check|'
        r'sanitizer|leak|vulnerability|exploit|syzkaller|locking|race condition)\b',
        re.IGNORECASE
    ),
    'performance': re.compile(
        r'\b(optimize|performance|speed up|latency|throughput|cache|'
        r'alloc|memory usage|cpu usage|overhead|fast path|bottleneck)\b',
        re.IGNORECASE
    ),
    'refactor': re.compile(
        r'\b(refactor|cleanup|simplify|restructure|rename|move|remove|delete)\b',
        re.IGNORECASE
    ),
    'documentation': re.compile(
        r'\b(doc|comment|typo|spelling|warning|readme)\b',
        re.IGNORECASE
    ),
    'backport': re.compile(
        r'(cherry picked from commit|backport|upstream commit)',
        re.IGNORECASE
    ),
    'revert': re.compile(
        r'^Revert "',
        re.IGNORECASE
    )
}

# Bitmask values for quality_flags
FLAGS = {
    'security': 1 << 0,      # 1
    'performance': 1 << 1,   # 2
    'refactor': 1 << 2,      # 4
    'documentation': 1 << 3, # 8
    'driver': 1 << 4,        # 16
    'backport': 1 << 5,      # 32
    'revert': 1 << 6         # 64
}

def get_client():
    return Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )

def classify_commits(batch_size: int = 10000):
    client = get_client()
    
    # Get total count
    total = client.execute("SELECT count() FROM commits")[0][0]
    console.print(f"[bold cyan]Classifying {total:,} commits...[/bold cyan]")
    
    offset = 0
    updates = 0
    
    while offset < total:
        # Fetch batch
        rows = client.execute(
            f"""
            SELECT commit_hash, subject, body, files_changed 
            FROM commits 
            ORDER BY commit_hash 
            LIMIT {batch_size} OFFSET {offset}
            """
        )
        
        if not rows:
            break
            
        batch_updates = []
        
        for row in rows:
            commit_hash, subject, body, files_changed = row
            text = f"{subject}\n{body}"
            flags = 0
            
            # Check regrex patterns
            if PATTERNS['security'].search(text):
                flags |= FLAGS['security']
            
            if PATTERNS['performance'].search(text):
                flags |= FLAGS['performance']
                
            if PATTERNS['refactor'].search(text):
                flags |= FLAGS['refactor']
                
            if PATTERNS['documentation'].search(text):
                flags |= FLAGS['documentation']
                
            if PATTERNS['backport'].search(text):
                flags |= FLAGS['backport']
                
            if PATTERNS['revert'].search(subject):
                flags |= FLAGS['revert']
                
            # Check driver heuristic (subject prefix)
            if any(re.search(p, subject, re.IGNORECASE) for p in DRIVER_PATTERNS):
                flags |= FLAGS['driver']
            
            # Only update if we found something
            if flags > 0:
                batch_updates.append((flags, commit_hash))
        
        # specific update query for ClickHouse (using mutation or specialized update)
        # For bulk updates in ClickHouse, it's best to use `ALTER TABLE UPDATE`
        # OR insert into a generic temp table and join?
        # Actually, standard UPDATE is async mutations.
        # Better approach: We can't easily update 1M rows one by one.
        # Efficient way: Use clickhouse-driver with `execute` generic with parameters is tricky for updates.
        
        # Let's try constructing a big CASE statement or using the client to execute many mutations? NO.
        # ClickHouse mutations are heavy.
        # OPTIMAL WAY: Dict lookup?
        # Since this is a "one-off" post-process, maybe we just print what we found for now?
        # NO, user wants to use data.
        
        # Plan B: "INSERT INTO commits (commit_hash, quality_flags) VALUES ..." works if we use ReplacingMergeTree?
        # But we are using standard MergeTree (deduplication not guaranteed unless Replacing).
        
        # Let's check schema/clickhouse.sql. It IS MergeTree.
        # Updating 1M rows in MergeTree is costly.
        
        # ALTERNATIVE: Create a "commit_classifications" join table?
        # CREATE TABLE classifications (commit_hash FixedString(40), tags UInt32) ENGINE = Join(ANY, LEFT, commit_hash);
        # This is fast.
        
        # Wait, let's keep it simple. `ALTER TABLE commits UPDATE quality_flags = bitOr(quality_flags, {val}) WHERE commit_hash = '{hash}'`
        # Doing this 1M times is impossible.
        
        # BEST APPROACH: Re-ingest? No.
        # We will create a new table `commit_tags` and users can JOIN.
        
        if batch_updates:
            # Prepare data for insertion
            insert_data = []
            for flags, commit_hash in batch_updates:
                # Convert flags back to string list for readability
                tags = []
                for name, bit in FLAGS.items():
                    if flags & bit:
                        tags.append(name)
                insert_data.append({
                    'commit_hash': commit_hash,
                    'flags': flags,
                    'tags': tags
                })
            
            # Batch insert
            client.execute(
                'INSERT INTO commit_tags (commit_hash, flags, tags) VALUES',
                insert_data
            )

        offset += batch_size
        updates += len(batch_updates)
        console.print(f"Processed {offset:,}/{total:,} (Found tags for {len(batch_updates):,})")

    console.print(f"[green]Classification complete! Tagged {updates:,} commits.[/green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()
    
    client = get_client()
    
    # Reset table
    console.print("[yellow]Recreating commit_tags table...[/yellow]")
    client.execute("DROP TABLE IF EXISTS commit_tags")
    client.execute("""
        CREATE TABLE commit_tags (
            commit_hash FixedString(40),
            flags UInt32,
            tags Array(String)
        ) ENGINE = MergeTree() ORDER BY commit_hash
    """)
    
    classify_commits(args.batch_size)
