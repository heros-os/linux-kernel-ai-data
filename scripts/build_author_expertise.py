#!/usr/bin/env python3
"""
Author expertise scoring based on commit frequency and maintainer status.

Creates 'author_expertise' table with reputation scores.
"""

import logging
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from rich.console import Console

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Known top maintainers (can be expanded by parsing MAINTAINERS file)
TOP_MAINTAINERS = {
    'torvalds@linux-foundation.org': 100,
    'gregkh@linuxfoundation.org': 95,
    'akpm@linux-foundation.org': 90,
    'davem@davemloft.net': 85,
    'tglx@linutronix.de': 85,
    'mingo@kernel.org': 85,
    'peterz@infradead.org': 80,
    'viro@zeniv.linux.org.uk': 80,
    'hch@lst.de': 75,
    'axboe@kernel.dk': 75,
}

def get_client():
    return Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )

def build_author_expertise():
    client = get_client()
    
    console.print("[yellow]Creating author_expertise table...[/yellow]")
    client.execute("DROP TABLE IF EXISTS author_expertise")
    client.execute("""
        CREATE TABLE author_expertise (
            author_email String,
            commit_count UInt32,
            subsystems Array(String),
            first_commit DateTime,
            last_commit DateTime,
            expertise_score Float32,
            is_maintainer UInt8
        ) ENGINE = MergeTree() ORDER BY author_email
    """)
    
    # Calculate author statistics
    console.print("[bold cyan]Calculating author statistics...[/bold cyan]")
    
    authors = client.execute("""
        SELECT 
            author_email,
            count() as commit_count,
            min(author_date) as first_commit,
            max(author_date) as last_commit
        FROM commits
        GROUP BY author_email
        HAVING commit_count > 1
        ORDER BY commit_count DESC
    """)
    
    console.print(f"Found {len(authors):,} authors with 2+ commits")
    
    # Calculate expertise score based on:
    # - Commit count (log scale)
    # - Tenure (years active)
    # - Maintainer status
    
    import math
    from datetime import datetime
    
    insert_data = []
    
    for email, count, first, last in authors:
        # Base score from commit count (log scale, max ~50 points)
        count_score = min(50, math.log10(count + 1) * 20)
        
        # Tenure score (years active, max 30 points)
        if isinstance(first, datetime) and isinstance(last, datetime):
            tenure_years = (last - first).days / 365.0
            tenure_score = min(30, tenure_years * 3)
        else:
            tenure_score = 0
        
        # Maintainer bonus (20 points)
        is_maintainer = 1 if email.lower() in TOP_MAINTAINERS else 0
        maintainer_score = TOP_MAINTAINERS.get(email.lower(), 0) * 0.2 if is_maintainer else 0
        
        total_score = count_score + tenure_score + maintainer_score
        
        insert_data.append({
            'author_email': email,
            'commit_count': count,
            'subsystems': [],  # Would need another query to populate
            'first_commit': first if isinstance(first, datetime) else datetime.now(),
            'last_commit': last if isinstance(last, datetime) else datetime.now(),
            'expertise_score': round(total_score, 2),
            'is_maintainer': is_maintainer
        })
    
    # Batch insert
    console.print("[yellow]Inserting expertise data...[/yellow]")
    
    batch_size = 10000
    for i in range(0, len(insert_data), batch_size):
        batch = insert_data[i:i+batch_size]
        client.execute(
            'INSERT INTO author_expertise (author_email, commit_count, subsystems, first_commit, last_commit, expertise_score, is_maintainer) VALUES',
            batch
        )
        console.print(f"Inserted {min(i+batch_size, len(insert_data)):,}/{len(insert_data):,}")
    
    # Show top authors
    console.print("\n[green]Author expertise table created![/green]")
    console.print("\n[bold]Top 10 Authors by Expertise:[/bold]")
    
    top = client.execute("""
        SELECT author_email, commit_count, expertise_score, is_maintainer
        FROM author_expertise
        ORDER BY expertise_score DESC
        LIMIT 10
    """)
    
    for email, count, score, is_maint in top:
        maint_badge = "👑" if is_maint else ""
        console.print(f"  {email}: {score:.1f} ({count:,} commits) {maint_badge}")

if __name__ == "__main__":
    build_author_expertise()
