#!/usr/bin/env python3
"""Check how many files are changed per commit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings

client = Client(
    host=settings.clickhouse.host,
    port=settings.clickhouse.port,
    database=settings.clickhouse.database
)

# Get file counts for premium commits
query = """
SELECT 
    c.commit_hash,
    c.subject,
    COUNT(fc.file_path) as file_count
FROM commits c
JOIN file_changes fc ON c.commit_hash = fc.commit_hash
WHERE c.heuristic_score >= 90
GROUP BY c.commit_hash, c.subject
ORDER BY file_count DESC
LIMIT 20
"""

results = client.execute(query)

print("=" * 80)
print(" FILES CHANGED PER COMMIT (Premium Tier)")
print("=" * 80)
print(f"\n{'Commit':<15} {'Files':<8} {'Subject'}")
print("-" * 80)

for hash, subject, count in results:
    print(f"{hash[:12]:<15} {count:<8} {subject[:50]}")

# Get average
avg_query = """
SELECT AVG(file_count) as avg_files
FROM (
    SELECT COUNT(*) as file_count
    FROM file_changes
    WHERE commit_hash IN (
        SELECT commit_hash FROM commits WHERE heuristic_score >= 90
    )
    GROUP BY commit_hash
)
"""

avg = client.execute(avg_query)[0][0]

print("\n" + "=" * 80)
print(f"Average files per commit: {avg:.1f}")
print("=" * 80)

# Check current export strategy
print("\n[CURRENT EXPORT STRATEGY]")
print("  Per commit: 1 file (LIMIT 1 in query)")
print("  Selection: First file matching criteria")
print("  Context: Smart extracted from that one file")
