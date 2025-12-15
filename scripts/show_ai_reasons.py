#!/usr/bin/env python3
"""Show AI quality reasoning examples."""
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

query = """
SELECT 
    commit_hash, 
    subject, 
    ai_quality_score, 
    ai_score_reason
FROM commits 
WHERE ai_score_reason != '' 
    AND length(ai_score_reason) > 10
ORDER BY ai_quality_score DESC
LIMIT 15
"""

rows = client.execute(query)

print(f"Found {len(rows)} commits with AI reasoning:\n")
print("=" * 70)

for hash, subject, score, reason in rows:
    score_int = int(score) if score else 0
    stars = "*" * score_int + "-" * (5 - score_int)
    print(f"\n[{stars}] Score: {score_int}/5")
    print(f"Commit: {hash[:12]}")
    print(f"Subject: {subject[:60]}...")
    print(f"AI Reason: {reason}")
    print("-" * 70)
