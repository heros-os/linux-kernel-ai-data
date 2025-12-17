#!/usr/bin/env python3
"""Show database statistics."""
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

# Get counts
total = client.execute("SELECT count() FROM commits")[0][0]
heuristic = client.execute("SELECT count() FROM commits WHERE heuristic_score > 0")[0][0]
ai_scored = client.execute("SELECT count() FROM commits WHERE ai_quality_score > 0")[0][0]
with_reason = client.execute("SELECT count() FROM commits WHERE length(ai_score_reason) > 5")[0][0]
skipped = client.execute("SELECT count() FROM commits WHERE ai_quality_score = -1")[0][0]

# By tier
premium = client.execute("SELECT count() FROM commits WHERE heuristic_score >= 90")[0][0]
high_q = client.execute("SELECT count() FROM commits WHERE heuristic_score >= 70")[0][0]
standard = client.execute("SELECT count() FROM commits WHERE heuristic_score >= 50")[0][0]

print("=" * 50)
print(" DATABASE STATISTICS")
print("=" * 50)
print(f"\n[COMMITS]")
print(f"  Total commits:        {total:,}")
print(f"  Heuristic scored:     {heuristic:,}")
print(f"  AI scored (>0):       {ai_scored:,}")
print(f"  With AI reasoning:    {with_reason:,}")
print(f"  Skipped (no diff):    {skipped:,}")

print(f"\n[QUALITY TIERS]")
print(f"  Premium (>=90):       {premium:,}")
print(f"  High Quality (>=70):  {high_q:,}")
print(f"  Standard (>=50):      {standard:,}")

# Show AI scoring progress
pending = client.execute("SELECT count() FROM commits WHERE heuristic_score >= 30 AND (ai_quality_score = 0 OR ai_quality_score IS NULL) AND (ai_score_reason IS NULL OR ai_score_reason = '')")[0][0]
print(f"\n[AI SCORING PROGRESS]")
print(f"  Completed:            {with_reason:,}")
print(f"  Pending (score>=30):  {pending:,}")
pct = (with_reason / (with_reason + pending) * 100) if (with_reason + pending) > 0 else 0
print(f"  Progress:             {pct:.1f}%")
