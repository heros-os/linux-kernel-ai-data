#!/usr/bin/env python3
"""Add ai_score_reason column to commits table."""
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

print("Adding ai_score_reason column...")
client.execute("ALTER TABLE commits ADD COLUMN IF NOT EXISTS ai_score_reason String DEFAULT ''")
print("Done!")
