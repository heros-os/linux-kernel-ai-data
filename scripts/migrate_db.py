import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings

def migrate_db():
    print(f"Connecting to {settings.clickhouse.host}:{settings.clickhouse.port}...")
    client = Client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        database=settings.clickhouse.database
    )
    
    # 1. Add columns to commits table
    columns = [
        ("heuristic_score", "Float32 DEFAULT 0.0"),
        ("ai_quality_score", "Nullable(Float32)"),
        ("quality_flags", "UInt32 DEFAULT 0")
    ]
    
    for col_name, col_type in columns:
        try:
            client.execute(f"ALTER TABLE commits ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
            print(f"Added column {col_name} to commits")
        except Exception as e:
            print(f"Error adding {col_name}: {e}")

    # 2. Add columns to file_changes table
    try:
        client.execute("ALTER TABLE file_changes ADD COLUMN IF NOT EXISTS relevance_score Float32 DEFAULT 0.0")
        print("Added column relevance_score to file_changes")
    except Exception as e:
        print(f"Error adding relevance_score: {e}")

if __name__ == "__main__":
    migrate_db()
