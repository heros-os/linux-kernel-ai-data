#!/usr/bin/env python3
"""
Database initialization script.

Initializes the ClickHouse database and creates required tables.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.writer import init_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize ClickHouse database for kernel history"
    )
    parser.add_argument(
        "--host",
        default=settings.clickhouse.host,
        help="ClickHouse host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.clickhouse.port,
        help="ClickHouse native port"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Initializing database on {args.host}:{args.port}")
    
    try:
        init_database(host=args.host, port=args.port)
        logger.info("Database initialization complete!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
