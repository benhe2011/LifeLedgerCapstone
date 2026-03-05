"""One-time backfill script to compute pHash for existing documents.

Usage: python -m scripts.backfill_phash
Run from the LifeLedgerCapstone directory.
"""
import asyncio
import logging
import sys

from app.db import init_db
from app.s3 import download_from_s3
from app.dedup import compute_phash

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def backfill():
    pool = await init_db()
    async with pool.acquire() as db:
        rows = await db.fetch(
            "SELECT id, s3_key FROM documents WHERE phash IS NULL ORDER BY id"
        )
        logger.info("Found %d documents to backfill", len(rows))

        success = 0
        failed = 0

        for row in rows:
            try:
                image_bytes = await download_from_s3(row["s3_key"])
                phash = compute_phash(image_bytes)
                await db.execute(
                    "UPDATE documents SET phash = $1 WHERE id = $2",
                    phash, row["id"],
                )
                success += 1
                if success % 50 == 0:
                    logger.info("Progress: %d/%d", success, len(rows))
            except Exception as e:
                logger.warning("Failed doc %d (s3_key=%s): %s", row["id"], row["s3_key"], e)
                failed += 1

        logger.info("Backfill complete: %d succeeded, %d failed", success, failed)


if __name__ == "__main__":
    asyncio.run(backfill())
