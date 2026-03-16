"""Upload-time deduplication using perceptual hashing."""
import io

import imagehash
from PIL import Image


def compute_phash(image_bytes: bytes) -> str:
    """Compute perceptual hash for an image. Returns hex string."""
    img = Image.open(io.BytesIO(image_bytes))
    return str(imagehash.phash(img, hash_size=16))


async def check_duplicate(db, user_id: str, phash: str) -> int | None:
    """Check if a document with the same pHash exists for this user.

    Returns the existing document ID if duplicate, None otherwise.
    """
    return await db.fetchval(
        "SELECT id FROM documents WHERE user_id = $1 AND phash = $2 LIMIT 1",
        user_id, phash,
    )
