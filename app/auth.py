"""Cognito JWT authentication."""
import os
from typing import Dict, Any

import requests
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


security = HTTPBearer()

# Cache for JWKS
_jwks_cache: Dict[str, Any] | None = None


def get_jwks() -> Dict[str, Any]:
    """Fetch and cache Cognito JWKS."""
    global _jwks_cache

    if _jwks_cache is None:
        region = os.getenv("COGNITO_REGION", "us-west-2")
        pool_id = os.getenv("COGNITO_POOL_ID")

        if not pool_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="COGNITO_POOL_ID not configured",
            )

        jwks_url = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/jwks.json"
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()

    return _jwks_cache


def get_signing_key(token: str) -> Dict[str, Any]:
    """Get the signing key for a token from JWKS."""
    jwks = get_jwks()
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unable to find signing key",
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT and return user_id."""
    token = credentials.credentials

    # For development: allow bypass with dev token
    if os.getenv("DEV_MODE") == "true" and token.startswith("dev_"):
        return token.replace("dev_", "")

    try:
        signing_key = get_signing_key(token)

        region = os.getenv("COGNITO_REGION", "us-west-2")
        pool_id = os.getenv("COGNITO_POOL_ID")
        issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"

        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_aud": False},  # Cognito doesn't always set aud
        )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: no subject",
            )

        return user_id

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )
