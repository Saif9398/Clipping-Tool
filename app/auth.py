"""Single-admin session auth: HMAC-signed expiry-token cookie, stdlib only.

One shared username+password from .env guards the whole app. The cookie IS the
session — `<expiry_ts>.<hmac_sha256(secret, expiry_ts)>` — so there is no
session store. Auth is OFF until ADMIN_PASSWORD (or ADMIN_PASSWORD_SHA256) is
set, keeping local dev friction-free; the deployed server sets credentials.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets as _secrets
import time

from app import config

log = logging.getLogger(__name__)

COOKIE_NAME = "clipping_tool_session"
_FAIL_DELAY_S = 0.5  # blunt online brute force on the login endpoint


def enabled() -> bool:
    return bool(config.ADMIN_PASSWORD or config.ADMIN_PASSWORD_SHA256)


def _secret() -> bytes:
    if config.SESSION_SECRET:
        return config.SESSION_SECRET.encode("utf-8")
    # Auto-generate once and persist (gitignored work/) so sessions survive
    # restarts — otherwise restarting would log the local user out.
    f = config.WORK_DIR / ".session_secret"
    try:
        return bytes.fromhex(f.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        key = _secrets.token_bytes(32)
        f.write_text(key.hex(), encoding="utf-8")
        return key


def _sign(msg: str) -> str:
    return hmac.new(_secret(), msg.encode("utf-8"), hashlib.sha256).hexdigest()


def make_token(days: float | None = None) -> str:
    days = config.SESSION_DAYS if days is None else days
    expiry = str(int(time.time() + days * 86400))
    return f"{expiry}.{_sign(expiry)}"


def verify_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    expiry, sig = token.split(".", 1)
    if not hmac.compare_digest(sig, _sign(expiry)):
        return False
    try:
        return int(expiry) > time.time()
    except ValueError:
        return False


def check_credentials(username: str, password: str) -> bool:
    """Constant-time comparison; never log or echo either value."""
    user_ok = hmac.compare_digest(username.encode("utf-8"),
                                  config.ADMIN_USERNAME.encode("utf-8"))
    if config.ADMIN_PASSWORD_SHA256:
        digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
        pass_ok = hmac.compare_digest(digest, config.ADMIN_PASSWORD_SHA256.lower())
    else:
        pass_ok = hmac.compare_digest(password.encode("utf-8"),
                                      config.ADMIN_PASSWORD.encode("utf-8"))
    if not (user_ok and pass_ok):
        time.sleep(_FAIL_DELAY_S)
        return False
    return True
