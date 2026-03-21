"""
GitHub Device Flow authentication for the G2S application.

Usage:
    Set the GITHUB_CLIENT_ID environment variable to your GitHub OAuth App
    client ID (see .env.example).  Then call sign_in() to authenticate.
"""

import json
import os
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")

_TOKEN_DIR = Path.home() / ".g2s"
_TOKEN_FILE = _TOKEN_DIR / "token.json"

_DEVICE_CODE_URL = "https://github.com/login/device/code"
_TOKEN_URL = "https://github.com/login/oauth/access_token"
_USER_URL = "https://api.github.com/user"

# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


def _save_session(token: str, user: dict) -> None:
    """Persist the access token and user profile to disk."""
    _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(_TOKEN_FILE, "w") as fh:
        json.dump({"token": token, "user": user}, fh)


def _load_session() -> dict | None:
    """Return stored session dict or None if no session exists."""
    if _TOKEN_FILE.exists():
        try:
            with open(_TOKEN_FILE) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def clear_session() -> None:
    """Remove the stored token (sign out)."""
    if _TOKEN_FILE.exists():
        _TOKEN_FILE.unlink()


def get_current_user() -> dict | None:
    """Return the stored user profile dict, or None if not signed in."""
    session = _load_session()
    return session.get("user") if session else None


def is_signed_in() -> bool:
    """Return True if a stored access token exists."""
    return _load_session() is not None


# ---------------------------------------------------------------------------
# Device Flow
# ---------------------------------------------------------------------------


def _request_device_code() -> dict:
    """Request a device + user code from GitHub."""
    if not GITHUB_CLIENT_ID:
        raise ValueError(
            "GITHUB_CLIENT_ID is not set. "
            "Create a GitHub OAuth App and export GITHUB_CLIENT_ID."
        )
    resp = requests.post(
        _DEVICE_CODE_URL,
        json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
        headers={"Accept": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _poll_for_token(device_code: str, interval: int, expires_in: int) -> str | None:
    """
    Poll GitHub until the user approves the request or the code expires.

    Returns the access token string on success, or None on failure/expiry.
    """
    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(interval)
        resp = requests.post(
            _TOKEN_URL,
            json={
                "client_id": GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if "access_token" in data:
            return data["access_token"]

        error = data.get("error", "")
        if error == "slow_down":
            interval += 5
        elif error == "authorization_pending":
            pass  # keep polling
        else:
            # expired, access_denied, incorrect_client_credentials, etc.
            break

    return None


def _fetch_user(token: str) -> dict:
    """Fetch the authenticated user's GitHub profile."""
    resp = requests.get(
        _USER_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Public sign-in entry point
# ---------------------------------------------------------------------------


def sign_in(on_code: Callable[[str, str], None] | None = None) -> dict | None:
    """
    Run the GitHub Device Authorization Flow.

    Parameters
    ----------
    on_code:
        Optional callback invoked with ``(user_code, verification_uri)``
        once GitHub issues a device code.  Useful for showing the code in a
        custom UI.  The default behaviour opens the verification URL in the
        system browser and prints instructions to stdout.

    Returns
    -------
    dict
        The GitHub user profile on success (keys include ``login``, ``name``,
        ``avatar_url``, etc.).
    None
        If authentication fails or the device code expires.
    """
    # Return existing session if the stored token is still valid.
    session = _load_session()
    if session:
        try:
            resp = requests.get(
                _USER_URL,
                headers={"Authorization": f"Bearer {session['token']}"},
                timeout=10,
            )
            if resp.ok:
                return session.get("user")
        except requests.RequestException:
            pass
        # Token is invalid or unreachable — clear it and re-authenticate.
        clear_session()

    codes = _request_device_code()
    user_code = codes["user_code"]
    verification_uri = codes.get("verification_uri", "https://github.com/login/device")
    device_code = codes["device_code"]
    interval = int(codes.get("interval", 5))
    expires_in = int(codes.get("expires_in", 900))

    if on_code:
        on_code(user_code, verification_uri)
    else:
        print(f"\nOpen {verification_uri} in your browser")
        print(f"and enter the code:  {user_code}\n")
        webbrowser.open(verification_uri)

    token = _poll_for_token(device_code, interval, expires_in)
    if token is None:
        return None

    user = _fetch_user(token)
    _save_session(token, user)
    return user
