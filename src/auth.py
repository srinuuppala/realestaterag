"""Session authentication.

The brief calls for simple auth, not a user database. Credentials come from the
environment and are compared with a constant-time digest comparison — cheap to
do correctly, so there is no reason to do it badly.
"""

from __future__ import annotations

import hashlib
import hmac

import streamlit as st

from src.config import get_settings
from src.utils import get_logger

log = get_logger(__name__)

SESSION_KEY = "authenticated_user"
MAX_ATTEMPTS = 5


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def verify(username: str, password: str) -> bool:
    settings = get_settings()
    user_ok = hmac.compare_digest(_digest(username), _digest(settings.auth_username))
    password_ok = hmac.compare_digest(_digest(password), _digest(settings.auth_password))
    return user_ok and password_ok


def current_user() -> str | None:
    return st.session_state.get(SESSION_KEY)


def is_authenticated() -> bool:
    return current_user() is not None


def login(username: str, password: str) -> bool:
    if not verify(username, password):
        st.session_state["login_attempts"] = st.session_state.get("login_attempts", 0) + 1
        log.warning("Failed sign-in for %r (attempt %d)", username, st.session_state["login_attempts"])
        return False

    st.session_state[SESSION_KEY] = username
    st.session_state["login_attempts"] = 0
    log.info("User %r signed in", username)
    return True


def logout() -> None:
    for key in (SESSION_KEY, "memory", "pending_question"):
        st.session_state.pop(key, None)
    log.info("User signed out")


def attempts_remaining() -> int:
    return max(0, MAX_ATTEMPTS - st.session_state.get("login_attempts", 0))
