"""Shared API key lookup: environment variables first, then Streamlit secrets.

This lets the same code run locally (via .env / os.environ) and on Streamlit
Community Cloud (via st.secrets), without every module needing its own fallback logic.
"""

import os


def get_api_key(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get(name)
    except (ImportError, FileNotFoundError):
        return None
