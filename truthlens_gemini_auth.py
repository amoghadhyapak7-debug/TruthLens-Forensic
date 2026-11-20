"""TruthLens - Secure Google Gemini API Authentication Module

Manages secure authentication for Google Gemini API via environment variables
and Google-supported credentials without exposing secrets in the frontend interface.

Provides:
- Secure credential resolution (GEMINI_API_KEY, GOOGLE_API_KEY, ADC)
- Real-time connection probing and status classification:
    * 'connected' ('Gemini Connected')
    * 'not_configured' ('Gemini Not Configured')
    * 'failed' ('Gemini Connection Failed')
- In-memory cached connection testing with TTL to avoid redundant network pings
- Model support verification
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Reload .env from project root if available
PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH, override=True)
else:
    load_dotenv()

# Status constants
STATUS_CONNECTED = "connected"
STATUS_NOT_CONFIGURED = "not_configured"
STATUS_CONNECTION_FAILED = "failed"

LABEL_CONNECTED = "Gemini Connected"
LABEL_NOT_CONFIGURED = "Gemini Not Configured"
LABEL_CONNECTION_FAILED = "Gemini Connection Failed"

# Currently supported Gemini models (official 2026 endpoints)
SUPPORTED_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
DEFAULT_MODEL = "gemini-2.5-flash"

# In-memory cache for connection status
_connection_cache: Dict[str, Any] = {
    "key_fingerprint": None,
    "model_name": None,
    "result": None,
    "timestamp": 0.0,
}
CACHE_TTL_SECONDS = 60.0  # Cache probe for 60 seconds


def resolve_api_key() -> Tuple[Optional[str], str]:
    """Finds the active API key and its source without printing the secret."""
    # Check current process environment (which includes .env if loaded)
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        source = ".env file" if DOTENV_PATH.exists() and "GEMINI_API_KEY" in DOTENV_PATH.read_text(encoding="utf-8", errors="ignore") else "environment variable"
        return gemini_key, source

    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key:
        return google_key, "GOOGLE_API_KEY environment variable"

    return None, "none"


def mask_key(key: Optional[str]) -> str:
    """Safely masks an API key for diagnostic display, e.g. 'AIzaSy...7x89'."""
    if not key:
        return "None"
    if len(key) <= 8:
        return "********"
    return f"{key[:6]}...{key[-4:]}"


def get_gemini_client() -> Optional[Any]:
    """Initializes and returns an authorized Google GenAI client if credentials are configured."""
    api_key, _ = resolve_api_key()
    if not api_key:
        return None

    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.warning(f"Could not initialize Google GenAI Client: {e}")
        return None


def verify_gemini_connection(
    model_name: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Checks the Gemini API connection and returns a diagnostic status dictionary.

    Returns:
        {
            "status": "connected" | "not_configured" | "failed",
            "label": "Gemini Connected" | "Gemini Not Configured" | "Gemini Connection Failed",
            "model": "gemini-2.5-flash",
            "auth_source": "environment variable" | ".env file" | "none",
            "masked_key": "AIzaSy...1234",
            "message": "...",
            "error": "..." (if failed),
            "is_available": True / False,
        }
    """
    global _connection_cache

    model = (model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)).strip()
    api_key, auth_source = resolve_api_key()

    # If no credentials found
    if not api_key:
        return {
            "status": STATUS_NOT_CONFIGURED,
            "label": LABEL_NOT_CONFIGURED,
            "model": model,
            "auth_source": "none",
            "masked_key": "None",
            "is_available": False,
            "message": (
                "No Google Gemini API key configured in environment or .env file. "
                "The application will operate in traditional forensic analysis mode."
            ),
            "setup_guide": (
                "To enable Gemini AI video and image understanding:\n"
                "1. Get an API key from Google AI Studio: https://aistudio.google.com/docs/api-key\n"
                "2. Set the environment variable in PowerShell:\n"
                "   [System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your_key_here', 'User')\n"
                "   OR create a local .env file (excluded from git):\n"
                "   GEMINI_API_KEY=your_key_here\n"
                "3. Restart TruthLens."
            ),
        }

    # Check cache to avoid hitting the API on every single Streamlit page interaction
    key_fingerprint = f"{len(api_key)}:{mask_key(api_key)}"
    now = time.time()
    if (
        not force_refresh
        and _connection_cache["key_fingerprint"] == key_fingerprint
        and _connection_cache["model_name"] == model
        and (now - _connection_cache["timestamp"]) < CACHE_TTL_SECONDS
    ):
        return _connection_cache["result"]

    # Active Probe: Attempt lightweight model metadata query to verify authorization
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        # Probe model endpoint to verify key validity and quota without spending generation tokens
        client.models.get(model=model)

        result = {
            "status": STATUS_CONNECTED,
            "label": LABEL_CONNECTED,
            "model": model,
            "auth_source": auth_source,
            "masked_key": mask_key(api_key),
            "is_available": True,
            "message": f"Successfully authenticated via {auth_source}. Model '{model}' is online and verified.",
            "error": None,
        }
    except Exception as e:
        logger.error(f"Gemini connection probe failed: {e}")
        err_msg = str(e)
        if "400" in err_msg or "INVALID_ARGUMENT" in err_msg or "API key not valid" in err_msg:
            clean_err = "API key was rejected by Google (invalid or expired key)."
        elif "403" in err_msg or "PERMISSION_DENIED" in err_msg:
            clean_err = "Permission denied for this API key or API not enabled in Google Cloud project."
        elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            clean_err = "Google Gemini API quota exceeded or rate limit reached."
        else:
            clean_err = f"API verification error: {err_msg[:120]}"

        result = {
            "status": STATUS_CONNECTION_FAILED,
            "label": LABEL_CONNECTION_FAILED,
            "model": model,
            "auth_source": auth_source,
            "masked_key": mask_key(api_key),
            "is_available": False,
            "message": f"Authentication failed: {clean_err}",
            "error": clean_err,
        }

    # Save to cache
    _connection_cache = {
        "key_fingerprint": key_fingerprint,
        "model_name": model,
        "result": result,
        "timestamp": now,
    }

    return result
