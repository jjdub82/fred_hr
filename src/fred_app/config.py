import os
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

FRED_API_KEY = os.getenv("FRED_API_KEY", "e935a1e42e22ab72d87c190f6ec0a0ed")
FRED_BASE = "https://api.stlouisfed.org/fred"

def get_fred_api_key() -> str:
    """
    Returns the FRED API key from environment or fallback.
    Keeps a single point of truth for all FRED requests.
    """
    return FRED_API_KEY
