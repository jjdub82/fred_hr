import os
from dotenv import load_dotenv

load_dotenv()

def get_fred_api_key() -> str:
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FRED_API_KEY not set. Create a .env file or set an environment variable.")
    return key
