import requests
from typing import Dict, Any, List
from .config import get_fred_api_key

BASE = "https://api.stlouisfed.org/fred"

def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = {**params, "api_key": get_fred_api_key(), "file_type": "json"}
    r = requests.get(f"{BASE}/{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_series_observations(series_id: str, **kwargs) -> List[Dict[str, Any]]:
    data = _get("series/observations", {"series_id": series_id, **kwargs})
    return data.get("observations", [])

def search_series(text: str, limit: int = 10) -> List[Dict[str, Any]]:
    data = _get("series/search", {"search_text": text, "limit": limit})
    return data.get("seriess", [])
