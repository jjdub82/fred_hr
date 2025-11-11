from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt

def observations_to_df(obs: List[Dict]) -> pd.DataFrame:
    if not obs:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(obs)[["date", "value"]]
    # numeric coercion: Fred uses strings, '.' for missing
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def plot_df(df: pd.DataFrame, title: str):
    plt.figure()
    plt.plot(df["date"], df["value"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    return plt.gcf()
