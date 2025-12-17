import pandas as pd
from pathlib import Path
from typing import List, Dict

def parse_csv(file_path: str) -> List[Dict]:
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    chunks = []
    for i, row in df.iterrows():
        text = " | ".join([f"{k}: {row[k]}" for k in df.columns])
        chunks.append({
            "text": text,
            "metadata": {
                "source": Path(file_path).name,
                "row": int(i),
                "type": "csv"
            }
        })
    return chunks

if __name__ == "__main__":
    out = parse_csv("data/raw/claims.csv")
    print(out[0])
