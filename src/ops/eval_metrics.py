from typing import List, Dict

def precision_at_k(relevant_ids: set, retrieved_ids: List[str], k: int = 5) -> float:
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    hits = sum(1 for i in topk if i in relevant_ids)
    return hits / len(topk)
