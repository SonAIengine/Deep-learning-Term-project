"""Load frozen var_binding_tier1.jsonl for circuit analysis."""
import os
import json
from typing import List, Dict, Any

from shared.config import DATASETS_DIR
from tracks.code.config import TIER1_FILE


def load_tier1(path: str = None) -> List[Dict[str, Any]]:
    """Load Tier 1 cf pairs from JSONL.

    Returns list of records with keys:
    - id, cf_id, role (clean/corrupt)
    - prompt, answer, answer_token_id
    - distractor_answer_token_ids
    - source_var_token_pos (int or None)
    - n_vars, target_var, source_var, distractor_vars
    """
    if path is None:
        path = os.path.join(DATASETS_DIR, TIER1_FILE)

    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def get_cf_pairs(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by cf_id into {cf_id: [clean, corrupt]}.

    Assumes each cf_id has exactly 2 records (clean + corrupt).
    """
    pairs = {}
    for rec in records:
        cf_id = rec["cf_id"]
        if cf_id not in pairs:
            pairs[cf_id] = []
        pairs[cf_id].append(rec)

    # Verify each pair has 2 records
    for cf_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"cf_id {cf_id} has {len(pair)} records, expected 2")

        # Sort so clean is first
        pair.sort(key=lambda x: x["role"] == "corrupt")

    return pairs


def sample_pairs(pairs: Dict[str, List[Dict[str, Any]]],
                 n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Sample first n cf pairs for initial testing."""
    cf_ids = list(pairs.keys())[:n]
    return {cf_id: pairs[cf_id] for cf_id in cf_ids}
