"""Pre-clustering signal filter: removes unwanted AE hits based on configurable rules.

Reads filter_config.json (from repo root or a custom path) and builds a boolean
mask over rec_data rows. Signals matching ANY exclude rule are masked out.
"""
import json
import os
import numpy as np
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, 'filter_config.json')


def load_filter_config(path: Optional[str] = None) -> dict:
    p = path or _DEFAULT_CONFIG
    if not os.path.exists(p):
        return {'filters': []}
    with open(p) as f:
        return json.load(f)


def build_filter_mask(rec_data, config: Optional[dict] = None, config_path: Optional[str] = None) -> np.ndarray:
    """Return a boolean mask (True = keep) over rec_data rows."""
    if config is None:
        config = load_filter_config(config_path)

    n = len(rec_data)
    keep = np.ones(n, dtype=bool)
    field_map = config.get('field_name_map', {})
    filters = config.get('filters', [])
    names = rec_data.dtype.names

    removed_counts = []

    for rule in filters:
        field_name = rule.get('field', '')
        raw_field = field_map.get(field_name, field_name)

        if raw_field not in names:
            continue

        vals = rec_data[raw_field].astype(float)
        rule_mask = np.ones(n, dtype=bool)

        if 'exclude_values' in rule:
            for ev in rule['exclude_values']:
                rule_mask &= vals != float(ev)

        if 'min' in rule:
            rule_mask &= vals >= float(rule['min'])

        if 'max' in rule:
            rule_mask &= vals <= float(rule['max'])

        removed = int(np.sum(keep & ~rule_mask))
        if removed > 0:
            removed_counts.append((field_name, rule, removed))

        keep &= rule_mask

    total_removed = n - int(np.sum(keep))
    if total_removed > 0:
        details = "; ".join(f"{f}: -{c}" for f, _, c in removed_counts)
        print(f"      [filter] removed {total_removed}/{n} signals ({details})")

    return keep
