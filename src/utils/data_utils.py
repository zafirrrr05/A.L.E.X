# src/utils/data_utils.py

import numpy as np


def is_valid_sequence(npz_path: str,
                      min_t1: int = 9,
                      min_t2: int = 5) -> bool:
    """
    Returns True if sequence has enough valid players
    to be real match footage.

    Thresholds based on data analysis:
      - Warmup clips:  t2 = 3-5  (camera shows only one team)
      - Match clips:   t2 = 7-11 (both teams on pitch)
      - t1 is almost always 11, but require 9 to allow tracking drops

    min_t1 = 9  — allows 2 tracking drops from team1
    min_t2 = 7  — clear separator between warmup (3-5) and match (8-11)
    """
    d  = np.load(npz_path)
    t1 = d["players_team1"]
    t2 = d["players_team2"]
    v1 = int((~np.all(t1 == 0, axis=(0, 2))).sum())
    v2 = int((~np.all(t2 == 0, axis=(0, 2))).sum())
    return v1 >= min_t1 and v2 >= min_t2