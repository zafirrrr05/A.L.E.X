from importlib.resources import path
import os
import re
import numpy as np


def get_next_sequence_index(seq_dir):
    if not os.path.exists(seq_dir):
        return 0

    files = os.listdir(seq_dir)

    indices = []
    for f in files:
        m = re.match(r"seq_(\d+)\.npz", f)
        if m:
            indices.append(int(m.group(1)))

    if not indices:
        return 0

    return max(indices) + 1


# each sequence 
def save_sequences(sequences, out_dir, start_index=None):

    os.makedirs(out_dir, exist_ok=True)

    # ▶ automatically continue numbering
    if start_index is None:
        start_index = get_next_sequence_index(out_dir)

    saved_files = []

    for i, seq in enumerate(sequences):

        idx = start_index + i

        fname = f"seq_{idx:07d}.npz"
        path = os.path.join(out_dir, fname)

        if os.path.exists(path):
            raise RuntimeError(f"Refusing to overwrite existing file: {path}")

        np.savez_compressed(
            path,
            players_team1=seq["players_team1"],  # (T, 11, 4)
            players_team2=seq["players_team2"],  # (T, 11, 4)
            ball=seq["ball"],                    # (T, 4)
            referee=seq["referee"],              # (T, 4)
            start_frame=seq["start_frame"]       # added for debugging, can be removed later, used in testing_npz   
        )

        print("[SAVED]", path)
        saved_files.append(path)

    return saved_files