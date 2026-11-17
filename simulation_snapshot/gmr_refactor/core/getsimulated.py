import re
from pathlib import Path

import numpy as np


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?$")


def read_rsm_columns(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        labels = lines[index].split()
        if not labels or labels[0].upper() != "TIME":
            index += 1
            continue
        labels = [x.upper() for x in labels]
        index += 1
        values = []
        while index < len(lines):
            tokens = lines[index].split()
            if tokens and tokens[0].upper() == "TIME":
                break
            if len(tokens) == len(labels) and all(_NUMBER_RE.fullmatch(x) for x in tokens):
                values.append([float(x.replace("D", "E").replace("d", "e")) for x in tokens])
            elif values:
                break
            index += 1
        if not values:
            raise ValueError(f"No numeric rows following a TIME header in {path}")
        blocks.append((labels, np.asarray(values, dtype=np.float64)))

    if not blocks:
        raise ValueError(f"No TIME-led numeric blocks in {path}")

    time = blocks[0][1][:, 0]
    result = {"TIME": time.copy()}
    counters = {"WBHP": 0, "WGOR": 0, "WWCT": 0}
    for labels, matrix in blocks:
        if matrix.shape[0] != time.size or not np.array_equal(matrix[:, 0], time):
            raise ValueError(f"Inconsistent block time grids in {path}")
        for column, keyword in enumerate(labels):
            if keyword == "FOE":
                if "FOE" in result:
                    raise ValueError(f"Duplicate FOE in {path}")
                result["FOE"] = matrix[:, column].copy()
            elif keyword in counters:
                counters[keyword] += 1
                result[f"{keyword}{counters[keyword]}"] = matrix[:, column].copy()
    return result


def getsimulated(timestep, N, history, n_obs=19, n_producers=6, n_timesteps=82):
    if N < 1 or history < 1 or n_producers < 1:
        raise ValueError("N, history and n_producers must be positive")
    if n_obs != 1 + 3 * n_producers:
        raise ValueError("n_obs must equal 1 + 3 * n_producers")

    channels = ["FOE"] + [
        f"{k}{j}"
        for k in ("WBHP", "WGOR", "WWCT")
        for j in range(1, n_producers + 1)
    ]
    members = []
    for member in range(1, N + 1):
        path = Path(f"MASTER{member:05d}") / "MASTER0.RSM"
        columns = read_rsm_columns(path)
        missing = [name for name in channels if name not in columns]
        if missing or len(columns["TIME"]) < history:
            raise ValueError(f"Missing channels {missing} or insufficient time rows in {path}")
        members.append(np.column_stack([columns[c][:history] for c in channels]))
    return np.stack(members, axis=2)
