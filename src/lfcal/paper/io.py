"""Result files: scalars in CSV tables, arrays in `.npz`.

A record (a dict of scalars, nested dicts of scalars and short lists of scalars) is one CSV row;
nested keys are joined with `.` (`pi_gate.u_mean`, `kappas.0`, `kappas.1`), and `unflatten` turns them back
into dicts and lists.  Floats are written with `repr`, so a file reads back bit for bit; an empty
cell is a field the record does not have.  Array data (time series) go into `.npz` archives with
the same dotted names.

The force's length scale changed convention with `ode-filters` 0.8. A record with fitted hyperparameters
written since carries `length_scale = textbook`; one written before has no such field and holds the log of
the squared length scale in its `log_ell_u` fields (also `dlog_ell_u` and prefixed copies). `read_csv`
converts those records, so every reader sees the textbook length scale. `phi_fields` writes the fields
with the mark.
"""

from __future__ import annotations

import csv
import numbers
import re
from pathlib import Path

import numpy as onp

__all__ = ["flatten", "phi_fields", "phi_u_offsets", "read_csv", "read_npz", "unflatten", "write_csv", "write_npz"]

SEP = "."
_INT = re.compile(r"-?\d+")


def flatten(d, prefix="", *, sequences=True):
    """`{"a": {"b": 1}, "p": [1.0, 2.0]}` -> `{"a.b": 1, "p.0": 1.0, "p.1": 2.0}`; with
    `sequences=False` lists and tuples are leaves (arrays)."""
    out = {}
    items = d.items() if isinstance(d, dict) else enumerate(d)
    for k, v in items:
        if SEP in str(k):
            raise ValueError(f"key {k!r} contains {SEP!r}")
        key = f"{prefix}{k}"
        if isinstance(v, dict) or (sequences and isinstance(v, (list, tuple))):
            out.update(flatten(v, key + SEP, sequences=sequences))
        else:
            out[key] = v
    return out


def _nest(flat):
    """Dotted keys -> nested dicts."""
    tree = {}
    for key, v in flat.items():
        *head, last = key.split(SEP)
        node = tree
        for h in head:
            node = node.setdefault(h, {})
        node[last] = v
    return tree


def unflatten(flat):
    """Inverse of `flatten`; a level whose keys are 0, 1, ... becomes a list."""

    def lists(node):
        if not isinstance(node, dict):
            return node
        node = {k: lists(v) for k, v in node.items()}
        if node and all(_INT.fullmatch(k) for k in node):
            return [node[str(i)] for i in range(len(node))]
        return node

    return lists(_nest(flat))


def _encode(v):
    if isinstance(v, (bool, onp.bool_)):
        return "True" if v else "False"
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        return repr(float(v))
    if isinstance(v, str) and v:
        return v
    raise TypeError(f"cannot write {v!r} to a CSV cell")


def _decode(s):
    if s in ("True", "False"):
        return s == "True"
    if _INT.fullmatch(s):
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s


def write_csv(path, rows, lead=()):
    """Records -> one CSV table (the columns `lead` first, the rest in order of first appearance)."""
    flat = [flatten(r) for r in rows]
    cols = list(dict.fromkeys([*lead, *(c for r in flat for c in r)]))
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in flat:
            w.writerow([_encode(r[c]) if c in r else "" for c in cols])
    tmp.replace(path)


def _textbook_length_scale(flat):
    """A flat record with its length-scale fields in the textbook convention (see the module docstring)."""
    fields = [k for k in flat if k.endswith("log_ell_u")]
    if not fields or any(k.endswith("length_scale") for k in flat):
        return flat
    return {**flat, **{k: 0.5 * flat[k] for k in fields}, "length_scale": "textbook"}


def read_csv(path):
    """CSV table -> records (types restored, empty cells left out, dotted keys unflattened, length scales in
    the textbook convention)."""
    with Path(path).open(newline="") as f:
        rd = csv.reader(f)
        cols = next(rd)
        return [unflatten(_textbook_length_scale({c: _decode(s) for c, s in zip(cols, row, strict=True) if s != ""}))
                for row in rd]


def write_npz(path, arrays):
    """Nested dict of arrays -> one `.npz` archive with dotted names."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    onp.savez(tmp, **{k: onp.asarray(v) for k, v in flatten(arrays, sequences=False).items()})
    tmp.replace(path)


def read_npz(path, tolist=True):
    """`.npz` archive -> nested dict of arrays (as lists with `tolist`)."""
    with onp.load(path) as z:
        return _nest({k: z[k].tolist() if tolist else z[k] for k in z.files})


def phi_fields(phi):
    """The hyperparameters `phi = (log sigma_x^2, log xi_u, log ell_u)` as named fields, marked with the
    length scale's convention (see `read_csv`)."""
    return {"log_sigma_x2": float(phi[0]), "log_xi_u": float(phi[1]), "log_ell_u": float(phi[2]),
            "length_scale": "textbook"}


def phi_u_offsets(phi, log_theta_u_true):
    """The force hyperparameters minus the truth, `dlog_xi_u` and `dlog_ell_u`."""
    d = onp.asarray(phi[1:], dtype=float) - onp.asarray(log_theta_u_true, dtype=float)
    return {"dlog_xi_u": float(d[0]), "dlog_ell_u": float(d[1])}
