"""Read results/ppl_table.json and print PASS/FAIL vs the success bar.

Success bar (from design.md): best PPL across non-base entries must be
<= 0.70 * base PPL.
"""
from __future__ import annotations

import json
import pathlib
import sys


def main() -> int:
    table_path = pathlib.Path(__file__).resolve().parent.parent / "results" / "ppl_table.json"
    if not table_path.exists():
        print(f"missing {table_path}; run ./run.sh eval first", file=sys.stderr)
        return 2
    table = json.loads(table_path.read_text())
    if "base" not in table:
        print("ppl_table.json has no 'base' entry", file=sys.stderr)
        return 2
    base_ppl = table["base"]["ppl"]
    others = {k: v for k, v in table.items() if k != "base"}
    if not others:
        print("no non-base PPL entries to compare", file=sys.stderr)
        return 2
    best_label = min(others, key=lambda k: others[k]["ce"])
    best_ppl = others[best_label]["ppl"]
    ratio = best_ppl / base_ppl
    print(f"base ppl       = {base_ppl:.4f}")
    print(f"best ({best_label}) = {best_ppl:.4f}")
    print(f"ratio          = {ratio:.4f}  (success bar: <= 0.70)")
    if ratio <= 0.70:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
