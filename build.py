"""
build.py — turn the percent-format sources in _src/ into notebooks/*.ipynb.

    python build.py            # build all notebooks
    python build.py --run      # build, then execute each one to verify it runs

Percent format:  "# %% [markdown]" starts a markdown cell (content is the
commented block that follows), "# %%" starts a code cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "_src"
OUT = ROOT / "notebooks"


def parse_percent(text: str):
    """Split percent-format source into a list of (kind, source) pairs."""
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if kind == "markdown":
            # strip one leading "# " (or a bare "#") from each line
            lines = []
            for ln in src.split("\n"):
                if ln.startswith("# "):
                    lines.append(ln[2:])
                elif ln == "#":
                    lines.append("")
                else:
                    lines.append(ln)
            src = "\n".join(lines)
        if src.strip():
            cells.append((kind, src))

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def to_notebook(cells) -> dict:
    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        cell = {
            "cell_type": kind,
            "id": f"cell-{i:03d}",
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="execute each notebook after building")
    ap.add_argument("--only", default=None, help="build only this stem, e.g. week03")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    sources = sorted(SRC.glob("week*.py"))
    if args.only:
        sources = [p for p in sources if p.stem.startswith(args.only)]
    if not sources:
        print("no sources found in", SRC)
        return 1

    built = []
    for src in sources:
        cells = parse_percent(src.read_text(encoding="utf-8"))
        nb = to_notebook(cells)
        dest = OUT / (src.stem + ".ipynb")
        dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        n_md = sum(1 for k, _ in cells if k == "markdown")
        n_code = sum(1 for k, _ in cells if k == "code")
        print(f"  built {dest.name:<52} {n_md:>3} md + {n_code:>3} code cells")
        built.append(dest)

    if args.run:
        import nbformat
        from nbclient import NotebookClient
        print("\nexecuting:")
        failed = 0
        for dest in built:
            nb = nbformat.read(dest, as_version=4)
            client = NotebookClient(nb, timeout=900, kernel_name="python3",
                                    resources={"metadata": {"path": str(ROOT)}})
            try:
                client.execute()
                print(f"  OK    {dest.name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {dest.name}\n        {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:160]}")
        if failed:
            print(f"\n{failed} notebook(s) failed to execute.")
            return 1
        print("\nAll notebooks executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
