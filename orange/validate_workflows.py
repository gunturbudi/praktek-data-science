"""
validate_workflows.py — check the generated .ows files.

Two levels of checking:

  STRUCTURAL (always runs, no Orange needed)
      * XML parses and has the elements Orange's reader expects
      * links refer to existing nodes, are enabled, and are not self-links
      * every node has a <properties> entry and is connected to something
      * pickled File settings decode, name Orange's real RecentPath class,
        and point at a dataset that exists on disk

  SEMANTIC (runs only if Orange 3 is importable)
      * every widget's qualified_name resolves to a real widget class
      * every source_channel is a real output of that widget
      * every sink_channel is a real input of that widget
      * a channel receiving several links is genuinely multi-input

Usage:  python validate_workflows.py
"""
from __future__ import annotations

import base64
import os
import pickletools
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = Path(__file__).parent
OWS_DIR = HERE.parent          # workflows live at the practicum root
DATA = (HERE.parent / "data").resolve()


# ------------------------------------------------------------- Orange layer
def try_import_orange():
    try:
        import Orange.widgets  # noqa: F401
        return True
    except Exception:                                           # noqa: BLE001
        return False


HAVE_ORANGE = try_import_orange()
_cache: dict = {}


def widget_class(qualified_name: str):
    if qualified_name not in _cache:
        import importlib
        mod, cls = qualified_name.rsplit(".", 1)
        _cache[qualified_name] = getattr(importlib.import_module(mod), cls)
    return _cache[qualified_name]


def signals(widget, side: str) -> dict:
    """All Input/Output descriptors, including inherited ones."""
    container = getattr(widget, side, None)
    if container is None:
        return {}
    found = {}
    for klass in reversed(container.__mro__):
        for value in vars(klass).values():
            if hasattr(value, "name"):
                found[value.name] = value
    return found


# ---------------------------------------------------------------- structure
def structural(path: Path, root: ET.Element) -> list[str]:
    errs: list[str] = []

    if root.tag != "scheme":
        errs.append(f"root element is <{root.tag}>, expected <scheme>")
    if not root.get("title"):
        errs.append("scheme has no title")

    nodes = {n.get("id"): n for n in root.findall("./nodes/node")}
    if not nodes:
        errs.append("no <node> elements")

    for nid, n in nodes.items():
        for attr in ("name", "qualified_name", "title", "position"):
            if not n.get(attr):
                errs.append(f"node {nid} missing @{attr}")
        if not (n.get("qualified_name") or "").startswith("Orange.widgets."):
            errs.append(f"node {nid}: odd qualified_name "
                        f"{n.get('qualified_name')!r}")

    linked = set()
    for ln in root.findall("./links/link"):
        s, k = ln.get("source_node_id"), ln.get("sink_node_id")
        if s not in nodes:
            errs.append(f"link {ln.get('id')}: source {s} is not a node")
        if k not in nodes:
            errs.append(f"link {ln.get('id')}: sink {k} is not a node")
        if s == k:
            errs.append(f"link {ln.get('id')} is self-referential")
        if ln.get("enabled") != "true":
            errs.append(f"link {ln.get('id')} is disabled")
        linked.update({s, k})

    for nid, n in nodes.items():
        if nid not in linked:
            errs.append(f"node {nid} ({n.get('title')}) is not connected")

    props = {p.get("node_id"): p
             for p in root.findall("./node_properties/properties")}
    for nid in nodes:
        if nid not in props:
            errs.append(f"node {nid} has no <properties>")

    for nid, p in props.items():
        fmt = p.get("format")
        if fmt not in ("literal", "pickle"):
            errs.append(f"properties {nid}: unknown format {fmt!r}")
            continue
        if fmt == "literal":
            import ast
            try:
                ast.literal_eval(p.text or "{}")
            except Exception as exc:                            # noqa: BLE001
                errs.append(f"properties {nid}: bad literal ({exc})")
            continue

        try:
            raw = base64.b64decode(p.text or "")
            ops = list(pickletools.genops(raw))
        except Exception as exc:                                # noqa: BLE001
            errs.append(f"properties {nid}: undecodable pickle ({exc})")
            continue

        cls = (nodes[nid].get("qualified_name") or "").rsplit(".", 1)[-1]
        if cls == "OWFile":
            globals_used = [a for op, a, _ in ops
                            if op.name in ("GLOBAL", "STACK_GLOBAL")]
            if not any("RecentPath" in str(g) for g in globals_used):
                errs.append(f"File node {nid}: settings do not reference "
                            f"RecentPath (globals: {globals_used})")
            paths = [a for op, a, _ in ops
                     if isinstance(a, str) and (a.endswith(".csv")
                                                or a.endswith(".tab"))]
            if not paths:
                errs.append(f"File node {nid}: no dataset path in settings")

            # The pickle holds BOTH an absolute path and a basedir-relative
            # one. Resolve relative entries against the .ows directory, which
            # is what Orange itself uses as basedir.
            for pth in paths:
                candidate = Path(pth)
                if not candidate.is_absolute():
                    candidate = (path.parent / candidate)
                if not candidate.exists():
                    errs.append(f"File node {nid}: missing dataset {pth}")

            # Portability: at least one entry must be basedir-relative, or the
            # workflow only works on the machine that created it.
            if not any(not Path(pth).is_absolute() for pth in paths):
                errs.append(f"File node {nid}: no basedir-relative path stored "
                            f"-- this workflow is NOT portable")
    return errs


# ----------------------------------------------------------------- semantic
def semantic(root: ET.Element) -> list[str]:
    errs: list[str] = []
    nodes = {n.get("id"): n for n in root.findall("./nodes/node")}

    resolved = {}
    for nid, n in nodes.items():
        q = n.get("qualified_name")
        try:
            resolved[nid] = widget_class(q)
        except Exception as exc:                                # noqa: BLE001
            errs.append(f"node {nid} ({n.get('title')}): cannot resolve "
                        f"{q} -- {type(exc).__name__}: {exc}")

    for ln in root.findall("./links/link"):
        s, k = ln.get("source_node_id"), ln.get("sink_node_id")
        if s not in resolved or k not in resolved:
            continue
        sc, kc = ln.get("source_channel"), ln.get("sink_channel")

        outs = signals(resolved[s], "Outputs")
        ins = signals(resolved[k], "Inputs")
        if sc not in outs:
            errs.append(f"{nodes[s].get('title')!r} has no output {sc!r} "
                        f"(has {sorted(outs)})")
        if kc not in ins:
            errs.append(f"{nodes[k].get('title')!r} has no input {kc!r} "
                        f"(has {sorted(ins)})")

    counts = Counter((ln.get("sink_node_id"), ln.get("sink_channel"))
                     for ln in root.findall("./links/link"))
    for (nid, ch), cnt in counts.items():
        if cnt <= 1 or nid not in resolved:
            continue
        desc = signals(resolved[nid], "Inputs").get(ch)
        # Input.single is False when the channel accepts multiple connections
        if desc is not None and getattr(desc, "single", True):
            errs.append(f"{nodes[nid].get('title')!r} input {ch!r} receives "
                        f"{cnt} links but accepts only one")
    return errs


def main() -> int:
    files = sorted(OWS_DIR.glob("*.ows"))
    if not files:
        print("no .ows files found -- run build_workflows.py first")
        return 1

    print(f"semantic checks: {'ENABLED (Orange found)' if HAVE_ORANGE else 'skipped (Orange not importable)'}\n")

    total = 0
    for f in files:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError as exc:
            print(f"  FAIL {f.name:<34} XML does not parse: {exc}")
            total += 1
            continue

        errs = structural(f, root)
        if HAVE_ORANGE:
            errs += semantic(root)

        n_nodes = len(root.findall("./nodes/node"))
        n_links = len(root.findall("./links/link"))
        print(f"  {'OK  ' if not errs else 'FAIL'} {f.name:<34} "
              f"{n_nodes:>2} widgets, {n_links:>2} links")
        for e in errs:
            print(f"         - {e}")
        total += len(errs)

    print()
    if total:
        print(f"{total} problem(s) found.")
        return 1
    print(f"All {len(files)} workflows validate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
