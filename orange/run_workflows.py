"""
run_workflows.py — actually EXECUTE each .ows headlessly and prove data flows.

Loading a scheme only proves the XML is well formed. This goes further: it
instantiates every widget, lets the signal manager propagate, and then asks each
widget what it actually received and emitted. That is the only check that
catches a File widget whose stored path no longer resolves, a learner that
errors on the data, or a link that is legal but carries nothing.

    python run_workflows.py                 # run every workflow
    python run_workflows.py week03          # run one
    python run_workflows.py --moved         # simulate a relocated course folder

`--moved` is the important one for course material: it rewrites each stored path
to a location that does not exist, then checks the widget still finds its
dataset through Orange's `basedir` mechanism. A workflow that only works on the
machine that authored it is not usable by students.

Requires Orange 3 with a Qt binding. Runs fully offscreen.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OWS_DIR = HERE.parent          # workflows live at the practicum root


def main() -> int:
    args = [a for a in sys.argv[1:]]
    simulate_move = "--moved" in args
    args = [a for a in args if not a.startswith("--")]
    only = args[0] if args else None

    from AnyQt.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(["run_workflows"])

    import importlib
    from xml.etree import ElementTree as ET
    from orangecanvas.registry import WidgetRegistry, WidgetDescription, \
        CategoryDescription
    from orangecanvas.scheme.readwrite import scheme_load
    from orangewidget.workflow.widgetsscheme import WidgetsScheme

    files = sorted(OWS_DIR.glob("*.ows"))
    if only:
        files = [f for f in files if f.stem.startswith(only)]
    if not files:
        print("no matching .ows files")
        return 1

    # ---- build a registry containing exactly the widgets we use -----------
    qnames = set()
    for f in files:
        for n in ET.parse(f).getroot().findall("./nodes/node"):
            qnames.add(n.get("qualified_name"))

    registry = WidgetRegistry()
    registry.register_category(CategoryDescription(name="Course"))
    for q in sorted(qnames):
        mod, cls = q.rsplit(".", 1)
        widget = getattr(importlib.import_module(mod), cls)
        desc = WidgetDescription(**widget.get_widget_description())
        desc.category = "Course"
        registry.register_widget(desc)

    if simulate_move:
        print("Simulating a relocated course folder: every stored absolute\n"
              "path is rewritten to one that does not exist, so the workflow\n"
              "must recover via the basedir-relative path.\n")

    total_problems = 0
    for f in files:
        problems = run_one(f, registry, WidgetsScheme, scheme_load, app,
                           simulate_move)
        total_problems += len(problems)
        print(f"  {'OK  ' if not problems else 'FAIL'} {f.name}")
        for p in problems:
            print(f"         - {p}")

    print()
    if total_problems:
        print(f"{total_problems} problem(s) found.")
        return 1
    print(f"All {len(files)} workflows executed and produced data.")
    return 0


def run_one(path, registry, WidgetsScheme, scheme_load, app, simulate_move,
            timeout=90.0):
    from AnyQt.QtCore import QCoreApplication, QEventLoop
    problems: list[str] = []

    scheme = WidgetsScheme()
    # basedir is what makes stored relative paths resolvable
    scheme.set_runtime_env("basedir", str(path.parent))
    with open(path, "rb") as fh:
        scheme_load(scheme, fh, registry=registry)

    if simulate_move:
        _break_absolute_paths(scheme)

    # Let widgets instantiate and signals propagate.
    #
    # OWFile defers its actual read with QTimer.singleShot(0, load_data), so
    # "the signal manager has nothing pending" is NOT a sufficient stopping
    # condition -- checking it alone reports a false failure before the timer
    # has even fired. Spin until every File widget holds data AND the manager
    # is idle, with a wall-clock timeout as the backstop.
    import time

    manager = scheme.signal_manager
    file_nodes = [n for n in scheme.nodes
                  if n.description.qualified_name.endswith("OWFile")]

    def files_loaded():
        return all(getattr(scheme.widget_for_node(n), "data", None) is not None
                   for n in file_nodes)

    deadline = time.time() + timeout
    while time.time() < deadline:
        QCoreApplication.processEvents(QEventLoop.AllEvents, 30)
        if files_loaded() and not manager.has_pending():
            break

    # give downstream widgets a further chance to consume what arrived
    settle = time.time() + 3.0
    while time.time() < settle:
        QCoreApplication.processEvents(QEventLoop.AllEvents, 30)
        if not manager.has_pending():
            break

    # ---- did anything actually flow? -------------------------------------
    for node in scheme.nodes:
        widget = scheme.widget_for_node(node)
        # any widget reporting an error is a hard failure
        err = getattr(widget, "Error", None)
        if err is not None:
            active = [m for m in dir(err)
                      if not m.startswith("_")
                      and hasattr(getattr(err, m, None), "is_shown")
                      and getattr(err, m).is_shown()]
            if active:
                problems.append(f"{node.title}: error(s) active {active}")

        # File widgets must have produced a table
        if node.description.qualified_name.endswith("OWFile"):
            data = getattr(widget, "data", None)
            if data is None:
                problems.append(f"{node.title}: produced NO data "
                                f"(stored path did not resolve)")
            else:
                n_rows = len(data)
                if n_rows == 0:
                    problems.append(f"{node.title}: produced an empty table")

    # ---- did downstream widgets receive it? ------------------------------
    sinks = {}
    for link in scheme.links:
        sinks.setdefault(link.sink_node, []).append(link)
    for node, links in sinks.items():
        widget = scheme.widget_for_node(node)
        got = _received_anything(widget)
        if got is False:
            problems.append(f"{node.title}: received nothing from "
                            f"{links[0].source_node.title}")

    scheme.clear()
    return problems


def _break_absolute_paths(scheme):
    """Point every File widget's abspath at a location that does not exist."""
    for node in scheme.nodes:
        if not node.description.qualified_name.endswith("OWFile"):
            continue
        widget = scheme.widget_for_node(node)
        from Orange.widgets.utils.filedialogs import RecentPath
        new = []
        for rp in widget.recent_paths:
            new.append(RecentPath(
                str(Path("Z:/definitely/not/here") / Path(rp.abspath).name),
                rp.prefix, rp.relpath, file_format=rp.file_format))
        widget.recent_paths = new
        widget._relocate_recent_files()
        widget.load_data()


def _received_anything(widget):
    """Best-effort: did this widget end up holding some input?

    Returns True/False/None (None = cannot tell for this widget type).
    """
    for attr in ("data", "dataset", "_data", "results", "learner",
                 "classifier", "model", "tree", "corpus"):
        if hasattr(widget, attr) and getattr(widget, attr) is not None:
            return True
    # Widgets that aggregate several inputs keep dicts
    for attr in ("learners", "_learners", "predictors", "_inputs"):
        val = getattr(widget, attr, None)
        if val:
            return True
    return None


if __name__ == "__main__":
    sys.exit(main())
