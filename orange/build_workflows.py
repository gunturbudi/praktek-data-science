"""
build_workflows.py — generate the Orange 3 workflow files (.ows) for weeks 1-7.

    python build_workflows.py

An .ows file is XML: <nodes>, <links>, <annotations>, and a <node_properties>
block holding each widget's settings as a literal-eval'd or pickled blob. We
write only the settings that matter pedagogically and let Orange fill in the
rest from its defaults, which keeps the files readable and version-stable.

Widget vocabulary is deliberately restricted to what the students have already
seen in the Orange "Introduction to Data Science" video series.
"""
from __future__ import annotations

import base64
import os
import pickle
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).parent

# The .ows files are written to the practicum ROOT, one level above this
# script, and that placement is load-bearing rather than cosmetic.
#
# Orange makes a stored path portable by recording it relative to "basedir" --
# the directory containing the .ows file. But RecentPath.create() only assigns
# that prefix when the dataset lives *inside* basedir. With workflows in
# practicum/orange/ the data folder is a SIBLING, so the prefix is silently
# dropped the first time Orange rewrites the path, and the workflow becomes
# absolute-path-only again.
#
# With the workflows at practicum/ root, data/ is a CHILD of basedir, the
# prefix survives, and the file keeps working after Orange re-saves it.
OUT = HERE.parent
DATA = (HERE.parent / "data").resolve()

# ---------------------------------------------------------------- registry
# short name -> (qualified widget class, display title, project name)
W = {
    "file":        ("Orange.widgets.data.owfile.OWFile", "File"),
    "datasets":    ("Orange.widgets.data.owdatasets.OWDataSets", "Datasets"),
    "table":       ("Orange.widgets.data.owtable.OWTable", "Data Table"),
    "featstats":   ("Orange.widgets.data.owfeaturestatistics.OWFeatureStatistics",
                    "Feature Statistics"),
    "selectcols":  ("Orange.widgets.data.owselectcolumns.OWSelectAttributes",
                    "Select Columns"),
    "selectrows":  ("Orange.widgets.data.owselectrows.OWSelectRows", "Select Rows"),
    "impute":      ("Orange.widgets.data.owimpute.OWImpute", "Impute"),
    "outliers":    ("Orange.widgets.data.owoutliers.OWOutliers", "Outliers"),
    "preprocess":  ("Orange.widgets.data.owpreprocess.OWPreprocess", "Preprocess"),
    "continuize":  ("Orange.widgets.data.owcontinuize.OWContinuize", "Continuize"),
    "discretize":  ("Orange.widgets.data.owdiscretize.OWDiscretize", "Discretize"),
    "editdomain":  ("Orange.widgets.data.oweditdomain.OWEditDomain", "Edit Domain"),
    "merge":       ("Orange.widgets.data.owmergedata.OWMergeData", "Merge Data"),
    "groupby":     ("Orange.widgets.data.owgroupby.OWGroupBy", "Group by"),
    "aggregate":   ("Orange.widgets.data.owpivot.OWPivot", "Pivot Table"),
    "sampler":     ("Orange.widgets.data.owdatasampler.OWDataSampler",
                    "Data Sampler"),
    "save":        ("Orange.widgets.data.owsave.OWSave", "Save Data"),
    "rank":        ("Orange.widgets.data.owrank.OWRank", "Rank"),
    "corr":        ("Orange.widgets.data.owcorrelations.OWCorrelations",
                    "Correlations"),
    "distributions": ("Orange.widgets.visualize.owdistributions.OWDistributions",
                      "Distributions"),
    "scatter":     ("Orange.widgets.visualize.owscatterplot.OWScatterPlot",
                    "Scatter Plot"),
    "boxplot":     ("Orange.widgets.visualize.owboxplot.OWBoxPlot", "Box Plot"),
    "violin":      ("Orange.widgets.visualize.owviolinplot.OWViolinPlot",
                    "Violin Plot"),
    "barplot":     ("Orange.widgets.visualize.owbarplot.OWBarPlot", "Bar Plot"),
    "lineplot":    ("Orange.widgets.visualize.owlineplot.OWLinePlot", "Line Plot"),
    "sieve":       ("Orange.widgets.visualize.owsieve.OWSieveDiagram",
                    "Sieve Diagram"),
    "mosaic":      ("Orange.widgets.visualize.owmosaic.OWMosaicDisplay",
                    "Mosaic Display"),
    "heatmap":     ("Orange.widgets.visualize.owheatmap.OWHeatMap", "Heat Map"),
    "pca":         ("Orange.widgets.unsupervised.owpca.OWPCA", "PCA"),
    "treeviewer":  ("Orange.widgets.visualize.owtreeviewer.OWTreeGraph",
                    "Tree Viewer"),
    "nomogram":    ("Orange.widgets.visualize.ownomogram.OWNomogram", "Nomogram"),
    "tree":        ("Orange.widgets.model.owtree.OWTreeLearner", "Tree"),
    "logreg":      ("Orange.widgets.model.owlogisticregression."
                    "OWLogisticRegression", "Logistic Regression"),
    "linreg":      ("Orange.widgets.model.owlinearregression.OWLinearRegression",
                    "Linear Regression"),
    "knn":         ("Orange.widgets.model.owknn.OWKNNLearner", "kNN"),
    "svm":         ("Orange.widgets.model.owsvm.OWSVM", "SVM"),
    "nb":          ("Orange.widgets.model.ownaivebayes.OWNaiveBayes",
                    "Naive Bayes"),
    "forest":      ("Orange.widgets.model.owrandomforest.OWRandomForest",
                    "Random Forest"),
    "constant":    ("Orange.widgets.model.owconstant.OWConstant", "Constant"),
    "testscore":   ("Orange.widgets.evaluate.owtestandscore.OWTestAndScore",
                    "Test and Score"),
    "confusion":   ("Orange.widgets.evaluate.owconfusionmatrix.OWConfusionMatrix",
                    "Confusion Matrix"),
    "roc":         ("Orange.widgets.evaluate.owrocanalysis.OWROCAnalysis",
                    "ROC Analysis"),
    "liftcurve":   ("Orange.widgets.evaluate.owliftcurve.OWLiftCurve",
                    "Lift Curve"),
    "predictions": ("Orange.widgets.evaluate.owpredictions.OWPredictions",
                    "Predictions"),
    "calibration": ("Orange.widgets.evaluate.owcalibrationplot."
                    "OWCalibrationPlot", "Calibration Plot"),
    "pythonscript": ("Orange.widgets.data.owpythonscript.OWPythonScript",
                     "Python Script"),
}


# ------------------------------------------------- widget name resolution
# A few widget classes have been renamed across Orange releases. The most
# consequential for us is the Data Table: it is `owtable.OWTable` in current
# Orange but was `owtable.OWDataTable` in older versions, and a workflow naming
# the wrong one fails to load with "UnknownWidgetDefinition".
#
# When Orange is importable we resolve each name against the installed version
# and fall back to the known alias, so a regenerated workflow always matches
# whatever the instructor actually has.
ALIASES = {
    "Orange.widgets.data.owtable.OWTable": [
        "Orange.widgets.data.owtable.OWDataTable",
    ],
}


def resolve_widget(qualified_name: str) -> str:
    """Return a qualified name that exists in the installed Orange, if we can tell."""
    try:
        import importlib
    except ImportError:                                          # pragma: no cover
        return qualified_name

    def exists(name):
        mod, cls = name.rsplit(".", 1)
        try:
            return hasattr(importlib.import_module(mod), cls)
        except Exception:                                        # noqa: BLE001
            return False

    if not _ORANGE_PRESENT:
        return qualified_name          # cannot check; trust the default
    if exists(qualified_name):
        return qualified_name
    for alt in ALIASES.get(qualified_name, []):
        if exists(alt):
            print(f"  note: {qualified_name} not found in this Orange; "
                  f"using {alt}")
            return alt
    print(f"  WARNING: {qualified_name} does not exist in the installed "
          f"Orange and no alias is known")
    return qualified_name


def _blob(obj) -> str:
    """Orange stores widget settings as a base64 pickle."""
    return base64.b64encode(pickle.dumps(obj, protocol=2)).decode("ascii")


# ---------------------------------------------------------- widget settings
def python_script_settings(name: str, script: str) -> dict:
    """Pre-load a script into the Python Script widget so it runs as shipped."""
    return {
        "__version__": 2,
        "currentScriptIndex": 0,
        "scriptLibrary": [{"name": name, "script": script, "filename": None}],
        "scriptText": script,
        "splitterState": None,
        "vimModeEnabled": False,
        "controlAreaVisible": True,
        "savedWidgetGeometry": None,
    }


def preprocess_normalize_settings() -> dict:
    """Preprocess widget with 'Normalize Features' already enabled.

    Week 5's whole point is comparing PCA with and without scaling, so the
    scaled branch must actually be scaled when the file is opened.
    """
    return {
        "__version__": 2,
        "autocommit": True,
        "storedsettings": {
            "preprocessors": [
                # (qualname, params) -- scale to mean 0 / sd 1
                # method 2 = Scale.NormalizeBySD (mean 0, sd 1)
                ("orange.preprocess.scale", {"method": 2}),
            ]
        },
        "controlAreaVisible": True,
        "savedWidgetGeometry": None,
    }


SCRIPTS = HERE / "scripts"


def load_script(filename: str) -> str:
    """Read an embedded script from scripts/, so it stays independently runnable."""
    path = SCRIPTS / filename
    if not path.exists():
        raise SystemExit(f"missing embedded script: {path}")
    return path.read_text(encoding="utf-8")


class Workflow:
    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
        self.nodes: list[dict] = []
        self.links: list[tuple] = []
        self.notes: list[dict] = []
        self._id = 0

    def node(self, kind: str, x: int, y: int, title: str | None = None,
             settings: dict | None = None) -> int:
        qname, default_title = W[kind]
        qname = resolve_widget(qname)
        nid = self._id
        self._id += 1
        self.nodes.append({
            "id": nid,
            "name": default_title,
            "qualified_name": qname,
            "title": title or default_title,
            "position": f"({x}.0, {y}.0)",
            "settings": settings or {},
        })
        return nid

    def link(self, src: int, sink: int, src_ch: str = "Data",
             sink_ch: str = "Data"):
        self.links.append((src, sink, src_ch, sink_ch))

    def note(self, x: int, y: int, w: int, h: int, text: str):
        self.notes.append({"rect": f"({x}.0, {y}.0, {w}.0, {h}.0)",
                           "text": text})

    def to_xml(self) -> str:
        out = ['<?xml version="1.0" encoding="utf-8"?>']
        out.append(
            f'<scheme version="2.0" title="{escape(self.title)}" '
            f'description="{escape(self.description)}">')

        out.append("\t<nodes>")
        for n in self.nodes:
            out.append(
                f'\t\t<node id="{n["id"]}" name="{escape(n["name"])}" '
                f'qualified_name="{n["qualified_name"]}" '
                f'project_name="Orange3" version="" '
                f'title="{escape(n["title"])}" position="{n["position"]}" />')
        out.append("\t</nodes>")

        out.append("\t<links>")
        for i, (src, sink, sc, kc) in enumerate(self.links):
            out.append(
                f'\t\t<link id="{i}" source_node_id="{src}" '
                f'sink_node_id="{sink}" enabled="true" '
                f'source_channel="{escape(sc)}" sink_channel="{escape(kc)}" />')
        out.append("\t</links>")

        out.append("\t<annotations>")
        for i, a in enumerate(self.notes):
            out.append(
                f'\t\t<text id="{i}" type="text/plain" rect="{a["rect"]}" '
                f'font-family="Helvetica" font-size="13">'
                f'{escape(a["text"])}</text>')
        out.append("\t</annotations>")

        out.append("\t<thumbnail />")

        out.append("\t<node_properties>")
        for n in self.nodes:
            fmt = "literal" if not n["settings"] else "pickle"
            if not n["settings"]:
                out.append(
                    f'\t\t<properties node_id="{n["id"]}" format="literal">'
                    f'{{}}</properties>')
            else:
                out.append(
                    f'\t\t<properties node_id="{n["id"]}" format="pickle">'
                    f'{_blob(n["settings"])}</properties>')
        out.append("\t</node_properties>")
        out.append("</scheme>")
        return "\n".join(out) + "\n"

    def save(self, filename: str):
        path = OUT / filename
        path.write_text(self.to_xml(), encoding="utf-8")
        print(f"  wrote {filename:<44} {len(self.nodes):>2} widgets, "
              f"{len(self.links):>2} links")


# --------------------------------------------------------- RecentPath shim
# Orange stores the File widget's chosen file as a list of RecentPath objects.
# Pickle serialises instances *by class reference* (module + qualified name)
# plus the instance __dict__, so if we make our stand-in claim Orange's module
# path, Orange unpickles it into a genuine RecentPath and the File widget opens
# already pointing at the right dataset.
#
# If Orange really is installed we use the real class and skip the shim.
def _install_recentpath():
    try:
        from Orange.widgets.utils.filedialogs import RecentPath as _RP
        return _RP, True
    except ImportError:
        pass

    import sys
    import types

    class RecentPath:
        def __init__(self, abspath, prefix=None, relpath="", title="",
                     sheet="", file_format=None):
            self.abspath = abspath
            self.prefix = prefix
            self.relpath = relpath
            self.title = title
            self.sheet = sheet
            self.file_format = file_format

    RecentPath.__module__ = "Orange.widgets.utils.filedialogs"
    RecentPath.__qualname__ = "RecentPath"

    # Register the (empty) package chain so pickle's importability check passes.
    for name in ("Orange", "Orange.widgets", "Orange.widgets.utils",
                 "Orange.widgets.utils.filedialogs"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []           # mark as a package
            sys.modules[name] = mod
    sys.modules["Orange.widgets.utils.filedialogs"].RecentPath = RecentPath
    return RecentPath, False


RecentPath, _ORANGE_PRESENT = _install_recentpath()


# OWFile stores its settings under this version number. Writing a different
# one sends the settings through migrate_settings on load.
OWFILE_SETTINGS_VERSION = 1


def file_settings(name: str) -> dict:
    """Settings for an OWFile widget pointing at one of our datasets.

    The path is stored TWICE, deliberately:

      * `abspath` -- used verbatim when it exists, i.e. on this machine;
      * `prefix="basedir"` + `relpath` -- used when it does not.

    Orange's RecentPath.resolve() falls back to `<basedir>/<relpath>`, where
    basedir is the directory containing the .ows file. That is what makes the
    workflow survive being copied to a student's machine, where the absolute
    path is meaningless.

    Note that Orange's own RecentPath.create() cannot produce this: it only
    assigns a prefix when the data sits *inside* basedir, and our datasets are
    in a sibling folder (orange/../data). So we construct it by hand.
    """
    abspath = str((DATA / name).resolve())
    # relative to the folder the .ows file lives in
    relpath = os.path.relpath(abspath, start=str(OUT.resolve()))
    return {
        "__version__": OWFILE_SETTINGS_VERSION,
        "recent_paths": [RecentPath(abspath, "basedir", relpath)],
        "recent_urls": [],
        "source": 0,                 # 0 = local file
        "sheet_names": {},
        "url": "",
        "variables": [],
        "controlAreaVisible": True,
        "savedWidgetGeometry": None,
    }


# ===================================================================== W1
def week01():
    wf = Workflow(
        "Week 1 — Orientation: the Orange canvas",
        "Load a dataset, look at it, and meet the widget/channel idea.")

    wf.note(20, 20, 470, 96,
            "PRAKTIKUM 1 — Foundations\n\n"
            "Widgets communicate over CHANNELS. Double-click a link to see "
            "which channel it uses.\nStart at File, then follow the arrows. "
            "Every widget updates the moment its input changes.")

    f = wf.node("file", 60, 180, "File: credit.csv",
                settings=file_settings("credit.csv"))
    t = wf.node("table", 250, 120, "Data Table")
    fs = wf.node("featstats", 250, 210, "Feature Statistics")
    d = wf.node("distributions", 250, 300, "Distributions")

    wf.link(f, t)
    wf.link(f, fs)
    wf.link(f, d)

    wf.note(440, 110, 330, 128,
            "WHAT TO DO\n"
            "1. Open Data Table. How many rows and columns?\n"
            "2. Open Feature Statistics. Which column has the most\n"
            "   missing values? Which are identifiers?\n"
            "3. Open Distributions and set the variable to 'default'.\n"
            "   What is the base rate? Remember it for week 7.")
    wf.save("week01_orientation.ows")


# ===================================================================== W2
def week02():
    wf = Workflow(
        "Week 2 — Data acquisition and wrangling",
        "Profile, impute, remove outliers, join a reference table, save.")

    wf.note(20, 20, 520, 82,
            "PRAKTIKUM 2 — Wrangling\n\n"
            "The pipeline runs left to right. ORDER MATTERS: profile first, "
            "then impute,\nthen handle outliers. Compare the Data Table at "
            "each stage.")

    f = wf.node("file", 60, 190, "File: credit.csv",
                settings=file_settings("credit.csv"))
    fs = wf.node("featstats", 220, 110, "1. Profile")
    imp = wf.node("impute", 220, 210, "2. Impute")
    out = wf.node("outliers", 390, 210, "3. Outliers")
    t1 = wf.node("table", 390, 310, "After impute")
    t2 = wf.node("table", 560, 310, "After outliers")
    sel = wf.node("selectrows", 560, 190, "4. Select Rows")
    sv = wf.node("save", 720, 190, "5. Save Data")

    wf.link(f, fs)
    wf.link(f, imp)
    wf.link(imp, out)
    wf.link(imp, t1)
    wf.link(out, t2, src_ch="Inliers")
    wf.link(out, sel, src_ch="Inliers")
    wf.link(sel, sv, src_ch="Matching Data")

    wf.note(760, 90, 300, 176,
            "WHAT TO DO\n"
            "1. In Impute, try 'Average/Most frequent', then\n"
            "   'Model-based imputer'. Compare in Data Table.\n"
            "2. In Outliers, switch between One-Class SVM and\n"
            "   Local Outlier Factor. How many rows are dropped?\n"
            "3. In Select Rows, keep only age >= 25.\n"
            "4. Save the result as credit_clean.tab.\n\n"
            "QUESTION: is dropping the outliers here defensible?\n"
            "Justify in one sentence -- see book section 2.7.2.")
    wf.save("week02_wrangling.ows")


# ===================================================================== W3
def week03():
    wf = Workflow(
        "Week 3 — Exploratory data analysis",
        "Distributions, correlations, scatter plots, and a Simpson reversal.")

    wf.note(20, 20, 540, 82,
            "PRAKTIKUM 3 — EDA\n\n"
            "Correlations outputs a FEATURES channel as well as Data. That is "
            "how the\nScatter Plot below knows which pair to show. "
            "Double-click the link to inspect it.")

    f = wf.node("file", 60, 200, "File: rumah_yogya.tab",
                settings=file_settings("rumah_yogya.tab"))
    dist = wf.node("distributions", 240, 100, "Distributions")
    corr = wf.node("corr", 240, 200, "Correlations")
    sc = wf.node("scatter", 430, 200, "Scatter Plot")
    box = wf.node("boxplot", 240, 300, "Box Plot")
    heat = wf.node("heatmap", 430, 320, "Heat Map")
    ft = wf.node("featstats", 430, 100, "Feature Statistics")

    wf.link(f, dist)
    wf.link(f, corr)
    wf.link(f, box)
    wf.link(f, ft)
    wf.link(f, sc)
    wf.link(corr, sc, src_ch="Features", sink_ch="Features")
    wf.link(f, heat)

    wf.note(620, 90, 330, 190,
            "WHAT TO DO\n"
            "1. Distributions: set variable to 'harga'. Is it skewed?\n"
            "   Now split by 'kecamatan'.\n"
            "2. Correlations: which pair is strongest? Click it and\n"
            "   watch the Scatter Plot follow.\n"
            "3. Box Plot: 'harga' grouped by 'sertifikat'.\n"
            "4. SIMPSON HUNT: in Box Plot, put 'harga' by\n"
            "   'kecamatan', then add a Subgroup. Does any\n"
            "   comparison reverse once you condition?\n\n"
            "Compare everything with notebook week03_eda.ipynb.")
    wf.save("week03_eda.ows")


# ===================================================================== W4
def week04():
    wf = Workflow(
        "Week 4 — Inference and A/B testing",
        "Distributions by group, sieve/mosaic for association, "
        "and a t-test via Python Script.")

    wf.note(20, 20, 560, 82,
            "PRAKTIKUM 4 — Inference\n\n"
            "Orange is a visual tool, not a statistics package. It shows "
            "association clearly\n(Sieve, Mosaic, Distributions) and defers "
            "the formal test to the Python Script.")

    f = wf.node("file", 60, 200, "File: ab_test.tab",
                settings=file_settings("ab_test.tab"))
    dist = wf.node("distributions", 240, 110, "Distributions")
    box = wf.node("boxplot", 240, 200, "Box Plot")
    sieve = wf.node("sieve", 240, 290, "Sieve Diagram")
    mos = wf.node("mosaic", 430, 290, "Mosaic Display")
    py = wf.node("pythonscript", 430, 110, "Python Script: t-test",
                  settings=python_script_settings(
                      "A/B test", load_script("ab_test.py")))
    t = wf.node("table", 620, 110, "Test output")

    wf.link(f, dist)
    wf.link(f, box)
    wf.link(f, sieve)
    wf.link(f, mos)
    wf.link(f, py)
    wf.link(py, t)

    wf.note(620, 200, 340, 210,
            "WHAT TO DO\n"
            "1. Distributions: 'converted', split by 'variant'.\n"
            "   Eyeball the difference -- is it convincing?\n"
            "2. Box Plot: 'session_seconds' by 'variant'. Orange\n"
            "   prints a t-test at the bottom of this widget.\n"
            "3. Sieve Diagram: variant x converted. The colour\n"
            "   shows the deviation from independence.\n"
            "4. SIMPSON CHECK: Mosaic with variant, converted\n"
            "   AND device. Does the lift hold in both devices?\n"
            "5. Python Script: paste the snippet from the book\n"
            "   (section 4.8.2) to get z, p and the CI.")
    wf.save("week04_inference.ows")


# ===================================================================== W5
def week05():
    wf = Workflow(
        "Week 5 — Feature engineering and PCA",
        "Rank features, continuize, project with PCA, and see leakage.")

    wf.note(20, 20, 560, 82,
            "PRAKTIKUM 5 — Features and PCA\n\n"
            "PCA REQUIRES SCALING. Run it once with Preprocess in the chain "
            "and once\nwithout, and compare the variance explained. That "
            "difference is the whole lesson.")

    f = wf.node("file", 60, 220, "File: credit.tab",
                settings=file_settings("credit.tab"))
    rank = wf.node("rank", 240, 110, "Rank")
    prep = wf.node("preprocess", 240, 220, "Preprocess: normalise",
                   settings=preprocess_normalize_settings())
    pca_s = wf.node("pca", 420, 180, "PCA (scaled)")
    pca_u = wf.node("pca", 420, 300, "PCA (UNSCALED)")
    sc1 = wf.node("scatter", 600, 180, "PC1 vs PC2 (scaled)")
    sc2 = wf.node("scatter", 600, 300, "PC1 vs PC2 (unscaled)")
    t = wf.node("table", 420, 60, "Ranked features")

    wf.link(f, rank)
    wf.link(rank, t, src_ch="Reduced Data")
    wf.link(f, prep)
    wf.link(prep, pca_s, src_ch="Preprocessed Data")
    wf.link(f, pca_u)
    wf.link(pca_s, sc1, src_ch="Transformed Data")
    wf.link(pca_u, sc2, src_ch="Transformed Data")

    wf.note(780, 100, 330, 210,
            "WHAT TO DO\n"
            "1. Rank: which features score highest by information\n"
            "   gain? Compare with the notebook's mutual info.\n"
            "2. Preprocess: enable 'Normalize Features'.\n"
            "3. Open BOTH PCA widgets. How many components\n"
            "   reach 95% variance in each?\n"
            "   Scaled: ~? Unscaled: ~1-2. Why?\n"
            "4. Compare the two scatter plots. The unscaled one\n"
            "   is essentially a plot of one variable.\n\n"
            "See book section 5.6.4, limitation 2.")
    wf.save("week05_features_pca.ows")


# ===================================================================== W6
def week06():
    wf = Workflow(
        "Week 6 — Regression",
        "Linear regression on house prices; logistic regression with "
        "a nomogram on credit.")

    wf.note(20, 20, 560, 82,
            "PRAKTIKUM 6 — Regression\n\n"
            "TOP ROW: predicting a continuous target (house price).\n"
            "BOTTOM ROW: predicting a binary target (default). The Nomogram "
            "is the widget\nthat makes logistic regression readable to a "
            "non-technical stakeholder.")

    # --- regression on rumah -----------------------------------------
    f1 = wf.node("file", 60, 170, "File: rumah_yogya.tab",
                 settings=file_settings("rumah_yogya.tab"))
    lin = wf.node("linreg", 250, 130, "Linear Regression")
    ts1 = wf.node("testscore", 430, 170, "Test and Score")
    pred1 = wf.node("predictions", 620, 130, "Predictions")
    sc = wf.node("scatter", 620, 220, "Predicted vs actual")

    wf.link(f1, lin, sink_ch="Data")
    wf.link(f1, ts1)
    wf.link(lin, ts1, src_ch="Learner", sink_ch="Learner")
    wf.link(f1, pred1)
    wf.link(lin, pred1, src_ch="Model", sink_ch="Predictors")
    wf.link(pred1, sc, src_ch="Predictions")

    # --- classification on credit ------------------------------------
    f2 = wf.node("file", 60, 400, "File: credit.tab",
                 settings=file_settings("credit.tab"))
    log = wf.node("logreg", 250, 360, "Logistic Regression")
    ts2 = wf.node("testscore", 430, 400, "Test and Score")
    cm = wf.node("confusion", 620, 350, "Confusion Matrix")
    roc = wf.node("roc", 620, 440, "ROC Analysis")
    nom = wf.node("nomogram", 430, 500, "Nomogram")

    wf.link(f2, log, sink_ch="Data")
    wf.link(f2, ts2)
    wf.link(log, ts2, src_ch="Learner", sink_ch="Learner")
    wf.link(ts2, cm, src_ch="Evaluation Results", sink_ch="Evaluation Results")
    wf.link(ts2, roc, src_ch="Evaluation Results", sink_ch="Evaluation Results")
    wf.link(log, nom, src_ch="Model", sink_ch="Classifier")

    wf.note(800, 140, 330, 250,
            "WHAT TO DO\n"
            "1. Test and Score (top): note RMSE, MAE and R2.\n"
            "   Set 'Cross validation, 5 folds'.\n"
            "2. Predicted-vs-actual scatter: is the spread even\n"
            "   across the range, or does it fan out? That fan is\n"
            "   heteroscedasticity (book section 6.4.1).\n"
            "3. Logistic Regression: try Ridge (L2) with C=1,\n"
            "   then Lasso (L1). Watch coefficients vanish.\n"
            "4. NOMOGRAM: this is the payoff. Each feature is a\n"
            "   line; the length of its bar is its influence.\n"
            "   Drag the blue markers to score an applicant.\n"
            "5. Confusion Matrix: which error does the model\n"
            "   make more often? Is that the cheap one?")
    wf.save("week06_regression.ows")


# ===================================================================== W7
def week07():
    wf = Workflow(
        "Week 7 — Classification: a fair comparison",
        "Five learners on identical folds, with ROC, lift and "
        "a confusion matrix.")

    wf.note(20, 20, 620, 96,
            "PRAKTIKUM 7 — Classification\n\n"
            "THE POINT OF THIS WORKFLOW: every learner meets Test and Score "
            "on the SAME\nfolds with the SAME preprocessing. That is what "
            "makes the comparison fair.\n"
            "Constant is the do-nothing baseline -- if a model cannot beat "
            "it, it has learned nothing.")

    f = wf.node("file", 60, 300, "File: credit.tab",
                settings=file_settings("credit.tab"))

    const = wf.node("constant", 260, 90, "Constant (baseline)")
    log = wf.node("logreg", 260, 165, "Logistic Regression")
    tree = wf.node("tree", 260, 240, "Tree")
    nb = wf.node("nb", 260, 315, "Naive Bayes")
    knn = wf.node("knn", 260, 390, "kNN")
    svm = wf.node("svm", 260, 465, "SVM")

    ts = wf.node("testscore", 500, 280, "Test and Score")
    cm = wf.node("confusion", 700, 170, "Confusion Matrix")
    roc = wf.node("roc", 700, 260, "ROC Analysis")
    lift = wf.node("liftcurve", 700, 350, "Lift Curve")
    cal = wf.node("calibration", 700, 440, "Calibration Plot")

    tv = wf.node("treeviewer", 500, 500, "Tree Viewer")
    misc = wf.node("table", 890, 170, "Misclassified rows")

    for learner in (const, log, tree, nb, knn, svm):
        wf.link(f, learner, sink_ch="Data")
        wf.link(learner, ts, src_ch="Learner", sink_ch="Learner")
    wf.link(f, ts)

    for w in (cm, roc, lift, cal):
        wf.link(ts, w, src_ch="Evaluation Results",
                sink_ch="Evaluation Results")
    wf.link(tree, tv, src_ch="Model", sink_ch="Tree")
    wf.link(cm, misc, src_ch="Selected Data")

    wf.note(890, 260, 340, 270,
            "WHAT TO DO\n"
            "1. Test and Score: 'Cross validation, 5 folds'.\n"
            "   Sort by AUC. Does every model beat Constant?\n"
            "2. Set Target class = 'yes' (the minority class).\n"
            "   Watch precision and recall change completely.\n"
            "3. Confusion Matrix: select the FALSE NEGATIVE cell\n"
            "   and look at those rows in the Data Table.\n"
            "   What do the missed defaulters have in common?\n"
            "4. Tree Viewer: set depth to 3. Can you read the\n"
            "   rules aloud? That is why trees survive in\n"
            "   regulated domains.\n"
            "5. Calibration Plot: compare Naive Bayes with\n"
            "   Logistic Regression. NB hugs the extremes --\n"
            "   its probabilities are not trustworthy.\n\n"
            "UTS PREP: be able to explain every number here.")
    wf.save("week07_classification.ows")


if __name__ == "__main__":
    print(f"Generating Orange workflows into {OUT}")
    print(f"(datasets expected at {DATA})\n")
    week01()
    week02()
    week03()
    week04()
    week05()
    week06()
    week07()
    print("\nDone. Open any .ows with:  orange-canvas file.ows")
