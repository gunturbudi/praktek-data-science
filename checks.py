"""
checks.py — self-check helpers for the practicum notebooks.

Usage inside a notebook:

    import checks
    jawaban = ...              # your answer
    checks.task("2.3", jawaban)

`checks.task` never raises. If the answer is missing it says so; if it is wrong
it says what it expected. This lets a notebook run top-to-bottom even when the
tasks have not been attempted yet.
"""
from __future__ import annotations

import math

OK = "✅"
NO = "❌"
WAIT = "⏳"


# --------------------------------------------------------------- comparators
def _num_close(got, want, tol):
    try:
        return abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        return False


def _approx(want, tol=0.01, note=""):
    def check(got):
        if _num_close(got, want, tol):
            return True, note or f"{got} is within {tol} of the expected value."
        return False, f"expected about {want} (+/- {tol}), got {got!r}."
    return check


def _one_of(options, note=""):
    opts = [str(o).strip().lower() for o in options]

    def check(got):
        if str(got).strip().lower() in opts:
            return True, note
        return False, f"expected one of {options}, got {got!r}."
    return check


def _rows(want, tol=0):
    def check(got):
        n = len(got) if hasattr(got, "__len__") else got
        if _num_close(n, want, tol):
            return True, f"{n:,} rows."
        return False, f"expected {want:,} rows (+/- {tol}), got {n:,}."
    return check


def _cols_contain(names, note=""):
    want = set(names)

    def check(got):
        try:
            have = set(got.columns)
        except AttributeError:
            return False, "that does not look like a DataFrame."
        missing = want - have
        if missing:
            return False, f"missing column(s): {sorted(missing)}."
        return True, note or f"all of {sorted(want)} present."
    return check


def _predicate(fn, msg, note=""):
    def check(got):
        try:
            return (True, note) if fn(got) else (False, msg)
        except Exception as exc:                                  # noqa: BLE001
            return False, f"could not evaluate ({type(exc).__name__}: {exc})."
    return check


# ------------------------------------------------------------------ registry
CHECKS = {
    # ---------------------------------------------------------------- week 1
    "1.1": _approx(0.9737, 0.02,
                   "That is the iris smoke-test accuracy - environment is sound."),
    "1.2": _one_of(["prediction", "prediksi"],
                   "Right: 'which 200 students' is a ranking/prediction problem."),
    "1.3": _approx(99.5, 0.05,
                   "Exactly - the do-nothing model beats the colleague's 99.4%."),

    # ---------------------------------------------------------------- week 2
    "2.1": _rows(5000, 0),
    "2.2": _approx(120, 0, "120 duplicated order_id rows, as injected."),
    "2.3": _cols_contain(["unit_price_was_missing"],
                         "Good - the missingness indicator survives imputation."),
    "2.4": _predicate(lambda d: d["quantity"].min() > 0,
                      "there are still non-positive quantities; the -999 sentinel "
                      "has not been dealt with."),
    "2.5": _approx(30, 0,
                   "30 orders belong to customers absent from customers.csv."),

    # ---------------------------------------------------------------- week 3
    "3.1": _predicate(lambda v: float(v) > 1.0,
                      "skewness of a price-like variable should be clearly "
                      "positive (> 1)."),
    "3.2": _predicate(lambda v: abs(float(v)) < 0.35,
                      "after a log transform the skewness should be near zero."),
    "3.3": _one_of(["spearman"],
                   "Spearman - it measures monotone association, not linear."),

    # ---------------------------------------------------------------- week 4
    "4.1": _approx(0.1558, 0.01,
                   "About 16% of flagged transactions are genuinely fraudulent."),
    "4.2": _predicate(lambda v: float(v) < 0.05,
                      "the p-value should be well below 0.05 - this test is "
                      "adequately powered."),
    "4.3": _approx(0.0072, 0.002,
                   "The absolute lift is about 0.72 percentage points."),
    "4.4": _predicate(lambda v: float(v) > 0.15,
                      "with daily peeking the false-positive rate should be far "
                      "above the nominal 5%."),

    # ---------------------------------------------------------------- week 5
    "5.1": _rows(2000, 0),
    "5.2": _cols_contain(["dti", "pti", "util"],
                         "The three ratios are the features that matter here."),
    "5.3": _predicate(lambda v: float(v) > 0.80,
                      "a leaky selection on pure noise should report an "
                      "impressively high AUC - that is the whole point."),
    "5.4": _predicate(lambda v: 0.40 <= float(v) <= 0.65,
                      "the honest pipeline should sit close to chance (~0.5)."),

    # ---------------------------------------------------------------- week 6
    "6.1": _predicate(lambda v: float(v) > 0.60,
                      "R^2 on the log-price model should comfortably exceed 0.60."),
    "6.2": _one_of(["ridge", "lasso", "elasticnet", "elastic net"]),
    "6.3": _approx(0.0909, 0.01,
                   "1/(1+10) = 0.09 - far from the default 0.5."),

    # ---------------------------------------------------------------- week 7
    "7.1": _approx(0.1200, 0.005, "Information gain = 0.420 - 0.300 = 0.120."),
    "7.2": _predicate(lambda v: float(v) < 1.5,
                      "in 1000 dimensions the farthest/nearest ratio collapses "
                      "towards 1."),
    "7.3": _predicate(lambda v: float(v) > 0.78,
                      "a reasonable model on the credit data should reach "
                      "ROC-AUC above 0.78."),
}


def task(tid: str, value=None):
    """Check one task. Prints a verdict; never raises."""
    checker = CHECKS.get(tid)
    if checker is None:
        print(f"{WAIT} No checker registered for task {tid}.")
        return
    if value is None:
        print(f"{WAIT} Task {tid}: not attempted yet "
              f"(assign your answer, then re-run this cell).")
        return
    try:
        ok, note = checker(value)
    except Exception as exc:                                      # noqa: BLE001
        print(f"{NO} Task {tid}: check failed to run "
              f"({type(exc).__name__}: {exc}).")
        return
    if ok:
        print(f"{OK} Task {tid}: correct. {note}".rstrip())
    else:
        print(f"{NO} Task {tid}: not right yet - {note}")


def hint(text: str):
    """Print a hint block."""
    print("\U0001f4a1 " + text)
